# Copyright (c) Meta Platforms, Inc. and affiliates
"""
MemoryShardedDTensor: A DTensor variant that shards storage across devices.

This module provides a memory-efficient DTensor implementation where the tensor's
storage is sharded across devices in a process group, reducing per-device memory
usage. Unlike regular DTensor sharding which affects the logical tensor view,
MemoryShardedDTensor physically partitions the underlying storage.
"""
from dataclasses import dataclass
from typing import Optional, Union
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor


@dataclass
class BlockStorageShardingSpec:
    """
    Unified specification for storage sharding (single or multi-dimensional).

    This spec supports both single-dimension sharding (FSDP v2 style) and
    multi-dimensional block sharding. Single-dim sharding is a special case
    where len(shard_dims) == 1.

    Attributes:
        orig_size: Original (full) tensor size before sharding.
        orig_stride: Original tensor stride before sharding.
        shard_dims: Tuple of tensor dimensions that are sharded.
        mesh_dims: Tuple of mesh dimension names corresponding to shard_dims.
        padded_shard_sizes: Per-dimension padded shard sizes for even division.
        actual_shard_sizes: Per-dimension actual shard sizes on this rank.
        mesh_dim_indices: Cached mesh dimension indices for performance.
    """

    orig_size: torch.Size
    orig_stride: tuple[int, ...]
    shard_dims: tuple[int, ...]
    mesh_dims: tuple[str, ...]
    padded_shard_sizes: tuple[int, ...]
    actual_shard_sizes: tuple[int, ...]
    mesh_dim_indices: tuple[int, ...]


@dataclass
class ShardParamInfo:
    """
    Tracks how a parameter maps to a rank's shard in flattened storage mode.

    In FSDP v1-style flattening, multiple parameters are concatenated into a
    single flat buffer and sharded across ranks. This class tracks which portion
    of each original parameter is present in each rank's shard.

    Attributes:
        in_shard: Whether any part of this parameter is in this rank's shard.
        offset_in_shard: Start offset within the local shard (None if not in shard).
        numel_in_shard: Number of elements from this param in the shard (None if not in shard).
        intra_param_start: Start index within the original parameter (None if not in shard).
        intra_param_end: End index (exclusive) within the original parameter (None if not in shard).
    """

    in_shard: bool
    offset_in_shard: Optional[int] = None
    numel_in_shard: Optional[int] = None
    intra_param_start: Optional[int] = None
    intra_param_end: Optional[int] = None


@dataclass
class FlattenedStorageShardingSpec:
    """
    Specification for FSDP v1-style flattened storage sharding.

    In this mode, multiple parameters are flattened to 1D, concatenated into
    a single buffer, and that buffer is sharded across ranks. Each rank holds
    a contiguous slice of the flattened buffer.

    Attributes:
        param_shapes: Original shapes of all parameters in the group.
        param_strides: Original strides of all parameters.
        param_numels: Number of elements in each parameter.
        total_numel: Total number of elements across all parameters.
        padded_total_numel: Total elements after padding for even division.
        mesh_dim: The mesh dimension name used for sharding.
        local_offset: Offset into the (unpadded) concatenated buffer for this rank.
        local_numel: Number of elements in this rank's shard.
        shard_param_infos: Per-parameter shard mapping information.
        param_index: Which parameter this MemoryShardedDTensor instance represents.
    """

    param_shapes: tuple[torch.Size, ...]
    param_strides: tuple[tuple[int, ...], ...]
    param_numels: tuple[int, ...]
    total_numel: int
    padded_total_numel: int
    mesh_dim: str
    local_offset: int
    local_numel: int
    shard_param_infos: tuple[ShardParamInfo, ...]
    param_index: int


@dataclass
class TensorGroupShardingSpec:
    """
    Specification for tensor-group sharding where each tensor is fully on one rank.

    In this mode, a group of tensors is distributed across ranks such that each
    tensor stays WHOLE on one rank (not split across ranks). This differs from
    FlattenedStorageShardingSpec where elements can span ranks.

    Distribution is contiguous: first N/world_size tensors to rank 0, etc.

    Attributes:
        param_shapes: Original shapes of all tensors in the group.
        param_strides: Original strides of all tensors.
        param_numels: Number of elements in each tensor.
        total_params: Total number of tensors in the group.
        mesh_dim: The mesh dimension name used for sharding.
        mesh_dim_idx: The mesh dimension index.
        param_to_rank: Mapping from param_index to the owning rank.
        rank_to_params: Mapping from rank to list of owned param indices.
        param_index: Which tensor this MemoryShardedDTensor instance represents.
        owns_tensor: Whether this rank owns this specific tensor.
    """

    param_shapes: tuple[torch.Size, ...]
    param_strides: tuple[tuple[int, ...], ...]
    param_numels: tuple[int, ...]
    total_params: int
    mesh_dim: str
    mesh_dim_idx: int
    param_to_rank: tuple[int, ...]
    rank_to_params: tuple[tuple[int, ...], ...]
    param_index: int
    owns_tensor: bool


# Union type for storage specs
StorageSpec = Union[BlockStorageShardingSpec, FlattenedStorageShardingSpec, TensorGroupShardingSpec]


def _validate_dtensor_for_storage_sharding(
    dtensor: DTensor,
    device_mesh: DeviceMesh,
    mesh_dim_idx: int,
) -> None:
    """
    Validate that a DTensor can be storage-sharded on the given mesh dimension.

    FSDP-style storage sharding requires the data to be replicated on the target
    mesh dimension. If data is already sharded or has pending reductions, we
    cannot properly distribute the storage.

    Args:
        dtensor: The DTensor to validate.
        device_mesh: The target device mesh for storage sharding.
        mesh_dim_idx: The mesh dimension index to shard on.

    Raises:
        ValueError: If the DTensor cannot be storage-sharded on the given mesh dim.
    """
    from torch.distributed.tensor.placement_types import Partial, Shard

    # Check same mesh
    if dtensor.device_mesh != device_mesh:
        raise ValueError(
            f"DTensor device_mesh {dtensor.device_mesh} does not match "
            f"target device_mesh {device_mesh}"
        )

    # Check placement on target mesh_dim is Replicate (not Shard or Partial)
    placement = dtensor.placements[mesh_dim_idx]
    if isinstance(placement, Shard):
        raise ValueError(
            f"Cannot shard storage on mesh_dim {mesh_dim_idx}: "
            f"DTensor already has Shard placement on this dimension"
        )
    if isinstance(placement, Partial):
        raise ValueError(
            f"Cannot shard storage on mesh_dim {mesh_dim_idx}: "
            f"DTensor has Partial placement (reduction pending)"
        )


class MemoryShardedDTensor(DTensor):
    """
    A DTensor subclass that physically shards storage across devices.

    MemoryShardedDTensor reduces per-device memory by partitioning the tensor's
    storage along a specified dimension. Each device holds only its local shard.
    The full tensor can be reconstructed via the unshard() method which performs
    an all-gather collective.

    This is useful for FSDP-style memory savings where parameters are sharded
    during forward/backward and gathered only when needed.

    Attributes:
        _storage_spec: StorageSpec describing the sharding configuration.
        _process_group: The process group used for collective operations.
        _padded_local: 1D flattened tensor with padding for even all-gather.
        _flat_buffer: For flattened mode, the shared flat buffer across params.
    """

    _storage_spec: StorageSpec
    _process_group: dist.ProcessGroup
    _padded_local: torch.Tensor
    _flat_buffer: Optional[torch.Tensor]
    __slots__ = ["_storage_spec", "_process_group", "_padded_local", "_flat_buffer"]

    def __new__(
        cls,
        local_tensor: torch.Tensor,
        spec: "DTensor._spec",  # type: ignore[name-defined]
        storage_spec: StorageSpec,
        process_group: dist.ProcessGroup,
        padded_local: torch.Tensor,
        *,
        requires_grad: bool,
        flat_buffer: Optional[torch.Tensor] = None,
    ) -> "MemoryShardedDTensor":
        # Create the DTensor base
        r = super().__new__(
            cls,
            local_tensor,
            spec,
            requires_grad=requires_grad,
        )
        r._storage_spec = storage_spec
        r._process_group = process_group
        r._padded_local = padded_local
        r._flat_buffer = flat_buffer
        return r

    def __init__(
        self,
        local_tensor: torch.Tensor,
        spec: "DTensor._spec",  # type: ignore[name-defined]
        storage_spec: StorageSpec,
        process_group: dist.ProcessGroup,
        padded_local: torch.Tensor,
        *,
        requires_grad: bool,
        flat_buffer: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

    def __repr__(self) -> str:
        if self.is_flattened_mode():
            spec = self._storage_spec
            return (
                f"MemoryShardedDTensor(flattened_mode=True, "
                f"local_shape={self.shape}, "
                f"full_shape={self.full_shape}, "
                f"param_index={spec.param_index}, "
                f"device_mesh={self._spec.mesh})"
            )
        elif self.is_tensor_group_mode():
            spec = self._storage_spec
            return (
                f"MemoryShardedDTensor(tensor_group_mode=True, "
                f"local_shape={self.shape}, "
                f"full_shape={self.full_shape}, "
                f"param_index={spec.param_index}, "
                f"owns_tensor={spec.owns_tensor}, "
                f"device_mesh={self._spec.mesh})"
            )
        spec = self._storage_spec
        return (
            f"MemoryShardedDTensor(local_shape={self.shape}, "
            f"full_shape={self.full_shape}, "
            f"shard_dims={spec.shard_dims}, "
            f"device_mesh={self._spec.mesh})"
        )

    def is_flattened_mode(self) -> bool:
        """
        Returns True if this tensor uses flattened storage sharding (FSDP v1 style).
        """
        return isinstance(self._storage_spec, FlattenedStorageShardingSpec)

    def is_tensor_group_mode(self) -> bool:
        """
        Returns True if this tensor uses tensor-group sharding.

        In tensor-group mode, each tensor in the group is fully on one rank
        (not split across ranks).
        """
        return isinstance(self._storage_spec, TensorGroupShardingSpec)

    @classmethod
    def _create(
        cls,
        local_tensor: torch.Tensor,
        device_mesh: DeviceMesh,
        storage_spec: StorageSpec,
        process_group: dist.ProcessGroup,
        placements: tuple,
        padded_local: Optional[torch.Tensor] = None,
        flat_buffer: Optional[torch.Tensor] = None,
    ) -> "MemoryShardedDTensor":
        """
        Factory method to create a MemoryShardedDTensor.

        Args:
            local_tensor: The local shard of the tensor on this rank.
            device_mesh: The DeviceMesh for this distributed tensor.
            storage_spec: StorageSpec describing sharding configuration.
            process_group: Process group for collective operations.
            placements: DTensor placements tuple.
            padded_local: Optional pre-computed 1D padded tensor. If None,
                will be computed from local_tensor and storage_spec.
            flat_buffer: For flattened mode, the shared flat buffer.

        Returns:
            A new MemoryShardedDTensor instance.
        """
        from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta

        # Compute padded_local if not provided (only for non-flattened mode)
        if padded_local is None:
            if isinstance(storage_spec, BlockStorageShardingSpec):
                padded_local = cls._compute_padded_local(local_tensor, storage_spec)
            else:
                # For flattened mode, use flat_buffer as padded_local
                padded_local = flat_buffer if flat_buffer is not None else local_tensor.view(-1)

        # Build DTensorSpec with the local tensor's metadata
        tensor_meta = TensorMeta(
            shape=local_tensor.shape,
            stride=local_tensor.stride(),
            dtype=local_tensor.dtype,
        )
        dtensor_spec = DTensorSpec(
            mesh=device_mesh,
            placements=placements,
            tensor_meta=tensor_meta,
        )

        return cls(
            local_tensor,
            dtensor_spec,
            storage_spec,
            process_group,
            padded_local,
            requires_grad=local_tensor.requires_grad,
            flat_buffer=flat_buffer,
        )

    @property
    def full_shape(self) -> torch.Size:
        """
        Returns the original (full) shape of the tensor before sharding.
        """
        if self.is_flattened_mode() or self.is_tensor_group_mode():
            spec = self._storage_spec
            return spec.param_shapes[spec.param_index]
        return self._storage_spec.orig_size

    def full_size(self, dim: Optional[int] = None) -> int | torch.Size:
        """
        Returns the original (full) size of the tensor.

        Args:
            dim: If specified, returns the size of that dimension.
                 If None, returns the full shape.

        Returns:
            Size of the specified dimension, or full shape if dim is None.
        """
        full_shape = self.full_shape
        if dim is None:
            return full_shape
        return full_shape[dim]

    def local(self) -> torch.Tensor:
        """
        Returns the local shard as a torch.Tensor.

        Returns:
            The underlying local tensor shard.
        """
        return self._local_tensor

    @property
    def storage_spec(self) -> StorageSpec:
        """
        Returns the storage sharding specification.
        """
        return self._storage_spec

    @property
    def process_group(self) -> dist.ProcessGroup:
        """
        Returns the process group used for collective operations.
        """
        return self._process_group

    @staticmethod
    def _compute_padded_local(
        local_tensor: torch.Tensor,
        storage_spec: BlockStorageShardingSpec,
    ) -> torch.Tensor:
        """
        Compute the 1D padded tensor from a local shard and storage spec.

        For multi-dimensional block sharding, pads each sharded dimension to
        its padded_shard_size, then flattens to 1D.

        Args:
            local_tensor: The local shard tensor.
            storage_spec: BlockStorageShardingSpec with padding info.

        Returns:
            1D flattened tensor with padding on all sharded dimensions.
        """
        # Check if any dimension needs padding
        needs_padding = any(
            actual != padded
            for actual, padded in zip(
                storage_spec.actual_shard_sizes,
                storage_spec.padded_shard_sizes,
            )
        )

        if not needs_padding:
            # No padding needed - just flatten
            return local_tensor.view(-1)

        # Build padded shape - local_tensor.shape has actual_shard_sizes on sharded dims
        padded_shape = list(local_tensor.shape)
        for i, shard_dim in enumerate(storage_spec.shard_dims):
            padded_shape[shard_dim] = storage_spec.padded_shard_sizes[i]

        # Create padded buffer and copy actual data
        padded = local_tensor.new_zeros(padded_shape)

        # Build slices to copy actual data
        slices = [slice(None)] * len(padded_shape)
        for i, shard_dim in enumerate(storage_spec.shard_dims):
            slices[shard_dim] = slice(0, storage_spec.actual_shard_sizes[i])

        padded[tuple(slices)] = local_tensor

        return padded.view(-1)

    def detach(self) -> "MemoryShardedDTensor":
        """
        Returns a detached MemoryShardedDTensor.

        This is required for nn.Parameter compatibility - the detach() method
        must return an instance of the same type.

        Returns:
            A new MemoryShardedDTensor with detached local tensor.
        """
        detached_local = self._local_tensor.detach()
        detached_padded = self._padded_local.detach()
        return self._create(
            local_tensor=detached_local,
            device_mesh=self._spec.mesh,
            storage_spec=self._storage_spec,
            process_group=self._process_group,
            placements=self._spec.placements,
            padded_local=detached_padded,
        )

    def _get_padded_local(self) -> torch.Tensor:
        """
        Returns the local tensor padded to the padded_shard_sizes as ND tensor.

        For uneven sharding, some ranks may have smaller shards than the
        padded size. This method returns an ND view of the padded local tensor
        for operations that need multi-dimensional access (like unshard).

        Returns:
            Local tensor padded on all sharded dimensions.
        """
        spec = self._storage_spec
        local_tensor = self._local_tensor

        # Build padded shape from local tensor shape with padded shard sizes
        padded_shape = list(local_tensor.shape)
        for i, shard_dim in enumerate(spec.shard_dims):
            padded_shape[shard_dim] = spec.padded_shard_sizes[i]

        return self._padded_local.view(padded_shape)

    def get_all_gather_input(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Returns a 1D flattened tensor suitable for all-gather collective operations.

        This method is used by FSDP to get the input tensor for batched all-gather.
        Returns the pre-computed padded tensor for O(1) access.

        For flattened mode, returns the shared flat buffer (single all-gather for
        all params in the group).

        Args:
            dtype: If provided, convert the tensor to this dtype before returning.
                   Used for mixed precision training where storage dtype differs
                   from compute dtype.

        Returns:
            A 1D flattened plain torch.Tensor (not DTensor) containing the padded
            local shard, suitable for passing to all-gather collectives.

        Example:
            >>> sharded = distribute_storage(dtensor, dim=0, mesh_dim="dp")
            >>> all_gather_input = sharded.get_all_gather_input(torch.float16)
            >>> # Use all_gather_input in batched all-gather collective
        """
        if self.is_flattened_mode():
            # In flattened mode, return the shared flat buffer
            result = self._flat_buffer
        else:
            # _padded_local is already 1D and padded - O(1) access
            result = self._padded_local

        # Apply dtype conversion if needed
        if dtype is not None and result.dtype != dtype:
            result = result.to(dtype)

        return result

    @classmethod
    def from_local_shard(
        cls,
        local_shard: torch.Tensor,
        full_shape: torch.Size,
        shard_dim: int,
        device_mesh: DeviceMesh,
        mesh_dim: int | str,
        *,
        requires_grad: bool = False,
        placements: tuple | None = None,
        padded_local: Optional[torch.Tensor] = None,
    ) -> "MemoryShardedDTensor":
        """
        Create a MemoryShardedDTensor from an already-sharded local tensor.

        This factory method is used by FSDP when it has already computed the
        local shard and needs to wrap it in a MemoryShardedDTensor. Unlike
        distribute_storage() which shards a full tensor, this method takes
        a pre-sharded local tensor.

        Args:
            local_shard: The local shard tensor on this rank.
            full_shape: The original (full) shape of the tensor before sharding.
            shard_dim: The dimension along which the tensor is sharded.
            device_mesh: The DeviceMesh for this distributed tensor.
            mesh_dim: The mesh dimension (name or index) used for sharding.
            requires_grad: Whether the tensor requires gradient computation.
            placements: Optional DTensor placements tuple. If None, defaults to
                all-Replicate placements. For TP+FSDP case, pass the combined
                SPMD placements (e.g., (Shard(dim), Shard(dim)) for TP sharding).
            padded_local: Optional pre-computed 1D padded tensor. If None,
                will be computed from local_shard.

        Returns:
            A MemoryShardedDTensor wrapping the local shard.

        Example:
            >>> # FSDP has already computed the local shard
            >>> local_shard = full_param.narrow(0, start, length).contiguous()
            >>> sharded = MemoryShardedDTensor.from_local_shard(
            ...     local_shard=local_shard,
            ...     full_shape=full_param.shape,
            ...     shard_dim=0,
            ...     device_mesh=mesh,
            ...     mesh_dim="dp",
            ... )
        """
        from torch.distributed.tensor.placement_types import Replicate

        # Resolve mesh_dim to index and name
        if isinstance(mesh_dim, str):
            mesh_dim_names = device_mesh.mesh_dim_names
            if mesh_dim_names is None or mesh_dim not in mesh_dim_names:
                raise ValueError(
                    f"mesh_dim '{mesh_dim}' not found in device mesh. "
                    f"Available dimensions: {mesh_dim_names}"
                )
            mesh_dim_name = mesh_dim
            mesh_dim_idx = mesh_dim_names.index(mesh_dim)
        else:
            mesh_dim_idx = mesh_dim
            if mesh_dim_idx < 0 or mesh_dim_idx >= device_mesh.ndim:
                raise ValueError(
                    f"mesh_dim {mesh_dim_idx} is out of range for mesh with "
                    f"{device_mesh.ndim} dimensions"
                )
            mesh_dim_names = device_mesh.mesh_dim_names
            mesh_dim_name = (
                mesh_dim_names[mesh_dim_idx] if mesh_dim_names else "default"
            )

        # Get process group and world size
        process_group = device_mesh.get_group(mesh_dim_idx)
        world_size = device_mesh.size(mesh_dim_idx)

        # Compute padded shard size from full shape
        full_size_on_dim = full_shape[shard_dim]
        padded_shard_size = (full_size_on_dim + world_size - 1) // world_size

        # Actual shard size is the size of the local tensor on shard_dim
        actual_shard_size = local_shard.size(shard_dim)

        # Compute original stride (assume contiguous layout for full tensor)
        orig_stride = []
        stride = 1
        for i in range(len(full_shape) - 1, -1, -1):
            orig_stride.insert(0, stride)
            stride *= full_shape[i]
        orig_stride = tuple(orig_stride)

        # Create storage sharding spec (single-dim is a special case of block sharding)
        storage_spec = BlockStorageShardingSpec(
            orig_size=full_shape,
            orig_stride=orig_stride,
            shard_dims=(shard_dim,),
            mesh_dims=(mesh_dim_name,),
            padded_shard_sizes=(padded_shard_size,),
            actual_shard_sizes=(actual_shard_size,),
            mesh_dim_indices=(mesh_dim_idx,),
        )

        # Use provided placements or default to all-Replicate
        if placements is None:
            placements = tuple(Replicate() for _ in range(device_mesh.ndim))

        # Ensure requires_grad is set correctly
        if requires_grad and not local_shard.requires_grad:
            local_shard = local_shard.requires_grad_(True)

        return cls._create(
            local_tensor=local_shard,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=process_group,
            placements=placements,
            padded_local=padded_local,
        )

    def unshard(self) -> DTensor:
        """
        Reconstruct the full tensor via all-gather collective.

        Performs an all-gather operation to collect all shards from all ranks
        in the process group, then reconstructs the original tensor shape.

        For flattened mode, this unshards this specific parameter by all-gathering
        the shared flat buffer and extracting this parameter's view.

        Returns:
            A DTensor containing the full (unsharded) tensor data replicated
            across all ranks.

        Example:
            >>> sharded = distribute_storage(dtensor, dim=0, mesh_dim="dp")
            >>> sharded.shape  # (4, 8) - local shard
            >>> full = sharded.unshard()
            >>> full.shape  # (16, 8) - full tensor
        """
        if self.is_flattened_mode():
            return self._unshard_flattened()
        elif self.is_tensor_group_mode():
            return self._unshard_tensor_group()
        return self._unshard_per_param()

    def _unshard_flattened(self) -> DTensor:
        """
        Unshard a parameter in flattened storage mode.

        All-gathers the shared flat buffer, then extracts this parameter's
        portion and reshapes it to the original shape.
        """
        from torch.distributed.tensor import DTensor
        from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
        from torch.distributed.tensor.placement_types import Replicate

        spec = self._storage_spec
        world_size = dist.get_world_size(self._process_group)

        # Get the shared flat buffer
        flat_buffer = self._flat_buffer

        # Detach for all-gather to avoid autograd issues
        orig_requires_grad = flat_buffer.requires_grad
        if orig_requires_grad:
            flat_buffer = flat_buffer.detach()

        # All-gather the flat buffer
        gathered_buffer = flat_buffer.new_empty(flat_buffer.numel() * world_size)
        dist.all_gather_into_tensor(
            gathered_buffer,
            flat_buffer,
            group=self._process_group,
        )

        # Slice to remove padding
        gathered_buffer = gathered_buffer[: spec.total_numel]

        # Extract this parameter's portion
        param_idx = spec.param_index
        offset = sum(spec.param_numels[:param_idx])
        numel = spec.param_numels[param_idx]
        param_data = gathered_buffer[offset : offset + numel]

        # Reshape to original shape
        param_shape = spec.param_shapes[param_idx]
        param_tensor = param_data.view(param_shape).contiguous()

        # Preserve requires_grad
        if orig_requires_grad:
            param_tensor = param_tensor.requires_grad_(True)

        # Create DTensor with Replicate placement
        device_mesh = self._spec.mesh
        placements = tuple(Replicate() for _ in range(device_mesh.ndim))

        tensor_meta = TensorMeta(
            shape=param_tensor.shape,
            stride=param_tensor.stride(),
            dtype=param_tensor.dtype,
        )
        dtensor_spec = DTensorSpec(
            mesh=device_mesh,
            placements=placements,
            tensor_meta=tensor_meta,
        )

        return DTensor(
            param_tensor,
            dtensor_spec,
            requires_grad=param_tensor.requires_grad,
        )

    def _unshard_tensor_group(self) -> DTensor:
        """
        Unshard a tensor in tensor-group mode via broadcast.

        In tensor-group mode, each tensor is fully on one rank. Unsharding
        broadcasts the tensor from its owning rank to all ranks.
        """
        from torch.distributed.tensor import DTensor
        from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
        from torch.distributed.tensor.placement_types import Replicate

        spec = self._storage_spec
        device_mesh = self._spec.mesh
        owning_rank = spec.param_to_rank[spec.param_index]
        shape = spec.param_shapes[spec.param_index]
        stride = spec.param_strides[spec.param_index]

        # Get or create tensor for broadcast
        if spec.owns_tensor:
            # This rank owns the tensor - use local data
            tensor = self._local_tensor.clone()
        else:
            # This rank doesn't own the tensor - create empty buffer
            tensor = self._local_tensor.new_zeros(shape)

        # Broadcast from owning rank
        dist.broadcast(tensor, src=owning_rank, group=self._process_group)

        # Ensure correct shape and contiguity
        if tensor.shape != shape:
            tensor = tensor.view(shape)
        tensor = tensor.contiguous()

        # Create DTensor with Replicate placements
        placements = tuple(Replicate() for _ in range(device_mesh.ndim))
        tensor_meta = TensorMeta(
            shape=tensor.shape,
            stride=tensor.stride(),
            dtype=tensor.dtype,
        )
        dtensor_spec = DTensorSpec(
            mesh=device_mesh,
            placements=placements,
            tensor_meta=tensor_meta,
        )

        return DTensor(
            tensor,
            dtensor_spec,
            requires_grad=self.requires_grad,
        )

    def _unshard_per_param(self) -> DTensor:
        """
        Unshard using block sharding mode via sequential all-gathers.

        This method handles both single-dimension sharding (FSDP v2 style) and
        multi-dimensional block sharding by performing all-gathers on each
        sharded dimension in reverse order (innermost to outermost).
        """
        from torch.distributed.tensor import DTensor
        from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
        from torch.distributed.tensor.placement_types import Replicate

        spec = self._storage_spec
        device_mesh = self._spec.mesh

        # Get padded local tensor for even all-gather
        padded_local = self._get_padded_local()

        # Detach for all-gather to avoid autograd issues, track original requires_grad
        orig_requires_grad = padded_local.requires_grad
        if orig_requires_grad:
            padded_local = padded_local.detach()

        # Perform all-gathers in reverse order (innermost to outermost)
        current_tensor = padded_local
        for i in range(len(spec.shard_dims) - 1, -1, -1):
            shard_dim = spec.shard_dims[i]
            mesh_dim_idx = spec.mesh_dim_indices[i]

            process_group = device_mesh.get_group(mesh_dim_idx)
            world_size = device_mesh.size(mesh_dim_idx)

            current_tensor = self._all_gather_on_dim(
                current_tensor, shard_dim, world_size, process_group
            )

        # Slice to original size (remove padding on all sharded dimensions)
        slices = [slice(None)] * current_tensor.ndim
        for i, shard_dim in enumerate(spec.shard_dims):
            slices[shard_dim] = slice(0, spec.orig_size[shard_dim])

        gathered_tensor = current_tensor[tuple(slices)].contiguous()

        # Preserve requires_grad
        if orig_requires_grad:
            gathered_tensor = gathered_tensor.requires_grad_(True)

        # Create DTensor with Replicate placement
        placements = tuple(Replicate() for _ in range(device_mesh.ndim))

        tensor_meta = TensorMeta(
            shape=gathered_tensor.shape,
            stride=gathered_tensor.stride(),
            dtype=gathered_tensor.dtype,
        )
        dtensor_spec = DTensorSpec(
            mesh=device_mesh,
            placements=placements,
            tensor_meta=tensor_meta,
        )

        return DTensor(
            gathered_tensor,
            dtensor_spec,
            requires_grad=gathered_tensor.requires_grad,
        )

    @staticmethod
    def _all_gather_on_dim(
        tensor: torch.Tensor,
        gather_dim: int,
        world_size: int,
        process_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        """
        Perform all-gather on a specific dimension of the tensor.

        Args:
            tensor: Input tensor to gather.
            gather_dim: Dimension to gather along.
            world_size: Number of ranks in the process group.
            process_group: The process group for collective.

        Returns:
            Gathered tensor with gather_dim size multiplied by world_size.
        """
        ndim = tensor.ndim

        # Move gather_dim to position 0 for simpler all-gather
        if gather_dim != 0:
            perm = [gather_dim] + [i for i in range(ndim) if i != gather_dim]
            tensor = tensor.permute(perm).contiguous()

        # Compute output shape
        output_shape = list(tensor.shape)
        output_shape[0] = output_shape[0] * world_size

        output = tensor.new_empty(output_shape)

        # All-gather
        dist.all_gather_into_tensor(output, tensor, group=process_group)

        # Permute back if needed
        if gather_dim != 0:
            inv_perm = [0] * ndim
            for i, p in enumerate(perm):
                inv_perm[p] = i
            output = output.permute(inv_perm).contiguous()

        return output


def distribute_storage(
    dtensor: DTensor,
    dim: int,
    mesh_dim: int | str,
) -> MemoryShardedDTensor:
    """
    Create a MemoryShardedDTensor by sharding a DTensor's storage along a dimension.

    This function takes a DTensor and shards its underlying storage along the
    specified dimension across devices in the given mesh dimension. Unlike
    DTensor's logical sharding, this physically partitions the storage to
    reduce per-device memory usage.

    Args:
        dtensor: The input DTensor to shard. Must be replicated on the target
            mesh dimension.
        dim: The tensor dimension along which to shard storage. Must be in
            range [-ndim, ndim).
        mesh_dim: The mesh dimension (name or index) to use for sharding.

    Returns:
        A MemoryShardedDTensor with storage sharded across devices.

    Raises:
        ValueError: If dim is out of range or mesh_dim doesn't exist.

    Example:
        >>> # FSDP-style sharding: shard parameters along dim 0
        >>> mesh = init_device_mesh("cuda", (4,), mesh_dim_names=("dp",))
        >>> param = distribute_tensor(torch.randn(16, 8), mesh, [Replicate()])
        >>> sharded = distribute_storage(param, dim=0, mesh_dim="dp")
        >>> sharded.shape  # Local shape: (4, 8)
        >>> sharded.full_shape  # Original shape: (16, 8)
    """
    from torch.distributed.tensor.placement_types import Replicate

    device_mesh = dtensor.device_mesh
    ndim = dtensor.ndim

    # Normalize negative dim
    if dim < 0:
        dim = dim + ndim

    # Validate dim is in range
    if dim < 0 or dim >= ndim:
        raise ValueError(f"dim {dim} is out of range for tensor with {ndim} dimensions")

    # Resolve mesh_dim to index if it's a string
    if isinstance(mesh_dim, str):
        mesh_dim_names = device_mesh.mesh_dim_names
        if mesh_dim_names is None or mesh_dim not in mesh_dim_names:
            raise ValueError(
                f"mesh_dim '{mesh_dim}' not found in device mesh. "
                f"Available dimensions: {mesh_dim_names}"
            )
        mesh_dim_name = mesh_dim
        mesh_dim_idx = mesh_dim_names.index(mesh_dim)
    else:
        mesh_dim_idx = mesh_dim
        if mesh_dim_idx < 0 or mesh_dim_idx >= device_mesh.ndim:
            raise ValueError(
                f"mesh_dim {mesh_dim_idx} is out of range for mesh with "
                f"{device_mesh.ndim} dimensions"
            )
        mesh_dim_names = device_mesh.mesh_dim_names
        mesh_dim_name = mesh_dim_names[mesh_dim_idx] if mesh_dim_names else "default"

    # Get process group and world size for the mesh dimension
    process_group = device_mesh.get_group(mesh_dim_idx)
    world_size = device_mesh.size(mesh_dim_idx)
    local_rank = device_mesh.get_local_rank(mesh_dim_idx)

    # Get the full tensor data (replicated on all ranks)
    full_tensor = dtensor.to_local()

    # Compute shard sizes
    full_size_on_dim = full_tensor.size(dim)
    # Use ceiling division for padded shard size
    padded_shard_size = (full_size_on_dim + world_size - 1) // world_size

    # Compute actual shard size for this rank
    start_idx = local_rank * padded_shard_size
    end_idx = min(start_idx + padded_shard_size, full_size_on_dim)
    actual_shard_size = max(0, end_idx - start_idx)

    # Extract the local shard
    if actual_shard_size > 0:
        local_shard = full_tensor.narrow(dim, start_idx, actual_shard_size)
        # Make contiguous copy to own the storage
        local_shard = local_shard.contiguous()
    else:
        # Empty shard for ranks beyond the tensor size
        shard_shape = list(full_tensor.shape)
        shard_shape[dim] = 0
        local_shard = full_tensor.new_empty(shard_shape)

    # Preserve requires_grad
    if full_tensor.requires_grad:
        local_shard = local_shard.requires_grad_(True)

    # Create storage sharding spec (single-dim is a special case of block sharding)
    storage_spec = BlockStorageShardingSpec(
        orig_size=full_tensor.size(),
        orig_stride=full_tensor.stride(),
        shard_dims=(dim,),
        mesh_dims=(mesh_dim_name,),
        padded_shard_sizes=(padded_shard_size,),
        actual_shard_sizes=(actual_shard_size,),
        mesh_dim_indices=(mesh_dim_idx,),
    )

    # Create placements - replicated on all dimensions
    placements = tuple(Replicate() for _ in range(device_mesh.ndim))

    return MemoryShardedDTensor._create(
        local_tensor=local_shard,
        device_mesh=device_mesh,
        storage_spec=storage_spec,
        process_group=process_group,
        placements=placements,
    )


def distribute_block_storage(
    dtensor: DTensor,
    shard_dims: list[int] | tuple[int, ...],
    mesh_dims: list[int | str] | tuple[int | str, ...] | None = None,
) -> MemoryShardedDTensor:
    """
    Create a MemoryShardedDTensor by block-sharding a DTensor's storage.

    This function shards the tensor across multiple dimensions simultaneously,
    creating a block/cube partitioning where each rank holds a multi-dimensional
    slice of the original tensor.

    Args:
        dtensor: The input DTensor to shard. Must be replicated on all target
            mesh dimensions.
        shard_dims: The tensor dimensions to shard. Each dimension is mapped
            to the corresponding mesh dimension in mesh_dims.
        mesh_dims: The mesh dimensions to use for sharding. If None, uses
            mesh dimensions 0, 1, 2, ... (first len(shard_dims) mesh dims).
            Can be names (str) or indices (int).

    Returns:
        A MemoryShardedDTensor with block-sharded storage.

    Raises:
        ValueError: If len(shard_dims) != len(mesh_dims), or if any dimension
            is out of range.

    Example:
        >>> # Block sharding: tensor [8, 4] on mesh (dp=4, tp=2) -> [2, 2] per rank
        >>> mesh = init_device_mesh("cuda", (4, 2), mesh_dim_names=("dp", "tp"))
        >>> param = distribute_tensor(torch.randn(8, 4), mesh, [Replicate(), Replicate()])
        >>> sharded = distribute_block_storage(param, shard_dims=[0, 1])
        >>> sharded.shape  # (2, 2)
        >>> sharded.full_shape  # (8, 4)
    """
    from torch.distributed.tensor.placement_types import Replicate

    device_mesh = dtensor.device_mesh
    ndim = dtensor.ndim

    # Default: use first len(shard_dims) mesh dimensions
    if mesh_dims is None:
        mesh_dims = tuple(range(len(shard_dims)))

    # Ensure tuples
    shard_dims = tuple(shard_dims)
    mesh_dims = tuple(mesh_dims)

    # Validate lengths match
    if len(shard_dims) != len(mesh_dims):
        raise ValueError(
            f"shard_dims and mesh_dims must have same length, "
            f"got {len(shard_dims)} and {len(mesh_dims)}"
        )

    # Normalize negative dims and validate
    normalized_shard_dims = []
    for d in shard_dims:
        if d < 0:
            d = d + ndim
        if d < 0 or d >= ndim:
            raise ValueError(f"shard_dim {d} is out of range for {ndim}D tensor")
        normalized_shard_dims.append(d)
    shard_dims = tuple(normalized_shard_dims)

    # Resolve mesh_dims to indices and names
    mesh_dim_indices = []
    mesh_dim_names = []
    for md in mesh_dims:
        if isinstance(md, str):
            names = device_mesh.mesh_dim_names
            if names is None or md not in names:
                raise ValueError(f"mesh_dim '{md}' not found in device mesh")
            mesh_dim_names.append(md)
            mesh_dim_indices.append(names.index(md))
        else:
            if md < 0 or md >= device_mesh.ndim:
                raise ValueError(f"mesh_dim {md} out of range for mesh")
            mesh_dim_indices.append(md)
            names = device_mesh.mesh_dim_names
            mesh_dim_names.append(names[md] if names else f"dim_{md}")

    mesh_dim_indices = tuple(mesh_dim_indices)
    mesh_dim_names = tuple(mesh_dim_names)

    # Get the full tensor data
    full_tensor = dtensor.to_local()

    # Compute shard sizes and extract local block
    padded_shard_sizes = []
    actual_shard_sizes = []
    local_slices = [slice(None)] * ndim

    for tensor_dim, mesh_dim_idx in zip(shard_dims, mesh_dim_indices):
        world_size = device_mesh.size(mesh_dim_idx)
        local_rank = device_mesh.get_local_rank(mesh_dim_idx)

        full_size = full_tensor.size(tensor_dim)
        padded_shard_size = (full_size + world_size - 1) // world_size

        start_idx = local_rank * padded_shard_size
        end_idx = min(start_idx + padded_shard_size, full_size)
        actual_shard_size = max(0, end_idx - start_idx)

        padded_shard_sizes.append(padded_shard_size)
        actual_shard_sizes.append(actual_shard_size)

        # Build slice for this dimension
        if actual_shard_size > 0:
            local_slices[tensor_dim] = slice(start_idx, end_idx)
        else:
            local_slices[tensor_dim] = slice(0, 0)

    # Extract local block
    local_block = full_tensor[tuple(local_slices)].contiguous()

    # Preserve requires_grad
    if full_tensor.requires_grad:
        local_block = local_block.requires_grad_(True)

    # Create BlockStorageShardingSpec
    storage_spec = BlockStorageShardingSpec(
        orig_size=full_tensor.size(),
        orig_stride=full_tensor.stride(),
        shard_dims=shard_dims,
        mesh_dims=mesh_dim_names,
        padded_shard_sizes=tuple(padded_shard_sizes),
        actual_shard_sizes=tuple(actual_shard_sizes),
        mesh_dim_indices=mesh_dim_indices,
    )

    # Use the first mesh dimension's process group as primary
    primary_pg = device_mesh.get_group(mesh_dim_indices[0])

    # Placements: Replicate on all dimensions
    placements = tuple(Replicate() for _ in range(device_mesh.ndim))

    return MemoryShardedDTensor._create(
        local_tensor=local_block,
        device_mesh=device_mesh,
        storage_spec=storage_spec,
        process_group=primary_pg,
        placements=placements,
    )


class TensorGroupStorage:
    """
    Manages a group of tensors distributed across ranks.

    This class supports two sharding modes:

    - "element" (FSDP v1 style): Tensors are flattened to 1D, concatenated into
      a single buffer, and that buffer is sharded element-wise across ranks.
      A single tensor may span multiple ranks.

    - "tensor": Each tensor is assigned to exactly one rank (not split).
      Distribution is contiguous: first N/world_size tensors to rank 0, etc.

    Example:
        >>> mesh = DeviceMesh("cuda", list(range(4)))
        >>> tensors = [torch.randn(8, 4), torch.randn(10, 6), torch.randn(4,)]

        >>> # Element mode (FSDP v1 style)
        >>> group = TensorGroupStorage(tensors, mesh, mesh_dim=0, mode="element")
        >>> sharded = group.shard()

        >>> # Tensor mode (each tensor whole on one rank)
        >>> group = TensorGroupStorage(tensors, mesh, mesh_dim=0, mode="tensor")
        >>> sharded = group.shard()
    """

    def __init__(
        self,
        params: list[torch.Tensor | DTensor],
        device_mesh: DeviceMesh,
        mesh_dim: int | str,
        mode: str = "element",
    ):
        """
        Initialize a TensorGroupStorage.

        Args:
            params: List of tensors to distribute. Can be plain torch.Tensor or
                DTensor. DTensor inputs must have Replicate placement on the
                target mesh dimension.
            device_mesh: The DeviceMesh for distributed operations.
            mesh_dim: The mesh dimension (name or index) to shard across.
            mode: Sharding mode - "element" (FSDP v1 style) or "tensor" (whole tensors).
        """
        if mode not in ("element", "tensor"):
            raise ValueError(f"mode must be 'element' or 'tensor', got '{mode}'")

        self._device_mesh = device_mesh
        self._mesh_dim = mesh_dim
        self._mode = mode
        self._flat_buffer: Optional[torch.Tensor] = None
        self._sharded_dtensors: list[MemoryShardedDTensor] = []
        self._process_group: Optional[dist.ProcessGroup] = None
        self._mesh_dim_name: Optional[str] = None

        # Resolve mesh_dim to index and name first (needed for validation)
        if isinstance(mesh_dim, str):
            mesh_dim_names = device_mesh.mesh_dim_names
            if mesh_dim_names is None or mesh_dim not in mesh_dim_names:
                raise ValueError(
                    f"mesh_dim '{mesh_dim}' not found in device mesh. "
                    f"Available dimensions: {mesh_dim_names}"
                )
            self._mesh_dim_name = mesh_dim
            self._mesh_dim_idx = mesh_dim_names.index(mesh_dim)
        else:
            self._mesh_dim_idx = mesh_dim
            if mesh_dim < 0 or mesh_dim >= device_mesh.ndim:
                raise ValueError(
                    f"mesh_dim {mesh_dim} is out of range for mesh with "
                    f"{device_mesh.ndim} dimensions"
                )
            mesh_dim_names = device_mesh.mesh_dim_names
            self._mesh_dim_name = (
                mesh_dim_names[mesh_dim] if mesh_dim_names else "default"
            )

        # Validate and extract local tensors from DTensor inputs
        local_params = []
        for p in params:
            if isinstance(p, DTensor):
                _validate_dtensor_for_storage_sharding(p, device_mesh, self._mesh_dim_idx)
                local_params.append(p.to_local())
            else:
                local_params.append(p)
        self._params = local_params

    def _get_process_group(self) -> dist.ProcessGroup:
        """Get or cache the process group for this mesh dimension."""
        if self._process_group is None:
            self._process_group = self._device_mesh.get_group(self._mesh_dim_idx)
        return self._process_group

    def _compute_shard_param_infos(
        self,
        numels: list[int],
        local_offset: int,
        shard_size: int,
    ) -> list[ShardParamInfo]:
        """
        Compute per-parameter shard mapping information.

        For each parameter, determines which portion (if any) falls within
        this rank's shard of the flattened buffer.

        Args:
            numels: Number of elements in each parameter.
            local_offset: Start offset of this rank's shard in the concatenated buffer.
            shard_size: Size of each rank's shard.

        Returns:
            List of ShardParamInfo, one per parameter.
        """
        shard_param_infos = []
        local_end = local_offset + shard_size
        param_start = 0

        for numel in numels:
            param_end = param_start + numel

            # Check if this param overlaps with local shard
            overlap_start = max(param_start, local_offset)
            overlap_end = min(param_end, local_end)

            if overlap_start < overlap_end:
                # Parameter is (partially) in this shard
                offset_in_shard = overlap_start - local_offset
                numel_in_shard = overlap_end - overlap_start
                intra_param_start = overlap_start - param_start
                intra_param_end = overlap_end - param_start

                shard_param_infos.append(
                    ShardParamInfo(
                        in_shard=True,
                        offset_in_shard=offset_in_shard,
                        numel_in_shard=numel_in_shard,
                        intra_param_start=intra_param_start,
                        intra_param_end=intra_param_end,
                    )
                )
            else:
                # Parameter not in this shard
                shard_param_infos.append(ShardParamInfo(in_shard=False))

            param_start = param_end

        return shard_param_infos

    def _extract_param_local(
        self,
        param_index: int,
        shard_param_info: ShardParamInfo,
    ) -> torch.Tensor:
        """
        Extract this parameter's portion from the local shard.

        Args:
            param_index: Index of the parameter in the group.
            shard_param_info: ShardParamInfo for this parameter.

        Returns:
            A tensor containing this parameter's local portion, or empty tensor
            if the parameter is not in this rank's shard.
        """
        if not shard_param_info.in_shard:
            # Parameter not in this shard - return empty tensor
            return self._flat_buffer.new_empty(0)

        # Extract from flat buffer
        offset = shard_param_info.offset_in_shard
        numel = shard_param_info.numel_in_shard
        return self._flat_buffer[offset : offset + numel]

    def shard(self) -> list[MemoryShardedDTensor]:
        """
        Distribute tensors across ranks based on the sharding mode.

        For "element" mode (FSDP v1 style):
        - Flattens each parameter to 1D
        - Concatenates them into a single buffer
        - Pads for even distribution across ranks
        - Takes this rank's shard of the buffer

        For "tensor" mode:
        - Assigns whole tensors to ranks (contiguous chunks)
        - Each tensor stays fully on one rank

        Returns:
            List of MemoryShardedDTensor, one per input tensor.
        """
        if self._mode == "element":
            return self._shard_element()
        else:
            return self._shard_tensor()

    def _shard_element(self) -> list[MemoryShardedDTensor]:
        """Shard using element mode (FSDP v1 style)."""
        from torch.distributed.tensor.placement_types import Replicate

        # 1. Collect metadata
        shapes = [p.shape for p in self._params]
        strides = [p.stride() for p in self._params]
        numels = [p.numel() for p in self._params]
        total_numel = sum(numels)

        # 2. Flatten and concatenate
        flat_tensors = [p.view(-1) for p in self._params]
        concatenated = torch.cat(flat_tensors, dim=0)

        # 3. Compute shard info
        world_size = self._device_mesh.size(self._mesh_dim_idx)
        rank = self._device_mesh.get_local_rank(self._mesh_dim_idx)
        padded_numel = math.ceil(total_numel / world_size) * world_size
        shard_size = padded_numel // world_size

        # 4. Pad if needed
        if total_numel < padded_numel:
            concatenated = F.pad(concatenated, [0, padded_numel - total_numel])

        # 5. Take this rank's shard
        local_offset = rank * shard_size
        self._flat_buffer = concatenated[local_offset : local_offset + shard_size].clone()

        # 6. Compute per-param shard mappings
        shard_param_infos = self._compute_shard_param_infos(
            numels, local_offset, shard_size
        )

        # 7. Create MemoryShardedDTensor with FlattenedStorageShardingSpec for each
        placements = tuple(Replicate() for _ in range(self._device_mesh.ndim))
        process_group = self._get_process_group()

        for i, (shape, stride, spi) in enumerate(
            zip(shapes, strides, shard_param_infos)
        ):
            spec = FlattenedStorageShardingSpec(
                param_shapes=tuple(shapes),
                param_strides=tuple(strides),
                param_numels=tuple(numels),
                total_numel=total_numel,
                padded_total_numel=padded_numel,
                mesh_dim=self._mesh_dim_name,
                local_offset=local_offset,
                local_numel=shard_size,
                shard_param_infos=tuple(shard_param_infos),
                param_index=i,
            )

            # Extract this param's portion of the local shard
            local_tensor = self._extract_param_local(i, spi)

            dtensor = MemoryShardedDTensor._create(
                local_tensor=local_tensor,
                device_mesh=self._device_mesh,
                storage_spec=spec,
                process_group=process_group,
                placements=placements,
                flat_buffer=self._flat_buffer,
            )
            self._sharded_dtensors.append(dtensor)

        return self._sharded_dtensors

    def _shard_tensor(self) -> list[MemoryShardedDTensor]:
        """Shard using tensor mode (each tensor fully on one rank)."""
        from torch.distributed.tensor.placement_types import Replicate

        # 1. Collect metadata
        shapes = tuple(p.shape for p in self._params)
        strides = tuple(p.stride() for p in self._params)
        numels = tuple(p.numel() for p in self._params)
        total_params = len(self._params)

        # 2. Compute tensor-to-rank assignment (contiguous chunks)
        world_size = self._device_mesh.size(self._mesh_dim_idx)
        rank = self._device_mesh.get_local_rank(self._mesh_dim_idx)

        param_to_rank, rank_to_params = self._compute_tensor_assignment(
            total_params, world_size
        )

        # 3. Create MemoryShardedDTensor for each tensor
        placements = tuple(Replicate() for _ in range(self._device_mesh.ndim))
        process_group = self._get_process_group()

        for i, (shape, stride, numel) in enumerate(zip(shapes, strides, numels)):
            owns_tensor = param_to_rank[i] == rank

            spec = TensorGroupShardingSpec(
                param_shapes=shapes,
                param_strides=strides,
                param_numels=numels,
                total_params=total_params,
                mesh_dim=self._mesh_dim_name,
                mesh_dim_idx=self._mesh_dim_idx,
                param_to_rank=param_to_rank,
                rank_to_params=rank_to_params,
                param_index=i,
                owns_tensor=owns_tensor,
            )

            # Get local tensor (actual tensor if owned, empty placeholder if not)
            if owns_tensor:
                local_tensor = self._params[i]
            else:
                # Create empty placeholder with correct shape
                local_tensor = self._params[i].new_empty(0)

            dtensor = MemoryShardedDTensor._create(
                local_tensor=local_tensor,
                device_mesh=self._device_mesh,
                storage_spec=spec,
                process_group=process_group,
                placements=placements,
            )
            self._sharded_dtensors.append(dtensor)

        return self._sharded_dtensors

    @staticmethod
    def _compute_tensor_assignment(
        total_params: int, world_size: int
    ) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
        """
        Compute tensor-to-rank assignment using contiguous chunks.

        First N/world_size tensors go to rank 0, next N/world_size to rank 1, etc.

        Returns:
            param_to_rank: Tuple mapping param_index to owning rank.
            rank_to_params: Tuple mapping rank to tuple of owned param indices.
        """
        if total_params == 0:
            return (), tuple(() for _ in range(world_size))

        base_count = total_params // world_size
        extra = total_params % world_size

        param_to_rank = []
        rank_to_params: list[list[int]] = [[] for _ in range(world_size)]

        param_idx = 0
        for r in range(world_size):
            # First 'extra' ranks get base_count + 1 tensors
            count = base_count + (1 if r < extra else 0)
            for _ in range(count):
                param_to_rank.append(r)
                rank_to_params[r].append(param_idx)
                param_idx += 1

        return tuple(param_to_rank), tuple(tuple(p) for p in rank_to_params)

    def unshard_all(self) -> list[DTensor]:
        """
        Reconstruct all tensors and return them replicated across ranks.

        For "element" mode: Performs a single all-gather on the flat buffer,
        then extracts each tensor's view.

        For "tensor" mode: Broadcasts each tensor from its owning rank.

        Returns:
            List of DTensor, one per tensor, with Replicate placements.
        """
        if not self._sharded_dtensors:
            raise RuntimeError("Must call shard() before unshard_all()")

        if self._mode == "element":
            return self._unshard_all_element()
        else:
            return self._unshard_all_tensor()

    def _unshard_all_element(self) -> list[DTensor]:
        """Unshard all tensors in element mode (single all-gather)."""
        from torch.distributed.tensor import DTensor
        from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
        from torch.distributed.tensor.placement_types import Replicate

        # Single all-gather for entire flat buffer
        world_size = self._device_mesh.size(self._mesh_dim_idx)
        process_group = self._get_process_group()

        # Detach for all-gather
        flat_buffer = self._flat_buffer
        orig_requires_grad = flat_buffer.requires_grad
        if orig_requires_grad:
            flat_buffer = flat_buffer.detach()

        full_buffer = flat_buffer.new_empty(flat_buffer.numel() * world_size)
        dist.all_gather_into_tensor(
            full_buffer, flat_buffer, group=process_group
        )

        # Get spec from first sharded tensor
        spec = self._sharded_dtensors[0]._storage_spec

        # Slice to remove padding
        full_buffer = full_buffer[: spec.total_numel]

        # Create views for each param
        unsharded = []
        placements = tuple(Replicate() for _ in range(self._device_mesh.ndim))
        offset = 0

        for i, numel in enumerate(spec.param_numels):
            shape = spec.param_shapes[i]
            param_data = full_buffer[offset : offset + numel]
            param_tensor = param_data.view(shape).contiguous()

            if orig_requires_grad:
                param_tensor = param_tensor.requires_grad_(True)

            tensor_meta = TensorMeta(
                shape=param_tensor.shape,
                stride=param_tensor.stride(),
                dtype=param_tensor.dtype,
            )
            dtensor_spec = DTensorSpec(
                mesh=self._device_mesh,
                placements=placements,
                tensor_meta=tensor_meta,
            )

            unsharded.append(
                DTensor(
                    param_tensor,
                    dtensor_spec,
                    requires_grad=param_tensor.requires_grad,
                )
            )
            offset += numel

        return unsharded

    def _unshard_all_tensor(self) -> list[DTensor]:
        """Unshard all tensors in tensor mode (broadcast from each owner)."""
        # Simply call unshard() on each sharded tensor
        return [msdt.unshard() for msdt in self._sharded_dtensors]

    def get_flat_buffer(self) -> torch.Tensor:
        """
        Returns the shared flat buffer.

        This can be used for FSDP integration where the all-gather input
        is needed directly.

        Returns:
            The 1D flat buffer containing this rank's shard of all parameters.
        """
        if self._flat_buffer is None:
            raise RuntimeError("Must call shard() before get_flat_buffer()")
        return self._flat_buffer


class FlattenedStorageGroup(TensorGroupStorage):
    """
    Deprecated: Use TensorGroupStorage with mode='element' instead.

    This class is kept for backward compatibility.
    """

    def __init__(
        self,
        params: list[torch.Tensor],
        device_mesh: DeviceMesh,
        mesh_dim: int | str,
    ):
        import warnings

        warnings.warn(
            "FlattenedStorageGroup is deprecated, use TensorGroupStorage with mode='element' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(params, device_mesh, mesh_dim, mode="element")


def distribute_tensor_group(
    tensors: list[torch.Tensor | DTensor],
    device_mesh: DeviceMesh,
    mesh_dim: int | str,
    mode: str = "element",
) -> list[MemoryShardedDTensor]:
    """
    Distribute a group of tensors across ranks.

    This convenience function creates a TensorGroupStorage and shards the tensors.

    Args:
        tensors: List of tensors to distribute. Can be plain torch.Tensor or
            DTensor. DTensor inputs must have Replicate placement on the
            target mesh dimension.
        device_mesh: The DeviceMesh for distributed operations.
        mesh_dim: The mesh dimension (name or index) to shard across.
        mode: Sharding mode - "element" (FSDP v1 style) or "tensor" (whole tensors).

    Returns:
        List of MemoryShardedDTensor, one per input tensor.

    Example:
        >>> mesh = init_device_mesh("cuda", (4,), mesh_dim_names=("dp",))
        >>> tensors = [torch.randn(8, 4), torch.randn(10, 6)]

        >>> # Element mode (FSDP v1 style)
        >>> sharded = distribute_tensor_group(tensors, mesh, "dp", mode="element")

        >>> # Tensor mode (each tensor whole on one rank)
        >>> sharded = distribute_tensor_group(tensors, mesh, "dp", mode="tensor")
    """
    group = TensorGroupStorage(tensors, device_mesh, mesh_dim, mode=mode)
    return group.shard()
