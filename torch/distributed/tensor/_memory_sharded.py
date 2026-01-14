# Copyright (c) Meta Platforms, Inc. and affiliates
"""
MemoryShardedDTensor: A DTensor variant that shards storage across devices.

This module provides a memory-efficient DTensor implementation where the tensor's
storage is sharded across devices in a process group, reducing per-device memory
usage. Unlike regular DTensor sharding which affects the logical tensor view,
MemoryShardedDTensor physically partitions the underlying storage.
"""
from dataclasses import dataclass
from typing import Optional

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
        _storage_spec: BlockStorageShardingSpec describing the sharding configuration.
        _process_group: The process group used for collective operations.
        _padded_local: 1D flattened tensor with padding for even all-gather.
    """

    _storage_spec: BlockStorageShardingSpec
    _process_group: dist.ProcessGroup
    _padded_local: torch.Tensor
    __slots__ = ["_storage_spec", "_process_group", "_padded_local"]

    def __new__(
        cls,
        local_tensor: torch.Tensor,
        spec: "DTensor._spec",  # type: ignore[name-defined]
        storage_spec: BlockStorageShardingSpec,
        process_group: dist.ProcessGroup,
        padded_local: torch.Tensor,
        *,
        requires_grad: bool,
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
        return r

    def __init__(
        self,
        local_tensor: torch.Tensor,
        spec: "DTensor._spec",  # type: ignore[name-defined]
        storage_spec: BlockStorageShardingSpec,
        process_group: dist.ProcessGroup,
        padded_local: torch.Tensor,
        *,
        requires_grad: bool,
    ) -> None:
        super().__init__()

    def __repr__(self) -> str:
        spec = self._storage_spec
        return (
            f"MemoryShardedDTensor(local_shape={self.shape}, "
            f"full_shape={self.full_shape}, "
            f"shard_dims={spec.shard_dims}, "
            f"device_mesh={self._spec.mesh})"
        )

    @classmethod
    def _create(
        cls,
        local_tensor: torch.Tensor,
        device_mesh: DeviceMesh,
        storage_spec: BlockStorageShardingSpec,
        process_group: dist.ProcessGroup,
        placements: tuple,
        padded_local: Optional[torch.Tensor] = None,
    ) -> "MemoryShardedDTensor":
        """
        Factory method to create a MemoryShardedDTensor.

        Args:
            local_tensor: The local shard of the tensor on this rank.
            device_mesh: The DeviceMesh for this distributed tensor.
            storage_spec: BlockStorageShardingSpec describing sharding configuration.
            process_group: Process group for collective operations.
            placements: DTensor placements tuple.
            padded_local: Optional pre-computed 1D padded tensor. If None,
                will be computed from local_tensor and storage_spec.

        Returns:
            A new MemoryShardedDTensor instance.
        """
        from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta

        # Compute padded_local if not provided
        if padded_local is None:
            padded_local = cls._compute_padded_local(local_tensor, storage_spec)

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
        )

    @property
    def full_shape(self) -> torch.Size:
        """
        Returns the original (full) shape of the tensor before sharding.
        """
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
    def storage_spec(self) -> BlockStorageShardingSpec:
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
        Compute 1D padded tensor for even all-gather (block sharding mode).

        For multi-dimensional sharding, we need to pad each dimension to the
        padded shard size, then flatten. This ensures that all ranks have
        the same size tensor for all-gather operations.
        """
        # Start with the local tensor
        current = local_tensor

        # Pad each sharded dimension if needed
        for i, shard_dim in enumerate(storage_spec.shard_dims):
            actual_size = storage_spec.actual_shard_sizes[i]
            padded_size = storage_spec.padded_shard_sizes[i]

            if actual_size < padded_size:
                # Need to pad this dimension
                pad_amount = padded_size - actual_size
                # F.pad pads from last dim to first, so we need to compute padding
                # For a tensor of ndim N, and we want to pad dim D:
                # pad = [0, 0] * (N - D - 1) + [0, pad_amount] + [0, 0] * D
                # But F.pad takes it in reverse order
                ndim = current.ndim
                # Padding is applied from last dimension backwards
                # We need (N - D - 1) pairs of zeros after, then the pad, then D pairs before
                pad = [0] * (2 * (ndim - shard_dim - 1)) + [0, pad_amount]
                current = F.pad(current, pad)

        return current.flatten().contiguous()

    def detach(self) -> "MemoryShardedDTensor":
        """
        Returns a detached MemoryShardedDTensor (no gradient tracking).
        """
        detached_local = self._local_tensor.detach()
        detached_padded = self._padded_local.detach()

        return MemoryShardedDTensor._create(
            local_tensor=detached_local,
            device_mesh=self._spec.mesh,
            storage_spec=self._storage_spec,
            process_group=self._process_group,
            placements=self._spec.placements,
            padded_local=detached_padded,
        )

    def _get_padded_local(self) -> torch.Tensor:
        """
        Get the 1D padded local tensor for all-gather operations.

        Returns:
            1D flattened tensor with padding for even division.
        """
        return self._padded_local

    def get_all_gather_input(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Get the tensor to use as input for all-gather operation.

        This returns the 1D padded local tensor, optionally cast to a
        different dtype. This is used by FSDP for the foreach fast path.

        Args:
            dtype: Optional dtype to cast the tensor to.

        Returns:
            1D tensor suitable for all-gather input.
        """
        padded = self._get_padded_local()
        if dtype is not None and padded.dtype != dtype:
            return padded.to(dtype)
        return padded

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

        Returns:
            A DTensor containing the full (unsharded) tensor data replicated
            across all ranks.

        Example:
            >>> sharded = distribute_storage(dtensor, dim=0, mesh_dim="dp")
            >>> sharded.shape  # (4, 8) - local shard
            >>> full = sharded.unshard()
            >>> full.shape  # (16, 8) - full tensor
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
            process_group: Process group for the collective.

        Returns:
            Tensor with gather_dim size multiplied by world_size.
        """
        # For 1D tensors or gathering on dim 0, use simple all-gather
        if tensor.ndim == 1 or gather_dim == 0:
            output_shape = list(tensor.shape)
            output_shape[0] = output_shape[0] * world_size

            output = tensor.new_empty(output_shape)
            dist.all_gather_into_tensor(output, tensor, group=process_group)
            return output

        # For higher dims, permute to bring gather_dim to front
        ndim = tensor.ndim
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
    mesh_dim_names_list = []
    for md in mesh_dims:
        if isinstance(md, str):
            names = device_mesh.mesh_dim_names
            if names is None or md not in names:
                raise ValueError(f"mesh_dim '{md}' not found in device mesh")
            mesh_dim_names_list.append(md)
            mesh_dim_indices.append(names.index(md))
        else:
            if md < 0 or md >= device_mesh.ndim:
                raise ValueError(f"mesh_dim {md} out of range for mesh")
            mesh_dim_indices.append(md)
            names = device_mesh.mesh_dim_names
            mesh_dim_names_list.append(names[md] if names else f"dim_{md}")

    mesh_dim_indices = tuple(mesh_dim_indices)
    mesh_dim_names_tuple = tuple(mesh_dim_names_list)

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
        mesh_dims=mesh_dim_names_tuple,
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
