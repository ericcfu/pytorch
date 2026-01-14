# Copyright (c) Meta Platforms, Inc. and affiliates
"""
Tests for MemoryShardedDTensor core class functionality.
"""
import torch
import torch.distributed as dist
from torch.distributed.tensor import (
    DTensor,
    init_device_mesh,
    Replicate,
)
from torch.distributed.tensor._memory_sharded import (
    BlockStorageShardingSpec,
    MemoryShardedDTensor,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class TestBlockStorageShardingSpec(TestCase):
    """Unit tests for BlockStorageShardingSpec dataclass."""

    def test_block_storage_sharding_spec_creation(self):
        """Test that BlockStorageShardingSpec can be created with all fields."""
        spec = BlockStorageShardingSpec(
            orig_size=torch.Size([16, 32]),
            orig_stride=(32, 1),
            shard_dims=(0,),
            mesh_dims=("dp",),
            padded_shard_sizes=(4,),
            actual_shard_sizes=(4,),
            mesh_dim_indices=(0,),
        )
        self.assertEqual(spec.orig_size, torch.Size([16, 32]))
        self.assertEqual(spec.orig_stride, (32, 1))
        self.assertEqual(spec.shard_dims, (0,))
        self.assertEqual(spec.mesh_dims, ("dp",))
        self.assertEqual(spec.padded_shard_sizes, (4,))
        self.assertEqual(spec.actual_shard_sizes, (4,))

    def test_block_storage_sharding_spec_uneven(self):
        """Test BlockStorageShardingSpec with uneven sharding (different actual vs padded)."""
        spec = BlockStorageShardingSpec(
            orig_size=torch.Size([13, 32]),
            orig_stride=(32, 1),
            shard_dims=(0,),
            mesh_dims=("dp",),
            padded_shard_sizes=(4,),  # ceiling(13/4) = 4
            actual_shard_sizes=(1,),  # last rank gets only 1 row
            mesh_dim_indices=(0,),
        )
        self.assertEqual(spec.padded_shard_sizes, (4,))
        self.assertEqual(spec.actual_shard_sizes, (1,))

    def test_block_storage_sharding_spec_multi_dim(self):
        """Test BlockStorageShardingSpec with multi-dimensional sharding."""
        spec = BlockStorageShardingSpec(
            orig_size=torch.Size([8, 4]),
            orig_stride=(4, 1),
            shard_dims=(0, 1),
            mesh_dims=("dp", "tp"),
            padded_shard_sizes=(2, 2),
            actual_shard_sizes=(2, 2),
            mesh_dim_indices=(0, 1),
        )
        self.assertEqual(spec.shard_dims, (0, 1))
        self.assertEqual(spec.mesh_dims, ("dp", "tp"))
        self.assertEqual(spec.padded_shard_sizes, (2, 2))
        self.assertEqual(spec.actual_shard_sizes, (2, 2))


class TestMemoryShardedDTensor(DTensorTestBase):
    """Distributed tests for MemoryShardedDTensor class."""

    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_memory_sharded_dtensor_creation(self):
        """Verify MemoryShardedDTensor can be instantiated."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type)

        storage_spec = BlockStorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dims=(0,),
            mesh_dims=("default",),
            padded_shard_sizes=(4,),
            actual_shard_sizes=(4,),
            mesh_dim_indices=(0,),
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertIsInstance(msdt, MemoryShardedDTensor)
        self.assertIsInstance(msdt, DTensor)

    @with_comms
    def test_shape_returns_local(self):
        """Test that .shape returns local sharded shape."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type)

        storage_spec = BlockStorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dims=(0,),
            mesh_dims=("default",),
            padded_shard_sizes=(4,),
            actual_shard_sizes=(4,),
            mesh_dim_indices=(0,),
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertEqual(msdt.shape, torch.Size([4, 8]))

    @with_comms
    def test_full_shape_returns_original(self):
        """Test that .full_shape returns original size."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type)

        storage_spec = BlockStorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dims=(0,),
            mesh_dims=("default",),
            padded_shard_sizes=(4,),
            actual_shard_sizes=(4,),
            mesh_dim_indices=(0,),
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertEqual(msdt.full_shape, torch.Size([16, 8]))

    @with_comms
    def test_size_with_dim(self):
        """Test that .size(dim) returns local size."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type)

        storage_spec = BlockStorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dims=(0,),
            mesh_dims=("default",),
            padded_shard_sizes=(4,),
            actual_shard_sizes=(4,),
            mesh_dim_indices=(0,),
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertEqual(msdt.size(0), 4)
        self.assertEqual(msdt.size(1), 8)

    @with_comms
    def test_full_size_with_dim(self):
        """Test that .full_size(dim) returns original size."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type)

        storage_spec = BlockStorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dims=(0,),
            mesh_dims=("default",),
            padded_shard_sizes=(4,),
            actual_shard_sizes=(4,),
            mesh_dim_indices=(0,),
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertEqual(msdt.full_size(0), 16)
        self.assertEqual(msdt.full_size(1), 8)
        self.assertEqual(msdt.full_size(), torch.Size([16, 8]))

    @with_comms
    def test_local_returns_tensor(self):
        """Test that .local() returns torch.Tensor."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type)

        storage_spec = BlockStorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dims=(0,),
            mesh_dims=("default",),
            padded_shard_sizes=(4,),
            actual_shard_sizes=(4,),
            mesh_dim_indices=(0,),
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        result = msdt.local()
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, torch.Size([4, 8]))

    @with_comms
    def test_ndim(self):
        """Test that .ndim is correct."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type)

        storage_spec = BlockStorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dims=(0,),
            mesh_dims=("default",),
            padded_shard_sizes=(4,),
            actual_shard_sizes=(4,),
            mesh_dim_indices=(0,),
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertEqual(msdt.ndim, 2)

    @with_comms
    def test_dtype(self):
        """Test that .dtype is preserved."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type, dtype=torch.float16)

        storage_spec = BlockStorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dims=(0,),
            mesh_dims=("default",),
            padded_shard_sizes=(4,),
            actual_shard_sizes=(4,),
            mesh_dim_indices=(0,),
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertEqual(msdt.dtype, torch.float16)

    @with_comms
    def test_device(self):
        """Test that .device is correct."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type)

        storage_spec = BlockStorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dims=(0,),
            mesh_dims=("default",),
            padded_shard_sizes=(4,),
            actual_shard_sizes=(4,),
            mesh_dim_indices=(0,),
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertEqual(msdt.device.type, self.device_type)

    @with_comms
    def test_requires_grad(self):
        """Test that requires_grad is preserved."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type, requires_grad=True)

        storage_spec = BlockStorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dims=(0,),
            mesh_dims=("default",),
            padded_shard_sizes=(4,),
            actual_shard_sizes=(4,),
            mesh_dim_indices=(0,),
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertTrue(msdt.requires_grad)

        # Test with requires_grad=False
        local_tensor_no_grad = torch.randn(4, 8, device=self.device_type)
        msdt_no_grad = MemoryShardedDTensor._create(
            local_tensor=local_tensor_no_grad,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )
        self.assertFalse(msdt_no_grad.requires_grad)

    @with_comms
    def test_storage_spec_property(self):
        """Test that storage_spec property returns the spec."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type)

        storage_spec = BlockStorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dims=(0,),
            mesh_dims=("default",),
            padded_shard_sizes=(4,),
            actual_shard_sizes=(4,),
            mesh_dim_indices=(0,),
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertEqual(msdt.storage_spec, storage_spec)
        self.assertEqual(msdt.storage_spec.shard_dims[0], 0)
        self.assertEqual(msdt.storage_spec.mesh_dims[0], "default")

    @with_comms
    def test_process_group_property(self):
        """Test that process_group property returns the PG."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type)

        storage_spec = BlockStorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dims=(0,),
            mesh_dims=("default",),
            padded_shard_sizes=(4,),
            actual_shard_sizes=(4,),
            mesh_dim_indices=(0,),
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertEqual(msdt.process_group, pg)


class TestFromLocalShard(DTensorTestBase):
    """Tests for MemoryShardedDTensor.from_local_shard() factory method."""

    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_from_local_shard_basic(self):
        """Test from_local_shard creates correct MemoryShardedDTensor."""
        device_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("dp",)
        )

        # Create a local shard as FSDP would
        local_shard = torch.randn(4, 8, device=self.device_type)
        full_shape = torch.Size([16, 8])

        msdt = MemoryShardedDTensor.from_local_shard(
            local_shard=local_shard,
            full_shape=full_shape,
            shard_dim=0,
            device_mesh=device_mesh,
            mesh_dim="dp",
        )

        self.assertIsInstance(msdt, MemoryShardedDTensor)
        self.assertEqual(msdt.shape, torch.Size([4, 8]))
        self.assertEqual(msdt.full_shape, torch.Size([16, 8]))
        self.assertEqual(msdt.storage_spec.shard_dims[0], 0)
        self.assertEqual(msdt.storage_spec.mesh_dims[0], "dp")

    @with_comms
    def test_from_local_shard_with_mesh_dim_index(self):
        """Test from_local_shard with mesh_dim as index."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        local_shard = torch.randn(4, 8, device=self.device_type)
        full_shape = torch.Size([16, 8])

        msdt = MemoryShardedDTensor.from_local_shard(
            local_shard=local_shard,
            full_shape=full_shape,
            shard_dim=0,
            device_mesh=device_mesh,
            mesh_dim=0,  # Use index instead of name
        )

        self.assertIsInstance(msdt, MemoryShardedDTensor)
        self.assertEqual(msdt.full_shape, torch.Size([16, 8]))

    @with_comms
    def test_from_local_shard_uneven(self):
        """Test from_local_shard with uneven sharding (last rank smaller)."""
        device_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("dp",)
        )

        rank = dist.get_rank()
        # Simulate FSDP's uneven sharding: 13 / 4 = 4, 4, 4, 1
        if rank < 3:
            local_shard = torch.randn(4, 8, device=self.device_type)
        else:
            local_shard = torch.randn(1, 8, device=self.device_type)

        full_shape = torch.Size([13, 8])

        msdt = MemoryShardedDTensor.from_local_shard(
            local_shard=local_shard,
            full_shape=full_shape,
            shard_dim=0,
            device_mesh=device_mesh,
            mesh_dim="dp",
        )

        self.assertEqual(msdt.full_shape, torch.Size([13, 8]))
        self.assertEqual(msdt.storage_spec.padded_shard_sizes[0], 4)

        if rank < 3:
            self.assertEqual(msdt.size(0), 4)
            self.assertEqual(msdt.storage_spec.actual_shard_sizes[0], 4)
        else:
            self.assertEqual(msdt.size(0), 1)
            self.assertEqual(msdt.storage_spec.actual_shard_sizes[0], 1)

    @with_comms
    def test_from_local_shard_requires_grad(self):
        """Test from_local_shard respects requires_grad."""
        device_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("dp",)
        )

        local_shard = torch.randn(4, 8, device=self.device_type)
        full_shape = torch.Size([16, 8])

        # With requires_grad=True
        msdt = MemoryShardedDTensor.from_local_shard(
            local_shard=local_shard,
            full_shape=full_shape,
            shard_dim=0,
            device_mesh=device_mesh,
            mesh_dim="dp",
            requires_grad=True,
        )
        self.assertTrue(msdt.requires_grad)

        # With requires_grad=False (default) - create fresh tensor
        local_shard_no_grad = torch.randn(4, 8, device=self.device_type)
        msdt_no_grad = MemoryShardedDTensor.from_local_shard(
            local_shard=local_shard_no_grad,
            full_shape=full_shape,
            shard_dim=0,
            device_mesh=device_mesh,
            mesh_dim="dp",
            requires_grad=False,
        )
        self.assertFalse(msdt_no_grad.requires_grad)

    @with_comms
    def test_from_local_shard_roundtrip(self):
        """Test from_local_shard followed by unshard recovers correct data."""
        device_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("dp",)
        )

        rank = dist.get_rank()
        # Create consistent shards across ranks
        torch.manual_seed(42)
        full_data = torch.randn(16, 8)

        # Each rank takes its slice
        local_shard = full_data[rank * 4 : (rank + 1) * 4].to(self.device_type)
        full_shape = torch.Size([16, 8])

        msdt = MemoryShardedDTensor.from_local_shard(
            local_shard=local_shard,
            full_shape=full_shape,
            shard_dim=0,
            device_mesh=device_mesh,
            mesh_dim="dp",
        )

        # Unshard and verify
        unsharded = msdt.unshard()

        self.assertEqual(unsharded.shape, torch.Size([16, 8]))
        self.assertTrue(
            torch.allclose(unsharded.to_local(), full_data.to(self.device_type))
        )

    @with_comms
    def test_from_local_shard_invalid_mesh_dim_name(self):
        """Test from_local_shard raises on invalid mesh_dim name."""
        device_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("dp",)
        )

        local_shard = torch.randn(4, 8, device=self.device_type)
        full_shape = torch.Size([16, 8])

        with self.assertRaises(ValueError):
            MemoryShardedDTensor.from_local_shard(
                local_shard=local_shard,
                full_shape=full_shape,
                shard_dim=0,
                device_mesh=device_mesh,
                mesh_dim="nonexistent",  # Invalid
            )

    @with_comms
    def test_from_local_shard_invalid_mesh_dim_index(self):
        """Test from_local_shard raises on invalid mesh_dim index."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        local_shard = torch.randn(4, 8, device=self.device_type)
        full_shape = torch.Size([16, 8])

        with self.assertRaises(ValueError):
            MemoryShardedDTensor.from_local_shard(
                local_shard=local_shard,
                full_shape=full_shape,
                shard_dim=0,
                device_mesh=device_mesh,
                mesh_dim=5,  # Invalid - out of range
            )


class TestUnshard(DTensorTestBase):
    """Tests for MemoryShardedDTensor.unshard() method."""

    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_unshard_basic(self):
        """Test unshard reconstructs correct shape."""
        device_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("dp",)
        )

        rank = dist.get_rank()
        local_shard = torch.randn(4, 8, device=self.device_type)
        full_shape = torch.Size([16, 8])

        msdt = MemoryShardedDTensor.from_local_shard(
            local_shard=local_shard,
            full_shape=full_shape,
            shard_dim=0,
            device_mesh=device_mesh,
            mesh_dim="dp",
        )

        unsharded = msdt.unshard()

        self.assertIsInstance(unsharded, DTensor)
        self.assertEqual(unsharded.shape, torch.Size([16, 8]))

    @with_comms
    def test_unshard_data_correctness(self):
        """Test unshard returns correct data from all ranks."""
        device_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("dp",)
        )

        rank = dist.get_rank()
        # Create reproducible data per rank
        torch.manual_seed(42)
        full_data = torch.randn(16, 8)

        # Each rank takes its slice
        local_shard = full_data[rank * 4 : (rank + 1) * 4].to(self.device_type)
        full_shape = torch.Size([16, 8])

        msdt = MemoryShardedDTensor.from_local_shard(
            local_shard=local_shard,
            full_shape=full_shape,
            shard_dim=0,
            device_mesh=device_mesh,
            mesh_dim="dp",
        )

        unsharded = msdt.unshard()

        # Every rank should have the same full tensor
        self.assertTrue(
            torch.allclose(unsharded.to_local(), full_data.to(self.device_type))
        )

    @with_comms
    def test_unshard_preserves_requires_grad(self):
        """Test unshard preserves requires_grad flag."""
        device_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("dp",)
        )

        local_shard = torch.randn(4, 8, device=self.device_type, requires_grad=True)
        full_shape = torch.Size([16, 8])

        msdt = MemoryShardedDTensor.from_local_shard(
            local_shard=local_shard,
            full_shape=full_shape,
            shard_dim=0,
            device_mesh=device_mesh,
            mesh_dim="dp",
            requires_grad=True,
        )

        unsharded = msdt.unshard()
        self.assertTrue(unsharded.requires_grad)

    @with_comms
    def test_unshard_uneven_sharding(self):
        """Test unshard correctly handles uneven sharding."""
        device_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("dp",)
        )

        rank = dist.get_rank()
        torch.manual_seed(42)
        # 13 rows: ranks get 4, 4, 4, 1
        full_data = torch.randn(13, 8)

        if rank < 3:
            local_shard = full_data[rank * 4 : (rank + 1) * 4].to(self.device_type)
        else:
            local_shard = full_data[12:13].to(self.device_type)

        full_shape = torch.Size([13, 8])

        msdt = MemoryShardedDTensor.from_local_shard(
            local_shard=local_shard,
            full_shape=full_shape,
            shard_dim=0,
            device_mesh=device_mesh,
            mesh_dim="dp",
        )

        unsharded = msdt.unshard()

        self.assertEqual(unsharded.shape, torch.Size([13, 8]))
        self.assertTrue(
            torch.allclose(unsharded.to_local(), full_data.to(self.device_type))
        )


class TestGetAllGatherInput(DTensorTestBase):
    """Tests for get_all_gather_input method."""

    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_get_all_gather_input_basic(self):
        """Test get_all_gather_input returns 1D flattened tensor."""
        device_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("dp",)
        )

        local_shard = torch.randn(4, 8, device=self.device_type)
        full_shape = torch.Size([16, 8])

        msdt = MemoryShardedDTensor.from_local_shard(
            local_shard=local_shard,
            full_shape=full_shape,
            shard_dim=0,
            device_mesh=device_mesh,
            mesh_dim="dp",
        )

        # Get all-gather input
        all_gather_input = msdt.get_all_gather_input()

        # Should be a plain tensor (not DTensor)
        self.assertIsInstance(all_gather_input, torch.Tensor)
        self.assertNotIsInstance(all_gather_input, DTensor)

        # Should be 1D
        self.assertEqual(all_gather_input.ndim, 1)

        # Should have correct size (4 rows * 8 cols = 32 elements per shard)
        self.assertEqual(all_gather_input.numel(), 4 * 8)

    @with_comms
    def test_get_all_gather_input_dtype_conversion(self):
        """Test get_all_gather_input with dtype conversion."""
        device_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("dp",)
        )

        local_shard = torch.randn(4, 8, device=self.device_type, dtype=torch.float32)
        full_shape = torch.Size([16, 8])

        msdt = MemoryShardedDTensor.from_local_shard(
            local_shard=local_shard,
            full_shape=full_shape,
            shard_dim=0,
            device_mesh=device_mesh,
            mesh_dim="dp",
        )

        # Get all-gather input with float16 dtype
        all_gather_input = msdt.get_all_gather_input(dtype=torch.float16)

        self.assertEqual(all_gather_input.dtype, torch.float16)


class TestDistributeStorage(DTensorTestBase):
    """Tests for distribute_storage factory function."""

    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_distribute_storage_basic(self):
        """Test distribute_storage creates MemoryShardedDTensor."""
        from torch.distributed.tensor import distribute_storage, distribute_tensor

        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(16, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = distribute_storage(dtensor, dim=0, mesh_dim=0)

        self.assertIsInstance(msdt, MemoryShardedDTensor)
        self.assertEqual(msdt.shape, torch.Size([4, 8]))
        self.assertEqual(msdt.full_shape, torch.Size([16, 8]))

    @with_comms
    def test_distribute_storage_dim1(self):
        """Test distribute_storage on dim 1."""
        from torch.distributed.tensor import distribute_storage, distribute_tensor

        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(8, 16, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = distribute_storage(dtensor, dim=1, mesh_dim=0)

        self.assertEqual(msdt.shape, torch.Size([8, 4]))
        self.assertEqual(msdt.full_shape, torch.Size([8, 16]))

    @with_comms
    def test_distribute_storage_with_mesh_dim_name(self):
        """Test distribute_storage with mesh_dim as string name."""
        from torch.distributed.tensor import distribute_storage, distribute_tensor

        device_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("dp",)
        )

        full_tensor = torch.randn(16, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = distribute_storage(dtensor, dim=0, mesh_dim="dp")

        self.assertIsInstance(msdt, MemoryShardedDTensor)
        self.assertEqual(msdt.storage_spec.mesh_dims[0], "dp")

    @with_comms
    def test_distribute_storage_roundtrip(self):
        """Test distribute_storage followed by unshard recovers data."""
        from torch.distributed.tensor import distribute_storage, distribute_tensor

        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(16, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = distribute_storage(dtensor, dim=0, mesh_dim=0)
        unsharded = msdt.unshard()

        self.assertTrue(
            torch.allclose(unsharded.to_local(), full_tensor)
        )

    @with_comms
    def test_distribute_storage_uneven(self):
        """Test distribute_storage with uneven tensor size."""
        from torch.distributed.tensor import distribute_storage, distribute_tensor

        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        # 13 rows, 4 ranks: ceil(13/4) = 4 padded shard size
        full_tensor = torch.randn(13, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = distribute_storage(dtensor, dim=0, mesh_dim=0)

        self.assertEqual(msdt.full_shape, torch.Size([13, 8]))
        self.assertEqual(msdt.storage_spec.padded_shard_sizes[0], 4)

        # Unshard and verify
        unsharded = msdt.unshard()
        self.assertTrue(
            torch.allclose(unsharded.to_local(), full_tensor)
        )


class TestDistributeBlockStorage(DTensorTestBase):
    """Tests for distribute_block_storage factory function."""

    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_distribute_block_storage_single_dim(self):
        """Test distribute_block_storage with single dimension (like distribute_storage)."""
        from torch.distributed.tensor import distribute_block_storage, distribute_tensor

        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(16, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = distribute_block_storage(dtensor, shard_dims=[0], mesh_dims=[0])

        self.assertIsInstance(msdt, MemoryShardedDTensor)
        self.assertEqual(msdt.shape, torch.Size([4, 8]))
        self.assertEqual(msdt.full_shape, torch.Size([16, 8]))

    @with_comms
    def test_distribute_block_storage_roundtrip(self):
        """Test distribute_block_storage followed by unshard recovers data."""
        from torch.distributed.tensor import distribute_block_storage, distribute_tensor

        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(16, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = distribute_block_storage(dtensor, shard_dims=[0])
        unsharded = msdt.unshard()

        self.assertTrue(
            torch.allclose(unsharded.to_local(), full_tensor)
        )


if __name__ == "__main__":
    run_tests()
