# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard

from torchtitan.models.deepseek_v3 import deepseekv3_configs
from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter
from torchtitan.models.glm5 import glm5_configs
from torchtitan.models.glm5.state_dict_adapter import Glm5StateDictAdapter


def _run_glm5_tp_sharded_fused_experts_roundtrip(
    rank: int,
    world_size: int,
    rendezvous_uri: str,
) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=rendezvous_uri,
        rank=rank,
        world_size=world_size,
    )
    try:
        config = glm5_configs["debugmodel"]()
        adapter = Glm5StateDictAdapter(config, hf_assets_path=None)
        mesh = init_device_mesh(
            "cpu",
            (world_size,),
            mesh_dim_names=("tp",),
        )
        num_experts, hidden_dim, dim = 8, 256, 256
        gate = torch.arange(
            num_experts * hidden_dim * dim,
            dtype=torch.float32,
        ).reshape(num_experts, hidden_dim, dim)
        up = gate + gate.numel()
        down = torch.arange(
            num_experts * dim * hidden_dim,
            dtype=torch.float32,
        ).reshape(num_experts, dim, hidden_dim)
        gate_dtensor = distribute_tensor(gate, mesh, (Shard(1),))
        up_dtensor = distribute_tensor(up, mesh, (Shard(1),))
        down_dtensor = distribute_tensor(down, mesh, (Shard(2),))

        hf_state = adapter.to_hf(
            {
                "layers.1.moe.routed_experts.inner_experts.w1_EFD": gate_dtensor,
                "layers.1.moe.routed_experts.inner_experts.w3_EFD": up_dtensor,
                "layers.1.moe.routed_experts.inner_experts.w2_EDF": down_dtensor,
            }
        )

        fused = hf_state["model.layers.1.mlp.experts.gate_up_proj"]
        if not isinstance(fused, DTensor):
            raise AssertionError(f"expected DTensor, got {type(fused)}")
        if fused.placements != (Shard(1),):
            raise AssertionError(f"unexpected fused placements: {fused.placements}")
        torch.testing.assert_close(
            fused.full_tensor(),
            torch.cat((gate, up), dim=1),
        )

        restored = adapter.from_hf(hf_state)
        for key, expected, expected_placement in (
            (
                "layers.1.moe.routed_experts.inner_experts.w1_EFD",
                gate,
                Shard(1),
            ),
            (
                "layers.1.moe.routed_experts.inner_experts.w3_EFD",
                up,
                Shard(1),
            ),
            (
                "layers.1.moe.routed_experts.inner_experts.w2_EDF",
                down,
                Shard(2),
            ),
        ):
            value = restored[key]
            if not isinstance(value, DTensor):
                raise AssertionError(f"expected DTensor for {key}, got {type(value)}")
            if value.placements != (expected_placement,):
                raise AssertionError(
                    f"unexpected placements for {key}: {value.placements}"
                )
            torch.testing.assert_close(value.full_tensor(), expected)
    finally:
        dist.destroy_process_group()


class DeepSeekV3StateDictAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory = tempfile.TemporaryDirectory()
        cls._owns_process_group = not dist.is_initialized()
        if cls._owns_process_group:
            dist.init_process_group(
                backend="gloo",
                init_method=f"file://{cls._temporary_directory.name}/rendezvous",
                rank=0,
                world_size=1,
            )

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._owns_process_group:
            dist.destroy_process_group()
        cls._temporary_directory.cleanup()

    def test_to_hf_handles_replicated_grouped_experts(self) -> None:
        config = deepseekv3_configs["debugmodel"](
            attn_backend="flex",
            moe_comm_backend="standard",
        )
        adapter = DeepSeekV3StateDictAdapter(config, hf_assets_path=None)
        mesh = init_device_mesh(
            "cpu",
            (1, 1),
            mesh_dim_names=("replicate", "shard"),
        )
        local_weight = torch.arange(8 * 2 * 3, dtype=torch.float32).reshape(8, 2, 3)
        grouped_expert_weight = DTensor.from_local(
            local_weight,
            mesh,
            (Replicate(), Shard(0)),
            run_check=False,
        )

        hf_state_dict = adapter.to_hf(
            {"layers.1.moe.routed_experts.inner_experts.w1_EFD": grouped_expert_weight}
        )

        expected_keys = {
            f"model.layers.1.mlp.experts.{expert}.gate_proj.weight"
            for expert in range(8)
        }
        self.assertEqual(set(hf_state_dict), expected_keys)
        for expert in range(8):
            key = f"model.layers.1.mlp.experts.{expert}.gate_proj.weight"
            self.assertIsInstance(hf_state_dict[key], DTensor)
            torch.testing.assert_close(
                hf_state_dict[key].to_local(),
                local_weight[expert],
            )


class Glm5DistributedStateDictAdapterTest(unittest.TestCase):
    def test_tp_sharded_fused_experts_roundtrip(self) -> None:
        """HF gate/up fusion must preserve global order across TP shards."""

        world_size = 2
        with tempfile.TemporaryDirectory() as temp_dir:
            rendezvous_uri = (Path(temp_dir) / "glm5_tp_rendezvous").as_uri()
            torch.multiprocessing.spawn(
                _run_glm5_tp_sharded_fused_experts_roundtrip,
                args=(world_size, rendezvous_uri),
                nprocs=world_size,
                join=True,
            )
