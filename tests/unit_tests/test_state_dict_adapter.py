# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.models.deepseek_v3 import deepseekv3_configs
from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter
from torchtitan.models.glm5 import glm5_configs
from torchtitan.models.glm5.state_dict_adapter import Glm5StateDictAdapter


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


class Glm5DistributedStateDictAdapterTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_tp_sharded_fused_experts_roundtrip(self) -> None:
        """HF gate/up fusion must preserve global order across TP shards."""

        config = glm5_configs["debugmodel"]()
        adapter = Glm5StateDictAdapter(config, hf_assets_path=None)
        mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("tp",),
        )
        num_experts, hidden_dim, dim = 8, 256, 256
        gate = torch.arange(
            num_experts * hidden_dim * dim,
            dtype=torch.float32,
            device=self.device_type,
        ).reshape(num_experts, hidden_dim, dim)
        up = gate + gate.numel()
        down = torch.arange(
            num_experts * dim * hidden_dim,
            dtype=torch.float32,
            device=self.device_type,
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
        self.assertIsInstance(fused, DTensor)
        self.assertEqual(fused.placements, (Shard(1),))
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
            self.assertIsInstance(restored[key], DTensor)
            self.assertEqual(restored[key].placements, (expected_placement,))
            torch.testing.assert_close(restored[key].full_tensor(), expected)
