# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.glm5 import build_model_config, MODEL_FLAVORS
from torchtitan.models.glm5.dsa import (
    Glm5FlexAttention,
    Glm5KVAllGatherCPFlexAttention,
)
from torchtitan.models.glm5.model import Glm5Attention


class TestGlm5Config(unittest.TestCase):
    def test_all_flavors_build(self) -> None:
        parallelism = ParallelismConfig()
        for flavor in MODEL_FLAVORS:
            with self.subTest(flavor=flavor):
                config = build_model_config(
                    flavor,
                    parallelism=parallelism,
                    seq_len=128,
                )
                self.assertEqual(config.max_context_length, 128)
                self.assertGreater(len(config.layers), 0)
                self.assertIsInstance(config.layers[0].attention, Glm5Attention.Config)

    def test_debug_flavor_preserves_dense_to_moe_transition(self) -> None:
        config = build_model_config(
            "glm5_2_debugmodel",
            parallelism=ParallelismConfig(),
            seq_len=128,
        )
        self.assertTrue(
            all(layer.feed_forward is not None for layer in config.layers[:3])
        )
        self.assertTrue(all(layer.moe is not None for layer in config.layers[3:]))

    def test_rejects_context_beyond_flavor_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds max context length"):
            build_model_config(
                "debugmodel",
                parallelism=ParallelismConfig(),
                seq_len=129,
            )

    def test_debug_model_builds_and_initializes(self) -> None:
        config = build_model_config(
            "glm5_2_debugmodel",
            parallelism=ParallelismConfig(),
            seq_len=8,
        )
        model = config.build()
        model.init_states()
        self.assertGreater(sum(parameter.numel() for parameter in model.parameters()), 0)

    def test_cp_selects_official_kv_all_gather_attention(self) -> None:
        eager_config = build_model_config(
            "debugmodel",
            parallelism=ParallelismConfig(),
            seq_len=128,
        )
        cp_config = build_model_config(
            "debugmodel",
            parallelism=ParallelismConfig(context_parallel_degree=2),
            seq_len=128,
        )
        self.assertIsInstance(
            eager_config.layers[0].attention.inner_attention,
            Glm5FlexAttention.Config,
        )
        self.assertIsInstance(
            cp_config.layers[0].attention.inner_attention,
            Glm5KVAllGatherCPFlexAttention.Config,
        )


if __name__ == "__main__":
    unittest.main()
