# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.glm5 import build_model_config
from torchtitan.models.glm5.state_dict_adapter import Glm5StateDictAdapter


class TestGlm5StateDictAdapter(unittest.TestCase):
    def test_debug_model_hf_roundtrip(self) -> None:
        config = build_model_config(
            "glm5_2_debugmodel",
            parallelism=ParallelismConfig(),
            seq_len=8,
        )
        model = config.build()
        model.init_states()
        native = model.state_dict()

        adapter = Glm5StateDictAdapter(config, hf_assets_path=None)
        restored = adapter.from_hf(adapter.to_hf(native))

        self.assertEqual(native.keys(), restored.keys())
        for key in native:
            torch.testing.assert_close(native[key], restored[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
