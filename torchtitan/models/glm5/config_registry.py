# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.trainer import Trainer

from . import model_registry


def glm5_debugmodel() -> Trainer.Config:
    """Short, local-assets configuration for the single-device GLM-5 model."""
    model_spec = model_registry("debugmodel")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=default_adamw(lr=8e-4),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=128,
            steps=10,
        ),
        parallelism=ParallelismConfig(),
        checkpoint=CheckpointManager.Config(interval=10),
        activation_checkpoint=None,
    )
