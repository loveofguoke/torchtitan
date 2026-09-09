# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer-level GLM-5 configurations.

The model registry describes architecture; this module adds data, loss,
optimizer, schedule, precision, checkpoint, and parallelism defaults. CLI
overrides used by torchtitan-test are applied on top of this debug baseline.
"""

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.trainer import Trainer

from . import model_registry


def _glm5_debugmodel(flavor: str) -> Trainer.Config:
    model_spec = model_registry(flavor)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
            shuffle=False,
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=2 * 128,
            max_context_length=128,
            steps=10,
            disable_cuda_graphs=True,
            # Data parallel (DDP/FSDP) wraps the model via apply_fsdp_to_decoder;
            # fp32 preserves the DSA indexer's pinned fp32 semantics. A user can
            # opt into bf16 explicitly as a documented precision change.
            mixed_precision_param="float32",
        ),
        parallelism=ParallelismConfig(
            enable_sequence_parallel=True,
            context_parallel_load_balancer=None,
            spmd_backend="partial_dtensor",
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=None,
    )


def glm5_debugmodel() -> Trainer.Config:
    """Short, local-assets configuration for independent DSA indexers."""
    return _glm5_debugmodel("debugmodel")


def glm5_shared_dsa_debugmodel() -> Trainer.Config:
    """Short configuration that reuses DSA indices across adjacent layers."""
    return _glm5_debugmodel("shared_dsa_debugmodel")
