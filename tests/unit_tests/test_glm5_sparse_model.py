# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from torchtitan.distributed.pipeline_parallel import (
    _generate_llm_fqn_per_model_part,
    _split_module,
)
from torchtitan.models.glm5 import glm5_configs
from torchtitan.models.glm5.dsa import build_dsa_block_mask, Glm5FlexAttention
from torchtitan.models.glm5.parallelize import validate_glm5_index_sharing


def test_index_sharing_accepts_stage_local_groups():
    model = SimpleNamespace(index_sources=(0, 0, 2, 2), layers={"2": None, "3": None})
    validate_glm5_index_sharing(model)


def test_index_sharing_accepts_one_cross_stage_producer():
    model = SimpleNamespace(index_sources=(0, 0, 0, 0), layers={"2": None, "3": None})
    validate_glm5_index_sharing(model)


def test_index_sharing_rejects_multiple_cross_stage_producers():
    model = SimpleNamespace(index_sources=(0, 0, 2, 2), layers={"1": None, "3": None})
    with pytest.raises(ValueError, match="only one source layer"):
        validate_glm5_index_sharing(model)


def test_pp_stages_forward_shared_indices_across_boundaries():
    class FakeLayer(nn.Module):
        def __init__(self, layer_id):
            super().__init__()
            self.layer_id = layer_id

        def forward(
            self,
            hidden_TD,
            attention_masks,
            positions,
            topk_indices_TS,
            *,
            return_indices,
        ):
            if topk_indices_TS is None:
                topk_indices_TS = torch.full(
                    (hidden_TD.shape[0], 1), self.layer_id, dtype=torch.long
                )
            return hidden_TD + self.layer_id, topk_indices_TS

    config = glm5_configs["shared_dsa_debugmodel"]()
    model = config.build()
    model.tok_embeddings = None
    model.norm = None
    model.lm_head = None
    model.layers = nn.ModuleDict(
        {str(layer): FakeLayer(layer) for layer in range(len(config.layers))}
    )
    stage_fqns = _generate_llm_fqn_per_model_part(
        8, len(config.layers), input_weight=1, output_weight=1
    )
    stages = [_split_module(model, fqns) for fqns in stage_fqns]

    payload = torch.zeros(4, config.dim)
    positions = torch.arange(4)
    for stage in stages:
        if isinstance(payload, tuple):
            payload = stage(*payload, positions=positions)
        else:
            payload = stage(payload, positions=positions)

    assert isinstance(payload, torch.Tensor)
    torch.testing.assert_close(payload, torch.full_like(payload, sum(range(8))))


def test_cp_mask_uses_global_key_indices_for_local_queries():
    # CP rank 1 owns global queries 3..5; selected key IDs remain global.
    mask = torch.zeros(1, 3, 6)
    indices = torch.tensor([[0, 3], [1, 4], [2, 5]])
    block_mask = build_dsa_block_mask(indices, mask, 16)
    q = torch.arange(3)[:, None]
    k = torch.arange(6)[None, :]
    expected = torch.zeros(3, 6, dtype=torch.bool).scatter_(1, indices, True)
    assert torch.equal(block_mask.mask_mod(0, 0, q, k), expected)


def test_selected_mask_handles_padding_duplicates_and_document_boundaries():
    mask = torch.full((1, 3, 3), float("-inf"))
    mask[0, 0, 0] = mask[0, 1, 0] = mask[0, 1, 1] = mask[0, 2, 2] = 0
    indices = torch.tensor([[0, -1, -1], [0, 0, 1], [0, 2, 9]])
    bm = build_dsa_block_mask(indices, mask, 16)
    q = torch.arange(3)[:, None]
    k = torch.arange(3)[None, :]
    actual = bm.mask_mod(0, 0, q, k)
    assert torch.equal(actual, mask[0] == 0)
    assert bm.kv_num_blocks.item() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlexAttention")
def test_sparse_forward_and_gradients_match_dense_reference():
    torch.manual_seed(61)
    Q, K, N, H = 32, 32, 2, 32
    q, k, v = [torch.randn(Q, N, H, device="cuda", requires_grad=True) for _ in range(3)]
    rows = torch.arange(Q, device="cuda")
    indices = torch.stack((rows, (rows - 1).clamp_min(0), (rows - 3).clamp_min(0)), dim=1)
    mask = torch.zeros(1, Q, K, device="cuda")
    op = Glm5FlexAttention(Glm5FlexAttention.Config(block_size=16))
    actual = op(q, k, v, mask, indices, scale=H**-0.5)
    allowed = torch.zeros(Q, K, device="cuda", dtype=torch.bool)
    allowed.scatter_(1, indices, True)
    scores = torch.einsum("qnh,knh->nqk", q, k) * H**-0.5
    probs = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
    expected = torch.einsum("nqk,knh->qnh", probs, v)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    upstream = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (q, k, v), upstream, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (q, k, v), upstream)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-5, rtol=1e-4)
