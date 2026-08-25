# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# pyrefly: ignore-errors

"""Opt-in Triton kernels for GLM-5 DSA.

The kernels use the public Triton language. CUDA loads them through Triton;
TorchTitanTurbo reuses the same mathematical kernels with Triton-Ascend and
provides the NPU-specific override registration.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from torchtitan.config import derive, override
from torchtitan.models.glm5.model import DSAIndexerTopK, SparseMLA


@triton.jit
def _index_scores_kernel(
    q,
    k,
    weights,
    attention_mask,
    scores,
    NUM_KEYS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
) -> None:
    query = tl.program_id(0).to(tl.int64)
    key = (
        tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    ).to(tl.int64)
    head_dim = tl.arange(0, BLOCK_H).to(tl.int64)
    key_mask = key < NUM_KEYS
    score = tl.zeros((BLOCK_K,), dtype=tl.float32)
    k_value = tl.load(
        k + key[:, None] * HEAD_DIM + head_dim[None, :],
        mask=key_mask[:, None] & (head_dim[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)

    for head in tl.static_range(0, NUM_HEADS):
        q_offset = (query * NUM_HEADS + head) * HEAD_DIM + head_dim
        q_value = tl.load(q + q_offset, mask=head_dim < HEAD_DIM, other=0.0).to(
            tl.float32
        )
        dot = tl.sum(k_value * q_value[None, :], axis=1)
        head_weight = tl.load(weights + query * NUM_HEADS + head).to(tl.float32)
        score += tl.maximum(dot * SOFTMAX_SCALE, 0.0) * head_weight

    score += tl.load(
        attention_mask + query * NUM_KEYS + key,
        mask=key_mask,
        other=float("-inf"),
    ).to(tl.float32)
    tl.store(scores + query * NUM_KEYS + key, score, mask=key_mask)


@triton.jit
def _sparse_mla_scores_kernel(
    q,
    kv,
    attention_mask,
    indices,
    scores,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_KEYS: tl.constexpr,
    TOP_K: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
) -> None:
    query_head = tl.program_id(0).to(tl.int64)
    query = query_head // NUM_HEADS
    head = query_head % NUM_HEADS
    slot = (
        tl.program_id(1) * BLOCK_S + tl.arange(0, BLOCK_S)
    ).to(tl.int64)
    head_dim = tl.arange(0, BLOCK_H).to(tl.int64)
    slot_mask = slot < TOP_K
    index = tl.load(
        indices + query * TOP_K + slot,
        mask=slot_mask,
        other=-1,
    ).to(tl.int64)
    valid = slot_mask & (index >= 0) & (index < NUM_KEYS)
    safe_index = tl.maximum(index, 0)

    q_value = tl.load(
        q + query_head * HEAD_DIM + head_dim,
        mask=head_dim < HEAD_DIM,
        other=0.0,
    ).to(tl.float32)
    kv_value = tl.load(
        kv + safe_index[:, None] * HEAD_DIM + head_dim[None, :],
        mask=valid[:, None] & (head_dim[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)
    score = tl.sum(kv_value * q_value[None, :], axis=1) * SCALE
    mask_value = tl.load(
        attention_mask + query * NUM_KEYS + safe_index,
        mask=valid,
        other=float("-inf"),
    ).to(tl.float32)
    allowed = valid & (mask_value == 0.0)
    score = tl.where(allowed, score, float("-inf"))
    tl.store(scores + query_head * TOP_K + slot, score, mask=slot_mask)


@triton.jit
def _sparse_mla_output_kernel(
    probabilities,
    kv,
    indices,
    output,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_KEYS: tl.constexpr,
    TOP_K: tl.constexpr,
    LATENT_DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
) -> None:
    query_head = tl.program_id(0).to(tl.int64)
    query = query_head // NUM_HEADS
    channel = (
        tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    ).to(tl.int64)
    channel_mask = channel < LATENT_DIM
    accumulator = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for slot_start in tl.static_range(0, TOP_K, BLOCK_S):
        slot = (slot_start + tl.arange(0, BLOCK_S)).to(tl.int64)
        slot_mask = slot < TOP_K
        index = tl.load(
            indices + query * TOP_K + slot,
            mask=slot_mask,
            other=-1,
        ).to(tl.int64)
        valid = slot_mask & (index >= 0) & (index < NUM_KEYS)
        safe_index = tl.maximum(index, 0)
        probability = tl.load(
            probabilities + query_head * TOP_K + slot,
            mask=slot_mask,
            other=0.0,
        ).to(tl.float32)
        value = tl.load(
            kv + safe_index[:, None] * HEAD_DIM + channel[None, :],
            mask=valid[:, None] & channel_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(probability[:, None] * value, axis=0)

    tl.store(
        output + query_head * LATENT_DIM + channel,
        accumulator,
        mask=channel_mask,
    )


@triton.jit
def _sparse_mla_probability_grad_kernel(
    kv,
    indices,
    grad_output,
    probability_grad,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_KEYS: tl.constexpr,
    TOP_K: tl.constexpr,
    LATENT_DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
) -> None:
    query_head = tl.program_id(0).to(tl.int64)
    query = query_head // NUM_HEADS
    slot = (
        tl.program_id(1) * BLOCK_S + tl.arange(0, BLOCK_S)
    ).to(tl.int64)
    slot_mask = slot < TOP_K
    index = tl.load(
        indices + query * TOP_K + slot,
        mask=slot_mask,
        other=-1,
    ).to(tl.int64)
    valid = slot_mask & (index >= 0) & (index < NUM_KEYS)
    safe_index = tl.maximum(index, 0)
    grad_probability = tl.zeros((BLOCK_S,), dtype=tl.float32)

    for channel_start in tl.static_range(0, LATENT_DIM, BLOCK_C):
        channel = (channel_start + tl.arange(0, BLOCK_C)).to(tl.int64)
        channel_mask = channel < LATENT_DIM
        grad_value = tl.load(
            grad_output + query_head * LATENT_DIM + channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        selected_value = tl.load(
            kv + safe_index[:, None] * HEAD_DIM + channel[None, :],
            mask=valid[:, None] & channel_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        grad_probability += tl.sum(
            selected_value * grad_value[None, :],
            axis=1,
        )
    tl.store(
        probability_grad + query_head * TOP_K + slot,
        grad_probability,
        mask=slot_mask,
    )


@triton.jit
def _sparse_mla_softmax_grad_kernel(
    probabilities_fp32,
    probability_grad,
    score_grad,
    TOP_K: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_S: tl.constexpr,
) -> None:
    query_head = tl.program_id(0).to(tl.int64)
    slot = tl.arange(0, BLOCK_S).to(tl.int64)
    slot_mask = slot < TOP_K
    probability = tl.load(
        probabilities_fp32 + query_head * TOP_K + slot,
        mask=slot_mask,
        other=0.0,
    ).to(tl.float32)
    grad_probability = tl.load(
        probability_grad + query_head * TOP_K + slot,
        mask=slot_mask,
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(probability * grad_probability, axis=0)
    grad_score = probability * (grad_probability - delta) * SCALE
    tl.store(
        score_grad + query_head * TOP_K + slot,
        grad_score,
        mask=slot_mask,
    )


@triton.jit
def _sparse_mla_query_grad_kernel(
    score_grad,
    kv,
    indices,
    grad_q,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_KEYS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
) -> None:
    query_head = tl.program_id(0).to(tl.int64)
    query = query_head // NUM_HEADS
    head_dim = (
        tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    ).to(tl.int64)
    head_mask = head_dim < HEAD_DIM
    accumulator = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for slot_start in tl.static_range(0, TOP_K, BLOCK_S):
        slot = (slot_start + tl.arange(0, BLOCK_S)).to(tl.int64)
        slot_mask = slot < TOP_K
        index = tl.load(
            indices + query * TOP_K + slot,
            mask=slot_mask,
            other=-1,
        ).to(tl.int64)
        valid = slot_mask & (index >= 0) & (index < NUM_KEYS)
        safe_index = tl.maximum(index, 0)
        grad_score = tl.load(
            score_grad + query_head * TOP_K + slot,
            mask=slot_mask,
            other=0.0,
        ).to(tl.float32)
        key_value = tl.load(
            kv + safe_index[:, None] * HEAD_DIM + head_dim[None, :],
            mask=valid[:, None] & head_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(grad_score[:, None] * key_value, axis=0)

    tl.store(
        grad_q + query_head * HEAD_DIM + head_dim,
        accumulator,
        mask=head_mask,
    )


@triton.jit
def _sparse_mla_kv_grad_kernel(
    score_grad,
    probabilities,
    q,
    grad_output,
    indices,
    grad_kv,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_KEYS: tl.constexpr,
    TOP_K: tl.constexpr,
    LATENT_DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
) -> None:
    query_head = tl.program_id(0).to(tl.int64)
    query = query_head // NUM_HEADS
    slot = (
        tl.program_id(1) * BLOCK_S + tl.arange(0, BLOCK_S)
    ).to(tl.int64)
    head_dim = (
        tl.program_id(2) * BLOCK_H + tl.arange(0, BLOCK_H)
    ).to(tl.int64)
    slot_mask = slot < TOP_K
    head_mask = head_dim < HEAD_DIM
    index = tl.load(
        indices + query * TOP_K + slot,
        mask=slot_mask,
        other=-1,
    ).to(tl.int64)
    valid = slot_mask & (index >= 0) & (index < NUM_KEYS)
    safe_index = tl.maximum(index, 0)
    grad_score = tl.load(
        score_grad + query_head * TOP_K + slot,
        mask=slot_mask,
        other=0.0,
    ).to(tl.float32)
    probability = tl.load(
        probabilities + query_head * TOP_K + slot,
        mask=slot_mask,
        other=0.0,
    ).to(tl.float32)
    query_value = tl.load(
        q + query_head * HEAD_DIM + head_dim,
        mask=head_mask,
        other=0.0,
    ).to(tl.float32)
    value_mask = head_dim < LATENT_DIM
    grad_value = tl.load(
        grad_output + query_head * LATENT_DIM + head_dim,
        mask=value_mask,
        other=0.0,
    ).to(tl.float32)
    contribution = grad_score[:, None] * query_value[None, :]
    contribution += tl.where(
        value_mask[None, :],
        probability[:, None] * grad_value[None, :],
        0.0,
    )
    tl.atomic_add(
        grad_kv + safe_index[:, None] * HEAD_DIM + head_dim[None, :],
        contribution,
        mask=valid[:, None] & head_mask[None, :],
    )


def triton_index_scores(
    q_QNH: torch.Tensor,
    k_KH: torch.Tensor,
    weights_QN: torch.Tensor,
    attention_mask_QK: torch.Tensor,
    *,
    softmax_scale: float,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute exact DSA index scores without materializing ``[N,Q,K]``."""

    if q_QNH.ndim != 3 or k_KH.ndim != 2:
        raise ValueError("DSA indexer q and k must have rank 3 and 2.")
    Q, N, H = q_QNH.shape
    K = k_KH.shape[0]
    if k_KH.shape != (K, H):
        raise ValueError("DSA indexer k must have shape [K, H].")
    if weights_QN.shape != (Q, N):
        raise ValueError("DSA indexer weights must have shape [Q, N].")
    if attention_mask_QK.shape != (Q, K):
        raise ValueError("DSA indexer attention mask must have shape [Q, K].")
    if any(
        tensor.device != q_QNH.device
        for tensor in (k_KH, weights_QN, attention_mask_QK)
    ):
        raise ValueError("DSA indexer inputs must be on the same device.")
    if not all(
        tensor.is_floating_point()
        for tensor in (q_QNH, k_KH, weights_QN, attention_mask_QK)
    ):
        raise ValueError("DSA indexer inputs must use floating dtypes.")
    q_QNH = q_QNH.contiguous()
    k_KH = k_KH.contiguous()
    weights_QN = weights_QN.contiguous()
    attention_mask_QK = attention_mask_QK.contiguous()
    if H > 256:
        raise ValueError("Triton DSA indexer supports head_dim <= 256")
    if output is None:
        output = torch.empty((Q, K), dtype=torch.float32, device=q_QNH.device)
    if output.shape != (Q, K) or output.dtype != torch.float32:
        raise ValueError("index score output must be float32 with shape [Q, K]")
    if output.device != q_QNH.device or not output.is_contiguous():
        raise ValueError("index score output must be contiguous on the input device")

    block_k = 32
    block_h = triton.next_power_of_2(H)
    _index_scores_kernel[(Q, triton.cdiv(K, block_k))](
        q_QNH,
        k_KH,
        weights_QN,
        attention_mask_QK,
        output,
        NUM_KEYS=K,
        NUM_HEADS=N,
        HEAD_DIM=H,
        SOFTMAX_SCALE=softmax_scale,
        BLOCK_K=block_k,
        BLOCK_H=block_h,
        num_warps=4,
    )
    return output


def _triton_sparse_mla_forward(
    q_QNH: torch.Tensor,
    kv_K1H: torch.Tensor,
    attention_masks_1QK: torch.Tensor,
    topk_indices_QS: torch.Tensor,
    *,
    scale: float,
    latent_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if q_QNH.ndim != 3 or kv_K1H.ndim != 3:
        raise ValueError("GLM-5 SparseMLA q and compressed kv must have rank 3.")
    Q, N, H = q_QNH.shape
    K = kv_K1H.shape[0]
    if kv_K1H.shape != (K, 1, H):
        raise ValueError("GLM-5 SparseMLA kv must have shape [K, 1, H].")
    if not 0 < latent_dim < H:
        raise ValueError("SparseMLA latent_dim must be in [1, head_dim).")
    if topk_indices_QS.ndim != 2 or topk_indices_QS.shape[0] != Q:
        raise ValueError("GLM-5 DSA top-k indices must have shape [Q, S].")
    if attention_masks_1QK.layout != torch.strided:
        raise ValueError("GLM-5 DSA requires dense attention_masks.")
    if attention_masks_1QK.shape != (1, Q, K):
        raise ValueError("attention_masks must have shape [1, Q, K].")
    if any(
        tensor.device != q_QNH.device
        for tensor in (kv_K1H, attention_masks_1QK, topk_indices_QS)
    ):
        raise ValueError("SparseMLA inputs must be on the same device.")
    if not q_QNH.is_floating_point() or not kv_K1H.is_floating_point():
        raise ValueError("SparseMLA q and compressed kv must use floating dtypes.")
    if not attention_masks_1QK.is_floating_point():
        raise ValueError("attention_masks must use a floating additive dtype.")
    if topk_indices_QS.dtype not in (torch.int32, torch.int64):
        raise ValueError("top-k indices must use int32 or int64.")

    q_QNH = q_QNH.contiguous()
    kv_K1H = kv_K1H.contiguous()
    attention_masks_1QK = attention_masks_1QK.contiguous()
    topk_indices_QS = topk_indices_QS.contiguous()
    S = topk_indices_QS.shape[1]
    if S == 0:
        raise ValueError("SparseMLA top-k dimension must be positive.")
    if H > 1024:
        raise ValueError("Triton SparseMLA supports head_dim <= 1024")

    scores_QNS = torch.empty((Q, N, S), dtype=torch.float32, device=q_QNH.device)
    block_s = 16
    block_h = triton.next_power_of_2(H)
    _sparse_mla_scores_kernel[(Q * N, triton.cdiv(S, block_s))](
        q_QNH,
        kv_K1H,
        attention_masks_1QK,
        topk_indices_QS,
        scores_QNS,
        NUM_HEADS=N,
        HEAD_DIM=H,
        NUM_KEYS=K,
        TOP_K=S,
        SCALE=scale,
        BLOCK_S=block_s,
        BLOCK_H=block_h,
        num_warps=4,
    )
    safe_indices_QS = topk_indices_QS.clamp_min(0).long()
    selected_mask_QS = attention_masks_1QK[0].gather(-1, safe_indices_QS)
    allowed_QS = (topk_indices_QS >= 0) & (selected_mask_QS == 0)
    probabilities_fp32_QNS = F.softmax(
        scores_QNS,
        dim=-1,
        dtype=torch.float32,
    )
    probabilities_fp32_QNS = probabilities_fp32_QNS.masked_fill(
        ~allowed_QS.unsqueeze(1), 0.0
    ).contiguous()
    probabilities_QNS = probabilities_fp32_QNS.to(q_QNH.dtype).masked_fill(
        ~allowed_QS.unsqueeze(1), 0.0
    ).contiguous()

    output_QNC = torch.empty(
        (Q, N, latent_dim), dtype=q_QNH.dtype, device=q_QNH.device
    )
    block_c = min(64, triton.next_power_of_2(latent_dim))
    _sparse_mla_output_kernel[(Q * N, triton.cdiv(latent_dim, block_c))](
        probabilities_QNS,
        kv_K1H,
        topk_indices_QS,
        output_QNC,
        NUM_HEADS=N,
        HEAD_DIM=H,
        NUM_KEYS=K,
        TOP_K=S,
        LATENT_DIM=latent_dim,
        BLOCK_S=block_s,
        BLOCK_C=block_c,
        num_warps=4,
    )
    return output_QNC, probabilities_fp32_QNS, probabilities_QNS


def _triton_sparse_mla_backward(
    q_QNH: torch.Tensor,
    kv_K1H: torch.Tensor,
    topk_indices_QS: torch.Tensor,
    probabilities_fp32_QNS: torch.Tensor,
    probabilities_QNS: torch.Tensor,
    grad_output_QNC: torch.Tensor,
    *,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_QNH = q_QNH.contiguous()
    kv_K1H = kv_K1H.contiguous()
    topk_indices_QS = topk_indices_QS.contiguous()
    probabilities_fp32_QNS = probabilities_fp32_QNS.contiguous()
    probabilities_QNS = probabilities_QNS.contiguous()
    grad_output_QNC = grad_output_QNC.contiguous()
    Q, N, H = q_QNH.shape
    K = kv_K1H.shape[0]
    S = topk_indices_QS.shape[1]
    C = grad_output_QNC.shape[-1]
    block_s = 8
    block_c = min(64, triton.next_power_of_2(C))
    block_h = min(64, triton.next_power_of_2(H))

    probability_grad_QNS = torch.empty(
        (Q, N, S),
        dtype=torch.float32,
        device=q_QNH.device,
    )
    _sparse_mla_probability_grad_kernel[
        (Q * N, triton.cdiv(S, block_s))
    ](
        kv_K1H,
        topk_indices_QS,
        grad_output_QNC,
        probability_grad_QNS,
        NUM_HEADS=N,
        HEAD_DIM=H,
        NUM_KEYS=K,
        TOP_K=S,
        LATENT_DIM=C,
        BLOCK_S=block_s,
        BLOCK_C=block_c,
        num_warps=4,
    )

    score_grad_QNS = torch.empty_like(probability_grad_QNS)
    softmax_block_s = triton.next_power_of_2(S)
    _sparse_mla_softmax_grad_kernel[(Q * N,)](
        probabilities_fp32_QNS,
        probability_grad_QNS,
        score_grad_QNS,
        TOP_K=S,
        SCALE=scale,
        BLOCK_S=softmax_block_s,
        num_warps=8,
    )

    grad_q_QNH = torch.empty_like(q_QNH)
    _sparse_mla_query_grad_kernel[(Q * N, triton.cdiv(H, block_h))](
        score_grad_QNS,
        kv_K1H,
        topk_indices_QS,
        grad_q_QNH,
        NUM_HEADS=N,
        HEAD_DIM=H,
        NUM_KEYS=K,
        TOP_K=S,
        BLOCK_S=block_s,
        BLOCK_H=block_h,
        num_warps=4,
    )

    grad_kv_K1H_fp32 = torch.zeros(
        (K, 1, H),
        dtype=torch.float32,
        device=kv_K1H.device,
    )
    _sparse_mla_kv_grad_kernel[
        (Q * N, triton.cdiv(S, block_s), triton.cdiv(H, block_h))
    ](
        score_grad_QNS,
        probabilities_QNS,
        q_QNH,
        grad_output_QNC,
        topk_indices_QS,
        grad_kv_K1H_fp32,
        NUM_HEADS=N,
        HEAD_DIM=H,
        NUM_KEYS=K,
        TOP_K=S,
        LATENT_DIM=C,
        BLOCK_S=block_s,
        BLOCK_H=block_h,
        num_warps=4,
    )
    return grad_q_QNH, grad_kv_K1H_fp32.to(kv_K1H.dtype)


class _TritonSparseMLAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, attention_mask, indices, scale, latent_dim):
        ctx.scale = scale
        output, probabilities_fp32, probabilities = _triton_sparse_mla_forward(
            q,
            kv,
            attention_mask,
            indices,
            scale=scale,
            latent_dim=latent_dim,
        )
        ctx.save_for_backward(
            q,
            kv,
            indices,
            probabilities_fp32,
            probabilities,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, kv, indices, probabilities_fp32, probabilities = ctx.saved_tensors
        grad_q, grad_kv = _triton_sparse_mla_backward(
            q,
            kv,
            indices,
            probabilities_fp32,
            probabilities,
            grad_output,
            scale=ctx.scale,
        )
        return grad_q, grad_kv, None, None, None, None


class TritonDSAIndexerTopK(DSAIndexerTopK):
    """Compute GLM-5 DSA index scores with Triton."""

    required_device_type = "cuda"

    @dataclass(kw_only=True, slots=True)
    class Config(DSAIndexerTopK.Config):
        query_block_size: int = 1024

        def __post_init__(self) -> None:
            super().__post_init__()
            if self.query_block_size <= 0:
                raise ValueError("query_block_size must be positive")

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.query_block_size = config.query_block_size

    def forward(self, q_QNH, k_KH, weights_QN, attention_mask_QK):
        if q_QNH.device.type != self.required_device_type:
            raise RuntimeError(
                f"Triton DSA indexer requires {self.required_device_type} tensors"
            )
        topk = min(self.index_topk, k_KH.shape[0])
        indices = []
        for start in range(0, q_QNH.shape[0], self.query_block_size):
            end = min(start + self.query_block_size, q_QNH.shape[0])
            scores_QK = triton_index_scores(
                q_QNH[start:end],
                k_KH,
                weights_QN[start:end],
                attention_mask_QK[start:end],
                softmax_scale=self.softmax_scale,
            )
            indices.append(scores_QK.topk(topk, dim=-1).indices.to(torch.int32))
        return torch.cat(indices, dim=0)


class TritonSparseMLA(SparseMLA):
    """Triton SparseMLA forward and backward for selected compressed KV."""

    required_device_type = "cuda"

    @dataclass(kw_only=True, slots=True)
    class Config(SparseMLA.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if self.attention_dropout != 0.0:
            raise ValueError("Triton SparseMLA does not support dropout")

    def forward(
        self,
        q_QNH,
        kv_K1H,
        attention_masks_1QK,
        topk_indices_QS,
        *,
        scale,
        latent_dim,
    ):
        if q_QNH.device.type != self.required_device_type:
            raise RuntimeError(
                f"Triton SparseMLA requires {self.required_device_type} tensors"
            )
        return _TritonSparseMLAFunction.apply(
            q_QNH,
            kv_K1H,
            attention_masks_1QK,
            topk_indices_QS,
            scale,
            latent_dim,
        )


@override(
    target=DSAIndexerTopK.Config,
    description="Use the CUDA Triton GLM-5 DSA indexer.",
)
def triton_dsa_indexer(cfg: DSAIndexerTopK.Config) -> TritonDSAIndexerTopK.Config:
    return derive(cfg, TritonDSAIndexerTopK.Config)


@override(
    target=SparseMLA.Config,
    description="Use the CUDA Triton GLM-5 SparseMLA forward and backward.",
)
def triton_sparse_mla(cfg: SparseMLA.Config) -> TritonSparseMLA.Config:
    return derive(cfg, TritonSparseMLA.Config)


__all__ = [
    "TritonDSAIndexerTopK",
    "TritonSparseMLA",
    "triton_dsa_indexer",
    "triton_index_scores",
    "triton_sparse_mla",
]
