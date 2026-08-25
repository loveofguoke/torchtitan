# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# pyrefly: ignore-errors

"""Unregistered Triton residual-add plus RMSNorm forward prototype.

The fusion pattern follows common TorchTitan and Megatron-style transformer
kernel boundaries. It is not imported by the GLM-5 model or override registry.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _residual_rmsnorm_kernel(
    x,
    residual,
    weight,
    normalized,
    residual_out,
    DIM: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0).to(tl.int64)
    dim = tl.arange(0, BLOCK_D).to(tl.int64)
    mask = dim < DIM
    value = tl.load(x + row * DIM + dim, mask=mask, other=0.0).to(tl.float32)
    value += tl.load(
        residual + row * DIM + dim,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    mean_square = tl.sum(value * value, axis=0) / DIM
    inverse_rms = tl.rsqrt(mean_square + EPS)
    scale = tl.load(weight + dim, mask=mask, other=0.0).to(tl.float32)
    tl.store(residual_out + row * DIM + dim, value, mask=mask)
    tl.store(normalized + row * DIM + dim, value * inverse_rms * scale, mask=mask)


def triton_residual_rmsnorm(
    x_TD: torch.Tensor,
    residual_TD: torch.Tensor,
    weight_D: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized residual and the materialized residual branch."""

    if x_TD.shape != residual_TD.shape or x_TD.ndim != 2:
        raise ValueError("x and residual must have the same [T, D] shape")
    if weight_D.shape != (x_TD.shape[1],):
        raise ValueError("RMSNorm weight must have shape [D]")
    if any(tensor.device != x_TD.device for tensor in (residual_TD, weight_D)):
        raise ValueError("all tensors must be on the same device")
    if eps <= 0:
        raise ValueError("eps must be positive")

    x_TD = x_TD.contiguous()
    residual_TD = residual_TD.contiguous()
    weight_D = weight_D.contiguous()
    normalized_TD = torch.empty_like(x_TD)
    residual_out_TD = torch.empty_like(x_TD)
    dim = x_TD.shape[1]
    block_d = triton.next_power_of_2(dim)
    if block_d > 65536:
        raise ValueError("residual RMSNorm prototype supports dim <= 65536")
    _residual_rmsnorm_kernel[(x_TD.shape[0],)](
        x_TD,
        residual_TD,
        weight_D,
        normalized_TD,
        residual_out_TD,
        DIM=dim,
        EPS=eps,
        BLOCK_D=block_d,
        num_warps=8,
    )
    return normalized_TD, residual_out_TD


__all__ = ["triton_residual_rmsnorm"]
