# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone BF16 isolation diagnostics for the GLM-5 DSA indexer parity gap.

Run from the repo root on a CUDA host with the pinned Transformers
GLM-MoE-DSA reference installed (archive the log for the runbook):

    python -m tests.unit_tests.diagnose_glm5_indexer | tee diagnose.log

The end-to-end BF16 test (test_glm5_parity.py) passes layer-1 MoE blocks but
fails on the full logits.  This script walks that failure down to its source:

  Section 2 -- full BF16 forward: per-layer block / indexer / router mismatch
  Section 3 -- indexer isolation: IDENTICAL q_resid and hidden_states are fed
               to both indexers.  Any top-k mismatch here is caused inside the
               indexer itself (its RoPE arithmetic), not upstream.
  Section 4 -- q_resid isolation: the two RMSNorm implementations evaluated on
               identical activations, compared bitwise in BF16.
  Section 5 -- RoPE isolation: pre-rotation q/k must be bitwise identical; the
               post-rotation q/k are compared against a shared FP32 reference
               to quantify the BF16 rounding noise each implementation adds.
  Section 6 -- layer byte trace: hooks every submodule of the first diverging
               layer and prints the first bitwise-divergent tensor, plus a
               manual comparison of the MAIN MLA RoPE (ComplexRoPE vs HF
               BF16 interleave).

Every section prints directly and the script exits 0: it is a diagnostic, not
an assertion, so it can be re-run after a fix to confirm convergence.
"""

from __future__ import annotations

import torch

from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    apply_rotary_pos_emb_interleave,
)

from tests.unit_tests.test_glm5_parity import (
    _build_models,
    _causal_mask,
    _convert_models_to_bfloat16,
    _selection_mismatch_positions,
)


def _count_unequal(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int]:
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    return int((a != b).sum().item()), a.numel()


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _report_tensor_pair(name: str, a: torch.Tensor, b: torch.Tensor) -> bool:
    """Print bitwise-unequal count and max abs diff; return whether equal."""
    unequal, total = _count_unequal(a, b)
    max_diff = _max_abs_diff(a, b)
    print(
        f"    {name:<26} unequal={unequal}/{total} "
        f"({100.0 * unequal / total:.2f}%) max_abs_diff={max_diff:.3e}"
    )
    return unequal == 0


def _interleave_to_adjacent(x: torch.Tensor) -> torch.Tensor:
    """Convert HF interleave-layout rotation output to the adjacent-pair layout.

    HF concatenates [rot(0), rot(2), ..., rot(D-2), rot(1), rot(3), ...,
    rot(D-1)]; the adjacent-pair layout is [rot(0), rot(1), ..., rot(D-1)].
    Both layouts hold the same mathematical rotation of the same coordinates.
    """
    dim = x.shape[-1]
    even = x[..., : dim // 2]
    odd = x[..., dim // 2 :]
    return torch.stack([even, odd], dim=-1).reshape(x.shape)


def _hf_cos_sin_f32(
    hf_model: torch.nn.Module,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replicate GlmMoeDsaRotaryEmbedding.forward in FP32 (no BF16 cast)."""
    config = hf_model.config
    base = config.rope_parameters["rope_theta"]
    dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    inv_freq = inv_freq[None, :, None].to(device=hidden_states.device)
    freqs = (inv_freq @ positions[:, None, :].float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _reference_rotation_f32(
    q_rot: torch.Tensor, cos_f32: torch.Tensor, sin_f32: torch.Tensor
) -> torch.Tensor:
    """Interleaved RoPE evaluated in FP32, output in adjacent-pair layout.

    cos/sin have the full emb width; the per-pair angle is the first half.
    """
    dim = q_rot.shape[-1]
    cos = cos_f32[..., : dim // 2].unsqueeze(1)
    sin = sin_f32[..., : dim // 2].unsqueeze(1)
    pairs = q_rot.float().reshape(*q_rot.shape[:-1], dim // 2, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    real = even * cos - odd * sin
    imag = even * sin + odd * cos
    return torch.stack([real, imag], dim=-1).reshape(q_rot.shape).to(q_rot.dtype)


def _main_rope_check(
    hf_layer: torch.nn.Module,
    titan_layer: torch.nn.Module,
    captured: dict,
    hf_pos: tuple[torch.Tensor, torch.Tensor],
    positions: torch.Tensor,
    cos_f32: torch.Tensor,
    sin_f32: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> None:
    """Compare the MAIN MLA RoPE (ComplexRoPE vs HF BF16 interleave)."""
    hf_attn = hf_layer.self_attn
    titan_attn = titan_layer.attention
    n_heads = titan_attn.n_heads
    qk_nope = titan_attn.qk_nope_head_dim
    qk_rope = titan_attn.qk_rope_head_dim
    qk_head = titan_attn.qk_head_dim
    kv_lora = titan_attn.kv_lora_rank

    q_b_out = captured[("hf", "q_b_proj")]  # [B, L, H*D]
    q_states = q_b_out.view(batch_size, seq_len, n_heads, qk_head).transpose(1, 2)
    _q_pass, q_rot_hf = torch.split(q_states, [qk_nope, qk_rope], dim=-1)

    kv_a_out = captured[("hf", "kv_a_proj")]  # [B, L, kv_lora + R]
    _kv_pass, k_rot_flat = torch.split(kv_a_out, [kv_lora, qk_rope], dim=-1)
    k_rot_hf = k_rot_flat.view(batch_size, 1, seq_len, qk_rope)

    cos, sin = hf_pos
    hf_q_rot, hf_k_rot = apply_rotary_pos_emb_interleave(
        q_rot_hf, k_rot_hf, cos, sin, unsqueeze_dim=1
    )

    titan_q_rot, titan_k_rot = titan_attn.rope(
        q_rot_hf.transpose(1, 2), k_rot_hf.transpose(1, 2), positions
    )
    hf_q_adj = _interleave_to_adjacent(hf_q_rot).transpose(1, 2)
    hf_k_adj = _interleave_to_adjacent(hf_k_rot).transpose(1, 2)

    print("    -- main q_rot/k_rot post-rotation vs shared FP32 reference --")
    ref_q = _reference_rotation_f32(q_rot_hf.transpose(1, 2), cos_f32, sin_f32)
    ref_k = _reference_rotation_f32(k_rot_hf.transpose(1, 2), cos_f32, sin_f32)
    _report_tensor_pair("titan main q_rot vs ref", titan_q_rot, ref_q)
    _report_tensor_pair("hf    main q_rot vs ref", hf_q_adj, ref_q)
    _report_tensor_pair("titan main k_rot vs ref", titan_k_rot, ref_k)
    _report_tensor_pair("hf    main k_rot vs ref", hf_k_adj, ref_k)
    print("    -- main q_rot/k_rot titan vs hf (the exact score-input gap) --")
    _report_tensor_pair("main q_rot", titan_q_rot, hf_q_adj)
    _report_tensor_pair("main k_rot", titan_k_rot, hf_k_adj)


def _byte_trace_layer0(
    hf_model: torch.nn.Module,
    titan_model: torch.nn.Module,
    hf_inputs: dict[int, torch.Tensor],
    causal_mask: torch.Tensor,
    positions: torch.Tensor,
    batch_size: int,
    seq_len: int,
    layer_index: int = 0,
) -> None:
    """Bisect a layer by comparing every submodule output bitwise."""
    print()
    print("=" * 78)
    print(f"Section 6 -- layer {layer_index} byte trace (first divergent tensor)")
    print("=" * 78)
    hf_layer = hf_model.model.layers[layer_index]
    titan_layer = titan_model.layers[str(layer_index)]
    common_hidden = hf_inputs[layer_index]
    hf_pos = hf_model.model.rotary_emb(common_hidden, position_ids=positions)
    cos_f32, sin_f32 = _hf_cos_sin_f32(hf_model, common_hidden, positions)

    pairs = [
        ("attention_norm", hf_layer.input_layernorm, titan_layer.attention_norm),
        ("q_a_proj", hf_layer.self_attn.q_a_proj, titan_layer.attention.wq_a),
        (
            "q_a_layernorm",
            hf_layer.self_attn.q_a_layernorm,
            titan_layer.attention.q_norm,
        ),
        ("q_b_proj", hf_layer.self_attn.q_b_proj, titan_layer.attention.wq_b),
        (
            "kv_a_proj",
            hf_layer.self_attn.kv_a_proj_with_mqa,
            titan_layer.attention.wkv_a,
        ),
        (
            "kv_a_layernorm",
            hf_layer.self_attn.kv_a_layernorm,
            titan_layer.attention.kv_norm,
        ),
        ("kv_b_proj", hf_layer.self_attn.kv_b_proj, titan_layer.attention.wkv_b),
        ("o_proj", hf_layer.self_attn.o_proj, titan_layer.attention.wo),
        ("indexer", hf_layer.self_attn.indexer, titan_layer.attention.indexer),
        ("attention_out", hf_layer.self_attn, titan_layer.attention),
        (
            "post_attn_norm",
            hf_layer.post_attention_layernorm,
            titan_layer.ffn_norm,
        ),
        ("gate_proj", hf_layer.mlp.gate_proj, titan_layer.feed_forward.w1),
        ("up_proj", hf_layer.mlp.up_proj, titan_layer.feed_forward.w3),
        ("down_proj", hf_layer.mlp.down_proj, titan_layer.feed_forward.w2),
        ("layer_out", hf_layer, titan_layer),
    ]
    captured: dict[tuple[str, str], torch.Tensor] = {}
    handles = []
    for label, hf_module, titan_module in pairs:

        def capture_hf(_m, _i, output, label=label):
            captured[("hf", label)] = (
                output[0] if isinstance(output, tuple) else output
            ).detach()

        def capture_titan(_m, _i, output, label=label):
            captured[("titan", label)] = output.detach()

        handles.append(hf_module.register_forward_hook(capture_hf))
        handles.append(titan_module.register_forward_hook(capture_titan))

    try:
        with torch.no_grad():
            hf_layer(
                common_hidden,
                attention_mask=causal_mask,
                position_ids=positions,
                position_embeddings=hf_pos,
                use_cache=False,
            )
            titan_layer(common_hidden, causal_mask, positions)
    finally:
        for handle in handles:
            handle.remove()

    print(f"  {'component':<16}{'unequal/total':<20}{'max_abs_diff':<15}verdict")
    first_divergent = None
    for label, _hf_module, _titan_module in pairs:
        hf_value = captured.get(("hf", label))
        titan_value = captured.get(("titan", label))
        if hf_value is None or titan_value is None:
            print(
                f"  {label:<16}{'rec_missing':<20}"
                f"hf={hf_value is not None} tit={titan_value is not None}"
            )
            continue
        if hf_value.shape != titan_value.shape:
            print(
                f"  {label:<16} SHAPE MISMATCH "
                f"{tuple(hf_value.shape)} vs {tuple(titan_value.shape)}"
            )
            continue
        unequal, total = _count_unequal(hf_value, titan_value)
        max_diff = _max_abs_diff(hf_value, titan_value)
        if unequal > 0 and first_divergent is None:
            first_divergent = label
            verdict = "FIRST DIVERGENT"
        else:
            verdict = "same" if unequal == 0 else "diff"
        print(f"  {label:<16}{f'{unequal}/{total}':<20}{max_diff:<15.3e}{verdict}")

    _main_rope_check(
        hf_layer,
        titan_layer,
        captured,
        hf_pos,
        positions,
        cos_f32,
        sin_f32,
        batch_size,
        seq_len,
    )
    if first_divergent is not None:
        print(f"  => first bitwise-divergent layer-0 component: {first_divergent}")
    else:
        print("  => no bitwise divergence found in layer-0 module outputs")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("GLM-5 parity diagnostics require one CUDA device")

    device = torch.device("cuda")
    seed = 53
    batch_size, seq_len = 1, 16

    print(f"Building GLM-5 peers (seed={seed}) and converting to BF16 ...")
    hf_model, titan_model = _build_models(device, seed=seed)
    _convert_models_to_bfloat16(hf_model, titan_model, device)

    torch.manual_seed(seed)
    tokens = torch.randint(
        0, 2048, (batch_size, seq_len), device=device, dtype=torch.long
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.long).expand(
        batch_size, -1
    )
    causal_mask = _causal_mask(positions, torch.bfloat16)
    layer_indices = range(len(hf_model.model.layers))

    # ------------------------------------------------------------------ #
    # Section 2: full BF16 forward, per-layer divergence table.           #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 78)
    print("Section 2 -- full BF16 forward, per-layer divergence")
    print("=" * 78)

    records = {
        "hf_blocks": {li: None for li in layer_indices},
        "titan_blocks": {li: None for li in layer_indices},
        "hf_indexer": {li: None for li in layer_indices},
        "titan_indexer": {li: None for li in layer_indices},
        "hf_router": {},
        "titan_router": {},
    }
    hf_inputs: dict[int, torch.Tensor] = {}
    handles = []
    for li in layer_indices:
        hf_layer = hf_model.model.layers[li]
        titan_layer = titan_model.layers[str(li)]

        def capture_hf_input(_m, i, li=li):
            hf_inputs[li] = i[0].detach()

        def capture_hf_block(_m, _i, o, li=li):
            records["hf_blocks"][li] = o[0].detach()

        def capture_titan_block(_m, _i, o, li=li):
            records["titan_blocks"][li] = o.detach()

        def capture_hf_indexer(_m, _i, o, li=li):
            records["hf_indexer"][li] = o.detach()

        def capture_titan_indexer(_m, _i, o, li=li):
            records["titan_indexer"][li] = o.detach()

        handles.append(hf_layer.register_forward_pre_hook(capture_hf_input))
        handles.append(hf_layer.register_forward_hook(capture_hf_block))
        handles.append(titan_layer.register_forward_hook(capture_titan_block))
        handles.append(
            hf_layer.self_attn.indexer.register_forward_hook(capture_hf_indexer)
        )
        handles.append(
            titan_layer.attention.indexer.register_forward_hook(capture_titan_indexer)
        )
        if getattr(titan_layer, "moe_enabled", False):
            records["hf_router"][li] = None
            records["titan_router"][li] = None

            def capture_hf_router(_m, i, o, li=li):
                records["hf_router"][li] = (
                    o[2].view(i[0].shape[0], i[0].shape[1], -1).detach()
                )

            def capture_titan_router(_m, _i, o, li=li):
                records["titan_router"][li] = o[1].detach()

            handles.append(hf_layer.mlp.gate.register_forward_hook(capture_hf_router))
            handles.append(
                titan_layer.moe.router.register_forward_hook(capture_titan_router)
            )

    try:
        with torch.no_grad():
            hf_outputs = hf_model(
                input_ids=tokens, position_ids=positions, use_cache=False
            )
            titan_logits = titan_model(tokens, positions=positions)
    finally:
        for handle in handles:
            handle.remove()

    print("  Layer  Type     block_maxdiff   indexer_pos         router_idx")
    first_full_indexer_mismatch = None
    for li in layer_indices:
        hf_b = records["hf_blocks"][li]
        tit_b = records["titan_blocks"][li]
        if hf_b is not None and tit_b is not None and hf_b.shape == tit_b.shape:
            block_col = f"{_max_abs_diff(hf_b, tit_b):.3e}"
        else:
            block_col = "n/a"
        hf_idx = records["hf_indexer"][li]
        tit_idx = records["titan_indexer"][li]
        if hf_idx is not None and tit_idx is not None:
            mismatch = _selection_mismatch_positions(tit_idx, hf_idx, positions)
            if mismatch and first_full_indexer_mismatch is None:
                first_full_indexer_mismatch = li
            indexer_col = (
                f"{len(mismatch)} [{','.join(str(p) for p in sorted(set(mismatch)))}]"
            )
        else:
            indexer_col = "n/a"
        hf_rt = records["hf_router"].get(li)
        tit_rt = records["titan_router"].get(li)
        if hf_rt is None and tit_rt is None:
            router_col = "dense"
        elif hf_rt is not None and tit_rt is not None:
            router_col = f"{int((hf_rt != tit_rt).sum().item())}"
        else:
            router_col = "n/a"
        layer_type = "MoE" if li in records["hf_router"] else "dense"
        print(f"  {li:<6}{layer_type:<9}{block_col:<16}{indexer_col:<18}{router_col}")
    if first_full_indexer_mismatch is not None:
        print(f"  first_full_indexer_mismatch_layer={first_full_indexer_mismatch}")
    print(
        f"  full_logits max_abs_diff="
        f"{_max_abs_diff(hf_outputs.logits, titan_logits):.3e}"
    )

    # ------------------------------------------------------------------ #
    # Sections 3-5: component isolation on common inputs.                 #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 78)
    print("Sections 3-5 -- component isolation on common inputs")
    print("=" * 78)

    common_mismatch_layers: list[int] = []
    q_resid_diff_layers: list[int] = []
    rot_gap_layers: list[int] = []

    with torch.no_grad():
        for li in layer_indices:
            print(f"\n  --- layer {li} ---")
            hf_layer = hf_model.model.layers[li]
            titan_layer = titan_model.layers[str(li)]
            hf_attn = hf_layer.self_attn
            titan_attn = titan_layer.attention

            common_hidden = hf_inputs[li]
            hf_pos = hf_model.model.rotary_emb(common_hidden, position_ids=positions)
            cos_f32, sin_f32 = _hf_cos_sin_f32(hf_model, common_hidden, positions)

            # --- Section 4: q_resid source ---
            print("  [Section 4] q_resid (linear + RMSNorm on identical input)")
            hf_wq_a_out = hf_attn.q_a_proj(common_hidden)
            titan_wq_a_out = titan_attn.wq_a(common_hidden)
            _report_tensor_pair("pre-norm (wq_a)", titan_wq_a_out, hf_wq_a_out)
            common_q_resid = hf_attn.q_a_layernorm(hf_wq_a_out)
            titan_q_resid = titan_attn.q_norm(titan_wq_a_out)
            q_resid_equal = _report_tensor_pair(
                "q_resid (after norm)", titan_q_resid, common_q_resid
            )
            if not q_resid_equal:
                q_resid_diff_layers.append(li)

            # --- Section 3: indexer with identical inputs ---
            print("  [Section 3] indexer top-k on identical inputs")
            hf_topk = hf_attn.indexer(
                common_hidden,
                common_q_resid,
                hf_pos,
                causal_mask[:, 0],
                positions,
            )
            titan_topk = titan_attn.indexer(
                common_hidden, common_q_resid, positions, causal_mask[:, 0]
            )
            mism = _selection_mismatch_positions(titan_topk, hf_topk, positions)
            print(f"    indexer_topk_mismatch_positions: {sorted(set(mism))}")
            if mism:
                common_mismatch_layers.append(li)

            # --- Section 5: RoPE arithmetic ---
            print("  [Section 5] indexer RoPE arithmetic")
            head_dim = titan_attn.indexer.head_dim
            rope_dim = titan_attn.indexer.qk_rope_head_dim
            hf_q_pre = hf_attn.indexer.wq_b(common_q_resid).view(
                batch_size, seq_len, hf_attn.indexer.n_heads, head_dim
            )
            titan_q_pre = titan_attn.indexer.wq_b(common_q_resid).view(
                batch_size, seq_len, titan_attn.indexer.n_heads, head_dim
            )
            pre_q_equal = _report_tensor_pair(
                "pre-rotation q (wq_b)", titan_q_pre, hf_q_pre
            )
            hf_k_pre = hf_attn.indexer.k_norm(hf_attn.indexer.wk(common_hidden))
            titan_k_pre = titan_attn.indexer.k_norm(
                titan_attn.indexer.wk(common_hidden)
            )
            pre_k_equal = _report_tensor_pair(
                "pre-rotation k (wk+k_norm)", titan_k_pre, hf_k_pre
            )

            q_rot = titan_q_pre[..., :rope_dim]
            k_rot = titan_k_pre[..., :rope_dim].unsqueeze(2)
            titan_q_rot, titan_k_rot = titan_attn.indexer.rope(q_rot, k_rot, positions)
            hf_q_rot_hf, hf_k_rot_hf = apply_rotary_pos_emb_interleave(
                q_rot, k_rot, hf_pos[0], hf_pos[1], unsqueeze_dim=2
            )
            hf_q_rot_adj = _interleave_to_adjacent(hf_q_rot_hf)
            hf_k_rot_adj = _interleave_to_adjacent(hf_k_rot_hf)

            ref_q = _reference_rotation_f32(q_rot, cos_f32, sin_f32)
            ref_k = _reference_rotation_f32(k_rot, cos_f32, sin_f32)
            print("    -- post-rotation vs shared FP32 reference --")
            _report_tensor_pair("titan q_rot vs ref", titan_q_rot, ref_q)
            _report_tensor_pair("hf    q_rot vs ref", hf_q_rot_adj, ref_q)
            _report_tensor_pair("titan k_rot vs ref", titan_k_rot, ref_k)
            _report_tensor_pair("hf    k_rot vs ref", hf_k_rot_adj, ref_k)
            print("    -- post-rotation titan vs hf (the exact score-input gap) --")
            q_rot_equal = _report_tensor_pair("q_rot", titan_q_rot, hf_q_rot_adj)
            k_rot_equal = _report_tensor_pair("k_rot", titan_k_rot, hf_k_rot_adj)
            if not (q_rot_equal and k_rot_equal):
                rot_gap_layers.append(li)

            if pre_q_equal and pre_k_equal and q_rot_equal and k_rot_equal:
                print("    => indexer internally bitwise-identical on common inputs")

    # ------------------------------------------------------------------ #
    # Section 6: layer-0 byte trace.                                     #
    # ------------------------------------------------------------------ #
    _byte_trace_layer0(
        hf_model,
        titan_model,
        hf_inputs,
        causal_mask,
        positions,
        batch_size,
        seq_len,
    )

    # ------------------------------------------------------------------ #
    # Section 7: verdict.                                                 #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 78)
    print("Section 7 -- verdict")
    print("=" * 78)
    if common_mismatch_layers:
        print("  indexer top-k differs even with IDENTICAL q_resid and hidden_states.")
        print(f"  layers: {common_mismatch_layers}")
        if rot_gap_layers:
            print(
                "  post-rotation q/k differ bitwise (ComplexRoPE FP32 round-once vs "
                "HF BF16 interleave); this rounding gap flips the FP32 index_scores "
                "at the top-k boundary -> RoPE arithmetic is the divergence source."
            )
        print(
            "  Fix direction: align the indexer RoPE path with HF's BF16 interleave "
            "arithmetic, or document the accepted gap."
        )
    elif q_resid_diff_layers:
        print("  indexer top-k is stable under identical q_resid, but q_resid itself")
        print(f"  differs (layers {q_resid_diff_layers}) -> the RMSNorm kernel")
        print(
            "  (HF explicit FP32 upcast vs torch fused kernel) is the upstream trigger."
        )
        print(
            "  Fix direction: give GLM-5 a GlmMoeDsaRMSNorm-equivalent norm "
            "(explicit FP32 upcast, single trailing cast)."
        )
    else:
        print("  No indexer or q_resid divergence under identical inputs; the")
        print("  full-logits gap must come from downstream MoE/attention arithmetic.")
    print(f"  full-forward indexer mismatch layers: {first_full_indexer_mismatch}")


if __name__ == "__main__":
    main()
