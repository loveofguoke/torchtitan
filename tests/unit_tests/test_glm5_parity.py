# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# 测试torchtitan glm和hf glm的精度一致性

"""Optional CUDA component-parity checks for the GLM-5 debug model.

These checks intentionally exercise the real Transformers implementation and
the state-dict adapter.  They remain CUDA-gated because Task 8 owns the
single-GPU end-to-end BF16 and gradient acceptance work.
"""

import unittest

import torch
import torch.nn.functional as F

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import TokenChoiceTopKRouter
from torchtitan.models.glm5 import glm5_configs, Glm5StateDictAdapter

_TRANSFORMERS_IMPORT_ERROR = None

try:
    from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM
except Exception as _TRANSFORMERS_IMPORT_ERROR:  # pragma: no cover - environment dependent
    GlmMoeDsaConfig = None
    GlmMoeDsaForCausalLM = None
else:
    _TRANSFORMERS_IMPORT_ERROR = None


def _hf_config():
    assert GlmMoeDsaConfig is not None
    return GlmMoeDsaConfig(
        vocab_size=2048,
        hidden_size=256,
        intermediate_size=1024,
        moe_intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        n_shared_experts=1,
        n_routed_experts=8,
        routed_scaling_factor=2.5,
        kv_lora_rank=64,
        q_lora_rank=128,
        qk_rope_head_dim=32,
        qk_nope_head_dim=32,
        v_head_dim=64,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        attention_dropout=0.0,
        index_topk=8,
        index_head_dim=64,
        index_n_heads=4,
        first_k_dense_replace=1,
        indexer_types=["full", "full", "full", "full"],
        rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0},
        use_cache=False,
        _attn_implementation="eager",
    )


# 构造 HF 模型和 TorchTitan 模型，并加载相同权重
def _build_models(
    device: torch.device, *, seed: int = 41
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Construct exact FP32 peers and load the HF tensors through the adapter."""
    assert GlmMoeDsaForCausalLM is not None
    torch.manual_seed(seed)
    # 配置化构造模型实例
    hf_model = GlmMoeDsaForCausalLM(_hf_config()).float()

    titan_config = glm5_configs["debugmodel"]()
    titan_model = titan_config.build()
    titan_model.init_states()  # Decoder的方法
    adapter = Glm5StateDictAdapter(titan_config, hf_assets_path=None)
    # HF随机初始化，提取HF state dict，通过adapter加载到TorchTitan模型
    # 保证两者权重完全一致
    titan_model.load_state_dict(adapter.from_hf(hf_model.state_dict()), strict=True)

    hf_model = hf_model.to(device).eval()
    titan_model = titan_model.to(device).eval()
    return hf_model, titan_model


def _convert_models_to_bfloat16(
    hf_model: torch.nn.Module,
    titan_model: torch.nn.Module,
    device: torch.device,
) -> None:
    """Move both peers to CUDA BF16 while retaining FP32 DSA head weights.

    The GLM DSA indexer's learned head weights are deliberately evaluated in
    FP32.  TorchTitan's indexer protects that parameter in ``_apply``; the
    Transformers reference needs the equivalent explicit restoration after
    ``Module.to(dtype=torch.bfloat16)``.
    """
    hf_model.to(device=device, dtype=torch.bfloat16)
    titan_model.to(device=device, dtype=torch.bfloat16)

    for module in hf_model.modules():
        indexer = getattr(module, "indexer", None)
        if indexer is not None:
            indexer.weights_proj.float()

    for model in (hf_model, titan_model):
        indexer_weights = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if name.endswith("indexer.weights_proj.weight")
        ]
        expected_indexer_weights = len(getattr(model, "model", model).layers)
        if len(indexer_weights) != expected_indexer_weights:
            raise AssertionError(
                "GLM-5 model has an unexpected number of "
                f"indexer.weights_proj.weight parameters: {indexer_weights}"
            )
        for name, parameter in indexer_weights:
            if parameter.dtype is not torch.float32:
                raise AssertionError(
                    "GLM-5 indexer.weights_proj.weight must remain FP32 after "
                    f"BF16 conversion: {name} has dtype {parameter.dtype}"
                )


def _causal_mask(positions_BL: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    B, L = positions_BL.shape
    token_indices_L = torch.arange(L, device=positions_BL.device)
    allowed_BLL = token_indices_L[None, :, None] >= token_indices_L[None, None, :]
    return torch.zeros(B, 1, L, L, dtype=dtype, device=positions_BL.device).masked_fill(
        ~allowed_BLL.unsqueeze(1), torch.finfo(dtype).min
    )


def _selection_mismatch_positions(
    actual_BLK: torch.Tensor,
    expected_BLK: torch.Tensor,
    positions_BL: torch.Tensor | None = None,
) -> list[int]:
    """Return query positions whose selected sets differ."""
    if actual_BLK.shape != expected_BLK.shape:
        raise ValueError(
            "selection shapes differ: "
            f"actual={tuple(actual_BLK.shape)}, expected={tuple(expected_BLK.shape)}"
        )
    if actual_BLK.ndim != 3:
        raise ValueError(
            f"selections must have shape [B, L, K], got {actual_BLK.shape}"
        )
    if positions_BL is not None and positions_BL.shape != actual_BLK.shape[:2]:
        raise ValueError(
            "positions must match the selection batch and sequence dimensions: "
            f"positions={tuple(positions_BL.shape)}, "
            f"selections={tuple(actual_BLK.shape)}"
        )

    actual_cpu = actual_BLK.detach().cpu()
    expected_cpu = expected_BLK.detach().cpu()
    positions_cpu = None if positions_BL is None else positions_BL.detach().cpu()
    mismatched_positions: list[int] = []
    for batch_index in range(actual_cpu.shape[0]):
        for sequence_index in range(actual_cpu.shape[1]):
            causal_position = (
                None
                if positions_cpu is None
                else int(positions_cpu[batch_index, sequence_index])
            )
            actual = {
                int(index)
                for index in actual_cpu[batch_index, sequence_index]
                if causal_position is None or int(index) <= causal_position
            }
            expected = {
                int(index)
                for index in expected_cpu[batch_index, sequence_index]
                if causal_position is None or int(index) <= causal_position
            }
            if actual != expected:
                mismatched_positions.append(sequence_index)
    return mismatched_positions


def _format_parity_diagnostics(
    records: dict[str, dict[int, torch.Tensor | None]],
    positions_BL: torch.Tensor,
) -> str:
    """Format GLM-5 parity diagnostics as a compact table.

    Output is a per-layer table followed by a summary section.
    """
    expected_record_names = (
        "hf_blocks",
        "titan_blocks",
        "hf_indexer",
        "titan_indexer",
        "hf_router",
        "titan_router",
    )
    missing_record_names = [
        name for name in expected_record_names if name not in records
    ]
    if missing_record_names:
        return f"missing record groups: {missing_record_names}"

    layer_indices = sorted(
        {
            layer_index
            for record_name in expected_record_names
            for layer_index in records[record_name]
        }
    )
    moe_layer_indices: set[int] = set()
    for record_name in ("hf_router", "titan_router"):
        moe_layer_indices.update(records.get(record_name, {}).keys())

    lines: list[str] = []
    lines.append("=" * 67)
    lines.append(" GLM-5 Parity Diagnostics")
    lines.append("=" * 67)
    lines.append("")

    discrete_mismatches: list[tuple[int, int, str]] = []

    # --- Table header ---
    lines.append("  Layer  Type     Block        Indexer              Router")
    lines.append(" " + "-" * 62)

    for layer_index in layer_indices:
        is_moe = layer_index in moe_layer_indices
        layer_type = "MoE" if is_moe else "dense"
        layer_label = f"{layer_index:<6}"

        hf_block = records["hf_blocks"].get(layer_index)
        titan_block = records["titan_blocks"].get(layer_index)

        # --- Block column ---
        if hf_block is None or titan_block is None:
            block_col = "rec_missing"
        elif (
            hf_block.ndim != 3
            or titan_block.ndim != 3
            or hf_block.shape[:2] != positions_BL.shape
            or titan_block.shape[:2] != positions_BL.shape
        ):
            block_col = "shape_err"
        elif hf_block.shape != titan_block.shape:
            block_col = f"shape_mismatch"
        else:
            block_difference = (hf_block.float() - titan_block.float()).abs()
            position_max_L = block_difference.amax(dim=(0, 2)).detach().cpu()
            max_diff = float(position_max_L.max())
            block_col = f"{max_diff:.6g}"

        block_col = f"{block_col:<13}"

        # --- Indexer column ---
        indexer_col = ""
        hf_indexer_records = records["hf_indexer"]
        titan_indexer_records = records["titan_indexer"]
        if layer_index in hf_indexer_records or layer_index in titan_indexer_records:
            hf_selection = hf_indexer_records.get(layer_index)
            titan_selection = titan_indexer_records.get(layer_index)
            if hf_selection is None or titan_selection is None:
                hf_ok = hf_selection is not None
                tit_ok = titan_selection is not None
                indexer_col = f"rec_missing hf={hf_ok} tit={tit_ok}"
            else:
                try:
                    mismatch_positions = _selection_mismatch_positions(
                        titan_selection,
                        hf_selection,
                        positions_BL,
                    )
                except ValueError as error:
                    indexer_col = f"err={error}"
                else:
                    count = len(mismatch_positions)
                    if count == 0:
                        indexer_col = "0"
                    else:
                        pos_str = ",".join(str(p) for p in sorted(set(mismatch_positions)))
                        indexer_col = f"{count} [{pos_str}]"
                    discrete_mismatches.extend(
                        (layer_index, position, "indexer") for position in mismatch_positions
                    )
        else:
            indexer_col = "-"

        indexer_col = f"{indexer_col:<20}"

        # --- Router column ---
        router_col = ""
        hf_router_records = records["hf_router"]
        titan_router_records = records["titan_router"]
        if layer_index in hf_router_records or layer_index in titan_router_records:
            hf_selection = hf_router_records.get(layer_index)
            titan_selection = titan_router_records.get(layer_index)
            if hf_selection is None or titan_selection is None:
                hf_ok = hf_selection is not None
                tit_ok = titan_selection is not None
                router_col = f"rec_missing hf={hf_ok} tit={tit_ok}"
            else:
                try:
                    mismatch_positions = _selection_mismatch_positions(
                        titan_selection,
                        hf_selection,
                        None,
                    )
                except ValueError as error:
                    router_col = f"err={error}"
                else:
                    count = len(mismatch_positions)
                    if count == 0:
                        router_col = "0"
                    else:
                        pos_str = ",".join(str(p) for p in sorted(set(mismatch_positions)))
                        router_col = f"{count} [{pos_str}]"
                    discrete_mismatches.extend(
                        (layer_index, position, "router") for position in mismatch_positions
                    )
        else:
            router_col = "N/A" if not is_moe else "-"

        router_col = f"{router_col:<20}"

        lines.append(f"  {layer_label}{layer_type:<8}{block_col}{indexer_col}{router_col}")

    lines.append(" " + "-" * 62)

    # --- Summary ---
    lines.append("  Summary")
    if discrete_mismatches:
        layer_index, position, source = min(discrete_mismatches)
        lines.append(
            f"    first_discrete_mismatch       : layer {layer_index}, position {position}, source={source}"
        )
        lines.append(f"    total_discrete_mismatches    : {len(discrete_mismatches)}")
        layers_with_block_errors = []
        for li in layer_indices:
            hf_b = records["hf_blocks"].get(li)
            tit_b = records["titan_blocks"].get(li)
            if hf_b is not None and tit_b is not None and hf_b.ndim == 3 and tit_b.ndim == 3 and hf_b.shape == tit_b.shape:
                diff = (hf_b.float() - tit_b.float()).abs().amax(dim=(0, 2)).max().item()
                if diff > 1e-6:
                    layers_with_block_errors.append(li)
        if layers_with_block_errors:
            lines.append(f"    layers_with_block_errors     : {layers_with_block_errors}")
        layers_with_indexer_mismatches = sorted(
            {li for li, _, src in discrete_mismatches if src == "indexer"}
        )
        if layers_with_indexer_mismatches:
            lines.append(f"    layers_with_indexer_mismatches: {layers_with_indexer_mismatches}")
        layers_with_router_mismatches = sorted(
            {li for li, _, src in discrete_mismatches if src == "router"}
        )
        if layers_with_router_mismatches:
            lines.append(f"    layers_with_router_mismatches : {layers_with_router_mismatches}")
    else:
        lines.append("    first_discrete_mismatch       : none")

    return "\n".join(lines)



class TestGlm5ParityDiagnostics(unittest.TestCase):
    def test_format_reports_layer_position_and_discrete_mismatches(self) -> None:
        zero_blocks = torch.zeros(1, 3, 2)
        divergent_blocks = zero_blocks.clone()
        divergent_blocks[0, 1, 0] = 0.5
        matching_indices = torch.tensor([[[0, 1], [0, 1], [0, 1]]])
        future_only_indices = torch.tensor([[[0, 2], [0, 1], [0, 1]]])
        indexer_indices = torch.tensor([[[0, 2], [0, 1], [0, 2]]])
        router_indices = torch.tensor([[[1, 0], [0, 2], [1, 0]]])
        records = {
            "hf_blocks": {0: zero_blocks, 1: zero_blocks},
            "titan_blocks": {0: zero_blocks, 1: divergent_blocks},
            "hf_indexer": {0: matching_indices, 1: matching_indices},
            "titan_indexer": {0: future_only_indices, 1: indexer_indices},
            "hf_router": {1: matching_indices},
            "titan_router": {1: router_indices},
        }

        diagnostics = _format_parity_diagnostics(records, torch.tensor([[0, 1, 2]]))

        self.assertIn("0.5", diagnostics)
        self.assertIn("1 [2]", diagnostics)
        self.assertIn("1 [1]", diagnostics)
        self.assertIn(
            "first_discrete_mismatch       : layer 1, position 1, source=router",
            diagnostics,
        )

    def test_format_reports_missing_expected_records(self) -> None:
        matching_indices = torch.tensor([[[0], [0]]])
        zero_blocks = torch.zeros(1, 2, 2)
        records = {
            "hf_blocks": {0: zero_blocks, 1: zero_blocks},
            "titan_blocks": {0: zero_blocks, 1: zero_blocks},
            "hf_indexer": {0: None, 1: matching_indices},
            "titan_indexer": {0: None, 1: matching_indices},
            "hf_router": {1: matching_indices},
            "titan_router": {1: None},
        }

        diagnostics = _format_parity_diagnostics(records, torch.tensor([[0, 1]]))

        self.assertIn("rec_missing hf=False tit=False", diagnostics)
        self.assertIn("rec_missing hf=True tit=False", diagnostics)

    def test_format_reports_incompatible_block_shape(self) -> None:
        malformed_blocks = torch.zeros(2, 2)
        records = {
            "hf_blocks": {0: malformed_blocks},
            "titan_blocks": {0: malformed_blocks},
            "hf_indexer": {},
            "titan_indexer": {},
            "hf_router": {},
            "titan_router": {},
        }

        diagnostics = _format_parity_diagnostics(records, torch.tensor([[0, 1]]))

        self.assertIn(
            "shape_err",
            diagnostics,
        )


class TestGlm5Bfloat16RouterPrecision(unittest.TestCase):
    # 测试Router的gate在BF16下仍然保持FP32精度, 并且路由结果与FP32计算一致
    def test_router_gate_is_evaluated_in_float32(self) -> None:
        """BF16 activations must not reduce the discrete router's precision."""
        titan_model = glm5_configs["debugmodel"]().build()
        titan_model.init_states()
        titan_model.bfloat16()
        router = titan_model.layers["1"].moe.router
        hidden_states = torch.randn(1, 16, 256, dtype=torch.bfloat16)

        _, _, scores = router(hidden_states, titan_model.layers["1"].moe.expert_bias_E)
        expected_scores = torch.sigmoid(
            F.linear(hidden_states.float(), router.gate.weight.float())
        )
        # 模型参数和输入hidden states BF16下gate参数及route scores保持FP32精度
        # 不会降低离散选择的精度
        self.assertEqual(scores.dtype, torch.float32)
        # 且计算结果与FP32计算完全一致, bitwise equality
        torch.testing.assert_close(scores, expected_scores, rtol=0, atol=0)

    def test_router_uses_gate_forward_for_fp32_biased_computation(self) -> None:
        """FP32 routing must retain the gate module's hooks and gradients."""
        # 测试correction bias是否参与route
        router = TokenChoiceTopKRouter.Config(
            num_experts=4,
            gate=Linear.Config(in_features=3, out_features=4, bias=True),
            top_k=2,
            score_func="sigmoid",
        ).build()
        router.bfloat16()
        hidden_states = torch.randn(1, 2, 3, dtype=torch.bfloat16)
        gate_call_count = 0

        # hook函数用于统计gate的前向调用次数, 确保gate forward hook能生效
        def count_gate_calls(*_args) -> None:
            nonlocal gate_call_count
            gate_call_count += 1

        # 测试gate forward hook能否生效
        hook = router.gate.register_forward_hook(count_gate_calls)
        try:
            _, _, scores = router(hidden_states)
        finally:
            hook.remove()

        # 测试correction bias是否参与route
        expected_scores = torch.sigmoid(
            F.linear(
                hidden_states.float(),
                router.gate.weight.float(),
                router.gate.bias.float(),
            )
        )
        self.assertEqual(gate_call_count, 1)
        self.assertEqual(scores.dtype, torch.float32)
        torch.testing.assert_close(scores, expected_scores, rtol=0, atol=0)

        scores.sum().backward()
        # 检测gate 梯度是否存在
        self.assertIsNotNone(router.gate.weight.grad)
        self.assertIsNotNone(router.gate.bias.grad)


# 测试torchtitan glm和hf glm在新增结构上的精度一致性
class TestGlm5TransformersComponentParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # 如果 Transformers GLM-MoE-DSA 不可用，则跳过测试
        if _TRANSFORMERS_IMPORT_ERROR is not None:
            raise unittest.SkipTest(
                "Transformers GLM-MoE-DSA is unavailable: "
                f"{_TRANSFORMERS_IMPORT_ERROR!r}"
            )
        # 如果没有可用的 CUDA 设备，则跳过测试
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "GLM-5 Transformers parity requires one CUDA device"
            )

        cls.device = torch.device("cuda")
        # 得到hf模型和torchtitan模型实例
        cls.hf_model, cls.titan_model = _build_models(cls.device)
        # 统一创建测试输入，直接使用随机生成的FP32激活
        cls.batch_size = 2
        cls.sequence_length = 16
        torch.manual_seed(41)
        cls.hidden_states = torch.randn(
            cls.batch_size,
            cls.sequence_length,
            256,  # hidden_size, 最好不要硬编码
            device=cls.device,
            dtype=torch.float32,
        )
        # 给seq创建自然位置编号[0, 1, 2, ..., sequence_length-1],
        cls.positions = torch.arange(
            cls.sequence_length, device=cls.device, dtype=torch.long
        ).expand(cls.batch_size, -1)
        # 创建causal mask
        cls.causal_mask = _causal_mask(cls.positions, cls.hidden_states.dtype)
        # 计算hf的旋转位置编码, (cos, sin)
        # 这两个张量会传给 HF indexer; HF indexer API 需要显式接收 position embeddings；TorchTitan indexer 则在内部根据 positions 计算 RoPE
        cls.hf_position_embeddings = cls.hf_model.model.rotary_emb(
            cls.hidden_states, position_ids=cls.positions
        )

    def _normalized_hidden_states(self, layer_index: int) -> torch.Tensor:
        return self.hf_model.model.layers[layer_index].input_layernorm(
            self.hidden_states
        )

    # 在相同的输入、相同的权重、相同的 mask 和相同的位置编码下，验证 HF 与 TorchTitan 的 DSA indexer 是否选择完全相同的 top-k 历史 token
    def test_indexer_topk_matches_transformers_exactly(self) -> None:
        layer_index = 0
        # 取出第 0 层的两个 attention 模块
        # 第 0 层的选择有两个原因：
        # 1. debugmodel 的第 0 层是 dense FFN，但 attention/indexer 仍然完整存在；
        # 2. 测试只关注 attention/indexer，不需要 MoE router，因此选择第 0 层可以隔离 MoE 影响。
        hf_attention = self.hf_model.model.layers[layer_index].self_attn
        titan_attention = self.titan_model.layers[str(layer_index)].attention
        # 得到attention的输入(经过 pre layernorm 的 hidden states)
        # NOTE: HF 和 TorchTitan 并没有分别调用各自的 layer input norm；两边共享同一个 HF layernorm 的输出作为 attention 输入
        # NOTE: 因为这是单元测试，关注indexer的行为，需要统一其他模块的行为
        hidden_states = self._normalized_hidden_states(layer_index)
        # 计算q_latent, [B, S, q_lora_rank], 用于计算dsa中的q
        # HF: 自定义 RMSNorm，显式转 FP32
        # TorchTitan: torch.nn.RMSNorm，可能选择不同 fused kernel
        # TODO: 分析RMSNorm的实现差异
        hf_q_resid = hf_attention.q_a_layernorm(hf_attention.q_a_proj(hidden_states))
        titan_q_resid = titan_attention.q_norm(titan_attention.wq_a(hidden_states))
        # pytorch的判断条件: abs(actual - expected) <= atol + rtol * abs(expected)
        # 要求绝对精度为0，意味着两边的计算结果必须完全一致，不能有任何差异, 也就是 bitwise equality
        # TODO: 这里220 / 4096 元素不相等, 最大绝对差 2.3841858e-7
        # TODO: 如果误差很小，可以放宽
        # torch.testing.assert_close(titan_q_resid, hf_q_resid, rtol=0, atol=0)
        torch.testing.assert_close(
            titan_q_resid,
            hf_q_resid,
            rtol=1e-6,
            atol=1e-7,
        )
        common_q_resid = hf_q_resid

        # TODO: 没有比较indexer的topk scores, 只比较了topk索引

        # 调用indexer，得到top-k的token索引
        hf_topk = hf_attention.indexer(
            hidden_states,
            common_q_resid,
            self.hf_position_embeddings,
            self.causal_mask[:, 0],
            self.positions,
        )
        titan_topk = titan_attention.indexer(
            hidden_states,
            common_q_resid,
            self.positions,
            self.causal_mask[:, 0],
        )
        torch.testing.assert_close(
            titan_q_resid,
            hf_q_resid,
            rtol=1e-6,
            atol=1e-7,
        )
        common_q_resid = hf_q_resid

        # TODO: 没有比较indexer的topk scores, 只比较了topk索引

        # 调用indexer，得到top-k的token索引
        hf_topk = hf_attention.indexer(
            hidden_states,
            common_q_resid,
            self.hf_position_embeddings,
            self.causal_mask[:, 0],
            self.positions,
        )
        titan_topk = titan_attention.indexer(
            hidden_states,
            common_q_resid,
            self.positions,
            self.causal_mask[:, 0],
        )
        # 验证两边的 top-k 索引完全一致
        # 要求shape、元素位置（顺序）、元素值完全一致
        # 因为 topk() 默认返回按 score 排序后的索引
        self.assertTrue(torch.equal(titan_topk, hf_topk))
        torch.testing.assert_close(titan_topk, hf_topk, rtol=0, atol=0)
        torch.testing.assert_close(titan_topk, hf_topk, rtol=0, atol=0)

    # 验证两种实现在moe router上的行为
    def test_router_selection_and_weights_match_transformers_exactly(self) -> None:
        layer_index = 1
        hf_layer = self.hf_model.model.layers[layer_index]
        titan_moe = self.titan_model.layers[str(layer_index)].moe
        normalized_hidden_states = self._normalized_hidden_states(layer_index)

        # 计算路由权重和专家索引
        _, hf_weights_RK, hf_indices_RK = hf_layer.mlp.gate(normalized_hidden_states)
        titan_weights_BLK, titan_indices_BLK, _ = titan_moe.router(
            normalized_hidden_states, titan_moe.expert_bias_E
        )
        # 验证 HF fused router 输出与 TorchTitan router 输出 shape 对齐
        hf_weights_BLK = hf_weights_RK.view_as(titan_weights_BLK)
        hf_indices_BLK = hf_indices_RK.view_as(titan_indices_BLK)
        # 验证专家索引完全一致
        self.assertTrue(torch.equal(titan_indices_BLK, hf_indices_BLK))
        # 验证路由权重数值接近
        torch.testing.assert_close(
            titan_weights_BLK, hf_weights_BLK, rtol=1e-6, atol=1e-7
        )

    # 验证完整 attention 输出
    # MLA + DSA top-k + sparse mask + softmax + value aggregation + output projection
    # 有误差再缩小测试粒度
    def test_attention_output_matches_transformers_fp32(self) -> None:
        layer_index = 0
        hf_attention = self.hf_model.model.layers[layer_index].self_attn
        titan_attention = self.titan_model.layers[str(layer_index)].attention
        normalized_hidden_states = self._normalized_hidden_states(layer_index)

        hf_output_BLD = hf_attention(
            hidden_states=normalized_hidden_states,
            position_embeddings=self.hf_position_embeddings,
            attention_mask=self.causal_mask,
            position_ids=self.positions,
        )[0]
        titan_output_BLD = titan_attention(
            normalized_hidden_states, self.causal_mask, self.positions
        )
        # 验证完整attention输出数值接近
        torch.testing.assert_close(
            titan_output_BLD, hf_output_BLD, rtol=1e-4, atol=1e-5
        )

    # 验证完整 dense block 输出
    # input norm + attetion + FFN
    def test_dense_block_output_matches_transformers_fp32(self) -> None:
        layer_index = 0
        hf_layer = self.hf_model.model.layers[layer_index]
        titan_layer = self.titan_model.layers[str(layer_index)]

        hf_output_BLD = hf_layer(
            self.hidden_states,
            attention_mask=self.causal_mask,
            position_ids=self.positions,
            position_embeddings=self.hf_position_embeddings,
            use_cache=False,
        )[0]
        titan_output_BLD = titan_layer(
            self.hidden_states, self.causal_mask, self.positions
        )
        # 验证完整dense block输出数值接近
        torch.testing.assert_close(
            titan_output_BLD, hf_output_BLD, rtol=1e-4, atol=1e-5
        )


# BF16端到端
class TestGlm5TransformersBfloat16Parity(unittest.TestCase):
    """Single-GPU BF16 output, routed-MoE, and gradient parity acceptance."""

    @classmethod
    def setUpClass(cls) -> None:
        if _TRANSFORMERS_IMPORT_ERROR is not None:
            raise unittest.SkipTest(
                "Transformers GLM-MoE-DSA is unavailable: "
                f"{_TRANSFORMERS_IMPORT_ERROR!r}"
            )
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "GLM-5 Transformers BF16 parity requires one CUDA device"
            )

        cls.device = torch.device("cuda")
        cls.hf_model, cls.titan_model = _build_models(cls.device, seed=53)
        # 统一将模型转换为 BF16
        _convert_models_to_bfloat16(cls.hf_model, cls.titan_model, cls.device)
        cls.batch_size = 1
        cls.sequence_length = 16
        torch.manual_seed(53)
        cls.tokens = torch.randint(
            0,
            2048,
            (cls.batch_size, cls.sequence_length),
            device=cls.device,
            dtype=torch.long,
        )
        cls.positions = torch.arange(
            cls.sequence_length, device=cls.device, dtype=torch.long
        ).expand(cls.batch_size, -1)
        cls.causal_mask = _causal_mask(cls.positions, torch.bfloat16)

    # 验证端到端的 BF16 输出、loss 和梯度与 HF 模型一致
    # TODO: 每一层都需要对比，判断逐层误差累积情况; 然后基于结果逐层对比相关模块
    def test_end_to_end_bfloat16_output_loss_moe_and_gradients(self) -> None:
        """Mapped GLM-5 peers agree across the complete debug training path."""
        self.hf_model.zero_grad(set_to_none=True)
        self.titan_model.zero_grad(set_to_none=True)

        # Exercise the second decoder block from an identical BF16 activation.
        # Layer 1 is the first routed-MoE layer in the debug configuration.
        # 验证routed MoE block output
        with torch.no_grad():
            moe_input = self.hf_model.model.embed_tokens(self.tokens)
            hf_position_embeddings = self.hf_model.model.rotary_emb(
                moe_input, position_ids=self.positions
            )
            hf_moe_block_output = self.hf_model.model.layers[1](
                moe_input,
                attention_mask=self.causal_mask,
                position_ids=self.positions,
                position_embeddings=hf_position_embeddings,
                use_cache=False,
            )[0]
            titan_moe_block_output = self.titan_model.layers["1"](
                moe_input, self.causal_mask, self.positions
            )
        torch.testing.assert_close(
            titan_moe_block_output,
            hf_moe_block_output,
            rtol=5e-2,
            atol=5e-2,
        )

        # 验证模型端到端完整logits和loss
        labels = self.tokens.clone()
        layer_indices = range(len(self.hf_model.model.layers))
        records: dict[str, dict[int, torch.Tensor | None]] = {
            "hf_blocks": {layer_index: None for layer_index in layer_indices},
            "titan_blocks": {layer_index: None for layer_index in layer_indices},
            "hf_indexer": {layer_index: None for layer_index in layer_indices},
            "titan_indexer": {layer_index: None for layer_index in layer_indices},
            "hf_router": {},
            "titan_router": {},
        }
        hook_handles = []
        try:
            for layer_index in layer_indices:
                hf_layer = self.hf_model.model.layers[layer_index]
                titan_layer = self.titan_model.layers[str(layer_index)]

                def capture_hf_block(
                    _module, _inputs, output, *, layer_index=layer_index
                ) -> None:
                    records["hf_blocks"][layer_index] = output[0].detach()

                def capture_titan_block(
                    _module, _inputs, output, *, layer_index=layer_index
                ) -> None:
                    records["titan_blocks"][layer_index] = output.detach()

                def capture_hf_indexer(
                    _module, _inputs, output, *, layer_index=layer_index
                ) -> None:
                    records["hf_indexer"][layer_index] = output.detach()

                def capture_titan_indexer(
                    _module, _inputs, output, *, layer_index=layer_index
                ) -> None:
                    records["titan_indexer"][layer_index] = output.detach()

                hook_handles.append(hf_layer.register_forward_hook(capture_hf_block))
                hook_handles.append(
                    titan_layer.register_forward_hook(capture_titan_block)
                )
                hook_handles.append(
                    hf_layer.self_attn.indexer.register_forward_hook(capture_hf_indexer)
                )
                hook_handles.append(
                    titan_layer.attention.indexer.register_forward_hook(
                        capture_titan_indexer
                    )
                )
                if getattr(titan_layer, "moe_enabled", False):
                    records["hf_router"][layer_index] = None
                    records["titan_router"][layer_index] = None

                    def capture_hf_router(
                        _module, inputs, output, *, layer_index=layer_index
                    ) -> None:
                        batch_size, sequence_length = inputs[0].shape[:2]
                        records["hf_router"][layer_index] = (
                            output[2].view(batch_size, sequence_length, -1).detach()
                        )

                    def capture_titan_router(
                        _module, _inputs, output, *, layer_index=layer_index
                    ) -> None:
                        records["titan_router"][layer_index] = output[1].detach()

                    hook_handles.append(
                        hf_layer.mlp.gate.register_forward_hook(capture_hf_router)
                    )
                    hook_handles.append(
                        titan_layer.moe.router.register_forward_hook(
                            capture_titan_router
                        )
                    )

            hf_outputs = self.hf_model(
                input_ids=self.tokens,
                position_ids=self.positions,
                labels=labels,
                use_cache=False,
            )
            titan_logits = self.titan_model(self.tokens, positions=self.positions)
        finally:
            for handle in hook_handles:
                handle.remove()

        hf_logits = hf_outputs.logits
        hf_loss = hf_outputs.loss
        titan_loss = F.cross_entropy(
            titan_logits[:, :-1].float().reshape(-1, 2048),
            labels[:, 1:].reshape(-1),
        )

        try:
            torch.testing.assert_close(
                titan_logits,
                hf_logits,
                rtol=5e-2,
                atol=5e-2,
            )
        except AssertionError as error:
            diagnostics = _format_parity_diagnostics(records, self.positions)
            raise AssertionError(
                f"{error}\n\nGLM-5 parity diagnostics:\n{diagnostics}"
            ) from error
        torch.testing.assert_close(titan_loss, hf_loss, rtol=5e-2, atol=5e-2)

        hf_loss.backward()
        titan_loss.backward()

        hf_layer0 = self.hf_model.model.layers[0]
        titan_layer0 = self.titan_model.layers["0"]
        hf_layer1 = self.hf_model.model.layers[1]
        titan_layer1 = self.titan_model.layers["1"]
        expert_width = self.hf_model.config.moe_intermediate_size
        # 验证梯度数值接近, embedding/attention/dense ffn/router gate/expert up&down/lm_head gradient
        # 验证indexer参数不参与梯度计算
        gradient_pairs = (
            (
                "embedding",
                self.hf_model.model.embed_tokens.weight.grad,
                self.titan_model.tok_embeddings.weight.grad,
            ),
            (
                "layer0 q_a",
                hf_layer0.self_attn.q_a_proj.weight.grad,
                titan_layer0.attention.wq_a.weight.grad,
            ),
            (
                "layer0 dense gate",
                hf_layer0.mlp.gate_proj.weight.grad,
                titan_layer0.feed_forward.w1.weight.grad,
            ),
            (
                "layer1 router",
                hf_layer1.mlp.gate.weight.grad,
                titan_layer1.moe.router.gate.weight.grad,
            ),
            (
                "layer1 routed gate",
                hf_layer1.mlp.experts.gate_up_proj.grad[:, :expert_width],
                titan_layer1.moe.routed_experts.inner_experts.w1_EFD.grad,
            ),
            (
                "layer1 routed up",
                hf_layer1.mlp.experts.gate_up_proj.grad[:, expert_width:],
                titan_layer1.moe.routed_experts.inner_experts.w3_EFD.grad,
            ),
            (
                "layer1 routed down",
                hf_layer1.mlp.experts.down_proj.grad,
                titan_layer1.moe.routed_experts.inner_experts.w2_EDF.grad,
            ),
            (
                "lm head",
                self.hf_model.lm_head.weight.grad,
                self.titan_model.lm_head.weight.grad,
            ),
        )
        for name, hf_gradient, titan_gradient in gradient_pairs:
            self.assertIsNotNone(hf_gradient, f"HF {name} gradient is missing")
            self.assertIsNotNone(
                titan_gradient, f"TorchTitan {name} gradient is missing"
            )
            assert hf_gradient is not None
            assert titan_gradient is not None
            torch.testing.assert_close(
                titan_gradient,
                hf_gradient,
                rtol=5e-2,
                atol=5e-2,
            )

        for model_name, model in (
            ("HF", self.hf_model),
            ("TorchTitan", self.titan_model),
        ):
            for parameter_name, parameter in model.named_parameters():
                if ".indexer." in parameter_name:
                    self.assertIsNone(
                        parameter.grad,
                        f"{model_name} indexer parameter unexpectedly received "
                        f"a gradient: {parameter_name}",
                    )
