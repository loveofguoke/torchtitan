# GLM-5 Debug Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native TorchTitan `glm5_debugmodel` that runs forward, loss, and backward on CPU and aligns on one CUDA GPU with Transformers `GlmMoeDsaForCausalLM`.

**Architecture:** Implement GLM-5 as independent `Glm5DsaIndexer`, `Glm5Attention`, `Glm5TransformerBlock`, and `Glm5Model` classes composed from TorchTitan common modules. Use the common dense FFN and MoE stack, keep the DSA eager path local to `glm5/model.py`, and convert fused HF expert tensors through a strict bidirectional adapter.

**Tech Stack:** Python 3.12, PyTorch, TorchTitan configurable modules, `unittest`/pytest, Transformers 5.14.1, one optional CUDA device.

## Global Constraints

- Work only in `/home/h50064611/torchtitan/pytorch-torchtitan-glm5` on branch `feat/glm5-model`.
- Follow `/home/h50064611/torchtitan/pytorch-torchtitan-glm5/AGENTS.md` and `torchtitan/models/README.md`.
- Reuse `torchtitan/models/common` for embedding, linear, normalization, RoPE, dense FFN, routing, experts, MoE, decoder, and transformer-block infrastructure.
- Do not inherit from DeepSeek-V3 model or adapter classes.
- Keep GLM-5-specific DSA and MLA computation in `torchtitan/models/glm5/model.py`.
- The only model flavor in this milestone is `debugmodel`: vocab 2048, dim 256, 4 layers, 8 attention heads, Q-LoRA 128, KV-LoRA 64, QK-nope 32, QK-RoPE 32, V-head 64, dense hidden 1024, MoE hidden 256, 8 routed experts, 1 shared expert, top-2 routing, 1 dense layer, 4 index heads, index head dim 64, index top-k 8, max sequence length 128, RoPE theta 1,000,000, RMS epsilon 1e-5, and attention dropout 0.
- Every layer has a full indexer. Cross-layer sharing, MTP, KV cache, incremental decoding, Flash-MLA, auxiliary indexer training, and distributed TP/CP/PP/EP/FSDP/HSDP are unsupported.
- The indexer runs under `torch.no_grad()` and `indexer.weights_proj.weight` remains FP32 when the rest of the model is converted to BF16.
- Use shape suffixes (`B`, `L`, `D`, `N`, `H`, `K`, `E`, `F`) on non-trivial tensor names.
- Use `ValueError` for invalid architecture input and `NotImplementedError` for explicitly unsupported runtime modes.
- New Python comments and docstrings are ASCII.
- Tests reuse `tests/unit_tests`; common operators are not retested independently.
- Use this Python for all focused commands:

```bash
/home/h50064611/torchtitan/.miniconda3/envs/torchtitan-dev/bin/python
```

## File Map

- Create `torchtitan/models/glm5/model.py`: DSA indexer, MLA attention, block, decoder, masks, config validation, and FLOP accounting.
- Create `torchtitan/models/glm5/__init__.py`: sub-config builders, debug architecture, model registry, and public exports.
- Create `torchtitan/models/glm5/state_dict_adapter.py`: strict HF/TorchTitan mappings and fused expert conversion.
- Create `torchtitan/models/glm5/sharding.py`: reject unsupported parallel degrees and non-default SPMD execution.
- Create `torchtitan/models/glm5/parallelize.py`: single-device activation checkpointing and compile hooks.
- Create `torchtitan/models/glm5/config_registry.py`: trainer-facing `glm5_debugmodel()` configuration.
- Create `torchtitan/models/glm5/README.md`: usage, reference, tested scope, and roadmap.
- Modify `torchtitan/models/__init__.py`: register `glm5` as a supported module.
- Modify `tests/unit_tests/test_config_manager.py`: reuse the existing CLI config test.
- Create `tests/unit_tests/test_glm5.py`: CPU component, mask, model, adapter, and backward coverage.
- Create `tests/unit_tests/test_glm5_parity.py`: optional Transformers component and one-CUDA parity coverage.

---

### Task 1: Implement the GLM-5 DSA indexer

**Files:**
- Create: `torchtitan/models/glm5/__init__.py`
- Create: `torchtitan/models/glm5/model.py`
- Create: `tests/unit_tests/test_glm5.py`

**Interfaces:**
- Consumes: common `Linear`, `LayerNorm`, `ComplexRoPE`, and `Module.Config.build()`.
- Produces: `Glm5DsaIndexer.Config` and `Glm5DsaIndexer.forward(hidden_states_BLD, q_resid_BLR, positions_BL, attention_mask_BLL) -> torch.Tensor` with `int32` shape `[B, L, min(index_topk, L)]`.

- [ ] **Step 1: Write failing indexer tests**

Add the repository license header, imports, and a `TestGlm5DsaIndexer(unittest.TestCase)` class to `tests/unit_tests/test_glm5.py`. Define `_indexer_config()` with `D=16`, Q-LoRA rank `8`, 2 index heads, index head dim `8`, RoPE dim `4`, top-k `3`, and `ComplexRoPE.Config(max_seq_len=8, theta=1_000_000)`. Add these tests:

```python
def test_indexer_returns_masked_int32_topk(self):
    indexer = _indexer_config().build()
    indexer.init_states()
    hidden_states_BLD = torch.randn(2, 5, 16)
    q_resid_BLR = torch.randn(2, 5, 8)
    positions_BL = torch.arange(5).expand(2, -1)
    attention_mask_BLL = torch.full((2, 5, 5), float("-inf"))
    attention_mask_BLL.masked_fill_(
        torch.ones(5, 5, dtype=torch.bool).tril().unsqueeze(0), 0.0
    )

    topk_indices_BLK = indexer(
        hidden_states_BLD,
        q_resid_BLR,
        positions_BL,
        attention_mask_BLL,
    )

    self.assertEqual(topk_indices_BLK.dtype, torch.int32)
    self.assertEqual(topk_indices_BLK.shape, (2, 5, 3))
    query_positions_BL1 = positions_BL.unsqueeze(-1)
    self.assertTrue(torch.all(topk_indices_BLK <= query_positions_BL1))

def test_indexer_matches_independent_reference(self):
    torch.manual_seed(17)
    indexer = _indexer_config().build()
    indexer.init_states()
    hidden_states_BLD = torch.randn(1, 4, 16)
    q_resid_BLR = torch.randn(1, 4, 8)
    positions_BL = torch.arange(4).unsqueeze(0)
    attention_mask_BLL = torch.zeros(1, 4, 4).masked_fill(
        ~torch.ones(4, 4, dtype=torch.bool).tril().unsqueeze(0),
        float("-inf"),
    )

    actual_BLK = indexer(
        hidden_states_BLD,
        q_resid_BLR,
        positions_BL,
        attention_mask_BLL,
    )
    expected_BLK = _reference_indexer_topk(
        indexer,
        hidden_states_BLD,
        q_resid_BLR,
        positions_BL,
        attention_mask_BLL,
    )
    self.assertTrue(torch.equal(actual_BLK, expected_BLK))

def test_indexer_is_no_grad_and_keeps_weights_projection_fp32(self):
    indexer = _indexer_config().build()
    indexer.init_states()
    indexer.bfloat16()
    self.assertEqual(indexer.weights_proj.weight.dtype, torch.float32)
    out_BLK = indexer(
        torch.randn(1, 4, 16, dtype=torch.bfloat16),
        torch.randn(1, 4, 8, dtype=torch.bfloat16),
        torch.arange(4).unsqueeze(0),
        torch.zeros(1, 4, 4, dtype=torch.bfloat16),
    )
    self.assertFalse(out_BLK.requires_grad)
```

Implement `_reference_indexer_topk()` in the test with direct `F.linear`, `F.layer_norm`, adjacent-pair complex rotation, FP32 score matmul, ReLU, per-head weighting, additive mask, and `topk`. Do not call `Glm5DsaIndexer.forward()` or any helper from `model.py` inside the reference.

- [ ] **Step 2: Run the tests and verify the import failure**

Run:

```bash
/home/h50064611/torchtitan/.miniconda3/envs/torchtitan-dev/bin/python -m pytest tests/unit_tests/test_glm5.py::TestGlm5DsaIndexer -v
```

Expected: FAIL because `torchtitan.models.glm5.model.Glm5DsaIndexer` does not exist.

- [ ] **Step 3: Implement the indexer config, validation, and forward**

In `model.py`, define the following config fields and behavior:

```python
class Glm5DsaIndexer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        q_lora_rank: int
        n_heads: int
        head_dim: int
        qk_rope_head_dim: int
        index_topk: int
        wq_b: Linear.Config
        wk: Linear.Config
        k_norm: LayerNorm.Config
        weights_proj: Linear.Config
        rope: ComplexRoPE.Config

        def __post_init__(self) -> None:
            if self.q_lora_rank <= 0:
                raise ValueError("GLM-5 DSA requires q_lora_rank > 0.")
            if self.head_dim < self.qk_rope_head_dim:
                raise ValueError(
                    "index_head_dim must be >= qk_rope_head_dim."
                )
            if self.qk_rope_head_dim % 2 != 0:
                raise ValueError("qk_rope_head_dim must be even.")
            if self.index_topk <= 0:
                raise ValueError("index_topk must be > 0.")

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = config.head_dim**-0.5
        self.wq_b = config.wq_b.build()
        self.wk = config.wk.build()
        self.k_norm = config.k_norm.build()
        self.weights_proj = config.weights_proj.build().float()
        self.rope = config.rope.build()

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.weights_proj.float()
        return self

    @torch.no_grad()
    def forward(
        self,
        hidden_states_BLD: torch.Tensor,
        q_resid_BLR: torch.Tensor,
        positions_BL: torch.Tensor,
        attention_mask_BLL: torch.Tensor | None,
    ) -> torch.Tensor:
        B, L, _ = hidden_states_BLD.shape
        q_BLNH = self.wq_b(q_resid_BLR).view(
            B, L, self.n_heads, self.head_dim
        )
        q_rot_BLNR, q_pass_BLNP = torch.split(
            q_BLNH,
            [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim],
            dim=-1,
        )
        k_BL1H = self.k_norm(self.wk(hidden_states_BLD)).unsqueeze(2)
        k_rot_BL1R, k_pass_BL1P = torch.split(
            k_BL1H,
            [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim],
            dim=-1,
        )
        q_rot_BLNR, k_rot_BL1R = self.rope(
            q_rot_BLNR, k_rot_BL1R, positions_BL
        )
        q_BLNH = torch.cat((q_rot_BLNR, q_pass_BLNP), dim=-1)
        k_BLH = torch.cat((k_rot_BL1R, k_pass_BL1P), dim=-1).squeeze(2)

        scores_BNLL = torch.matmul(
            q_BLNH.float().transpose(1, 2),
            k_BLH.float().transpose(1, 2).unsqueeze(1),
        ) * self.softmax_scale
        scores_BNLL = F.relu(scores_BNLL)
        weights_BLN = self.weights_proj(
            hidden_states_BLD.to(self.weights_proj.weight.dtype)
        ).float() * (self.n_heads**-0.5)
        index_scores_BLL = torch.matmul(
            weights_BLN.unsqueeze(-2), scores_BNLL.transpose(1, 2)
        ).squeeze(-2)
        if attention_mask_BLL is not None:
            index_scores_BLL = index_scores_BLL + attention_mask_BLL.float()
        else:
            key_positions_11L = torch.arange(L, device=positions_BL.device)[
                None, None, :
            ]
            index_scores_BLL = index_scores_BLL.masked_fill(
                key_positions_11L > positions_BL.unsqueeze(-1), float("-inf")
            )
        topk = min(self.index_topk, index_scores_BLL.shape[-1])
        return index_scores_BLL.topk(topk, dim=-1).indices.to(torch.int32)
```

Export `Glm5DsaIndexer` from `glm5/__init__.py`.

- [ ] **Step 4: Run the indexer tests**

Run the command from Step 2. Expected: all `TestGlm5DsaIndexer` tests PASS.

- [ ] **Step 5: Commit the indexer**

```bash
git add torchtitan/models/glm5/__init__.py torchtitan/models/glm5/model.py tests/unit_tests/test_glm5.py
git commit -m "feat: add GLM-5 DSA indexer"
```

---

### Task 2: Implement GLM-5 MLA with dynamic sparse attention

**Files:**
- Modify: `torchtitan/models/glm5/model.py`
- Modify: `torchtitan/models/glm5/__init__.py`
- Modify: `tests/unit_tests/test_glm5.py`

**Interfaces:**
- Consumes: `Glm5DsaIndexer.Config` and dense additive masks shaped `[B, 1, L, L]`.
- Produces: `Glm5Attention.Config` and `Glm5Attention.forward(x_BLD, attention_masks, positions_BL) -> torch.Tensor` shaped `[B, L, D]`.

- [ ] **Step 1: Write failing attention tests**

Add `_attention_config()` using `D=16`, 2 heads, Q-LoRA 8, KV-LoRA 4, QK-nope 4, QK-RoPE 4, V-head 4, index top-k 2, and zero dropout. Add:

```python
def test_attention_topk_cannot_reopen_causal_mask(self):
    attention = _attention_config().build()
    attention.init_states()
    x_BLD = torch.randn(1, 4, 16)
    positions_BL = torch.arange(4).unsqueeze(0)
    base_mask_B1LL = _dense_causal_mask(positions_BL, dtype=x_BLD.dtype)
    future_selecting_topk_BLK = torch.tensor(
        [[[3, 2], [3, 2], [3, 2], [3, 2]]], dtype=torch.int32
    )
    with mock.patch.object(
        attention.indexer,
        "forward",
        return_value=future_selecting_topk_BLK,
    ):
        actual_BLD = attention(x_BLD, base_mask_B1LL, positions_BL)
    expected_BLD = _reference_sparse_attention(
        attention,
        x_BLD,
        base_mask_B1LL,
        positions_BL,
        future_selecting_topk_BLK,
    )
    torch.testing.assert_close(actual_BLD, expected_BLD, rtol=1e-5, atol=1e-6)

def test_attention_output_shape_and_backward(self):
    attention = _attention_config().build()
    attention.init_states()
    x_BLD = torch.randn(2, 5, 16, requires_grad=True)
    positions_BL = torch.arange(5).expand(2, -1)
    mask_B1LL = _dense_causal_mask(positions_BL, dtype=x_BLD.dtype)
    output_BLD = attention(x_BLD, mask_B1LL, positions_BL)
    self.assertEqual(output_BLD.shape, x_BLD.shape)
    output_BLD.square().mean().backward()
    self.assertIsNotNone(attention.wq_a.weight.grad)
    self.assertIsNotNone(attention.wo.weight.grad)
    self.assertTrue(
        all(parameter.grad is None for parameter in attention.indexer.parameters())
    )
```

The test helper `_reference_sparse_attention()` must use the module weights through `F.linear` but independently perform reshape, split, common RoPE application, selected-key boolean mask construction, FP32 softmax, value aggregation, and output projection. It must combine the top-k mask with the supplied base mask before softmax.

- [ ] **Step 2: Run attention tests and verify failure**

```bash
/home/h50064611/torchtitan/.miniconda3/envs/torchtitan-dev/bin/python -m pytest tests/unit_tests/test_glm5.py::TestGlm5Attention -v
```

Expected: FAIL because `Glm5Attention` is not defined.

- [ ] **Step 3: Implement `Glm5Attention`**

Add a dataclass config with these exact fields: `dim`, `n_heads`, `q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`, `attention_dropout`, `wq_a`, `q_norm`, `wq_b`, `wkv_a`, `kv_norm`, `wkv_b`, `wo`, `rope`, `indexer`, and the inherited `inner_attention`. Add a read-only `qk_head_dim` property returning `qk_nope_head_dim + qk_rope_head_dim`; later FLOP accounting consumes this property. Use `FlexAttention.Config()` only as the protocol marker that tells the trainer this model consumes document masks; do not build or call FlexAttention in the GLM eager path.

Validation must reject non-positive Q-LoRA rank, odd RoPE dimension, non-positive head counts, non-zero dropout outside `[0, 1)`, and projection/head dimension mismatches.

Implement the full forward in this order:

```python
q_resid_BLR = self.q_norm(self.wq_a(x_BLD))
q_BLNH = self.wq_b(q_resid_BLR).view(B, L, self.n_heads, self.qk_head_dim)
q_nope_BLNP, q_rope_BLNR = torch.split(
    q_BLNH, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
)
compressed_kv_BLC = self.wkv_a(x_BLD)
kv_BLR, k_rope_BL1R = torch.split(
    compressed_kv_BLC,
    [self.kv_lora_rank, self.qk_rope_head_dim],
    dim=-1,
)
kv_BLR = self.kv_norm(kv_BLR)
k_rope_BL1R = k_rope_BL1R.unsqueeze(2)
q_rope_BLNR, k_rope_BL1R = self.rope(
    q_rope_BLNR, k_rope_BL1R, positions_BL
)
q_BLNH = torch.cat((q_nope_BLNP, q_rope_BLNR), dim=-1)
kv_BLNX = self.wkv_b(kv_BLR).view(
    B, L, self.n_heads, self.qk_nope_head_dim + self.v_head_dim
)
k_nope_BLNP, v_BLNV = torch.split(
    kv_BLNX, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
)
k_BLNH = torch.cat(
    (k_nope_BLNP, k_rope_BL1R.expand(-1, -1, self.n_heads, -1)),
    dim=-1,
)
topk_indices_BLK = self.indexer(
    x_BLD,
    q_resid_BLR,
    positions_BL,
    attention_masks[:, 0],
)
selected_BLL = torch.zeros(
    B, L, L, dtype=torch.bool, device=x_BLD.device
).scatter(-1, topk_indices_BLK.long(), True)
min_value = torch.finfo(x_BLD.dtype).min
sparse_mask_B1LL = attention_masks.masked_fill(
    ~selected_BLL.unsqueeze(1), min_value
)
scores_BNLL = torch.matmul(
    q_BLNH.transpose(1, 2), k_BLNH.transpose(1, 2).transpose(-1, -2)
) * self.softmax_scale
scores_BNLL = scores_BNLL + sparse_mask_B1LL
probs_BNLL = F.softmax(scores_BNLL, dim=-1, dtype=torch.float32).to(q_BLNH.dtype)
probs_BNLL = F.dropout(
    probs_BNLL, p=self.attention_dropout, training=self.training
)
output_BLNV = torch.matmul(probs_BNLL, v_BLNV.transpose(1, 2)).transpose(1, 2)
return self.wo(output_BLNV.contiguous().view(B, L, -1))
```

Require `attention_masks` to be a dense tensor and generate default positions only when `positions_BL is None`. Export the class.

- [ ] **Step 4: Run indexer and attention tests**

```bash
/home/h50064611/torchtitan/.miniconda3/envs/torchtitan-dev/bin/python -m pytest tests/unit_tests/test_glm5.py::TestGlm5DsaIndexer tests/unit_tests/test_glm5.py::TestGlm5Attention -v
```

Expected: PASS.

- [ ] **Step 5: Commit attention**

```bash
git add torchtitan/models/glm5/model.py torchtitan/models/glm5/__init__.py tests/unit_tests/test_glm5.py
git commit -m "feat: add GLM-5 sparse MLA attention"
```

---

### Task 3: Build the reduced GLM-5 decoder from common dense and MoE modules

**Files:**
- Modify: `torchtitan/models/glm5/model.py`
- Modify: `torchtitan/models/glm5/__init__.py`
- Modify: `tests/unit_tests/test_glm5.py`

**Interfaces:**
- Consumes: `Glm5Attention`, common `FeedForward`, `MoE`, `RMSNorm`, `Embedding`, `Linear`, and common config builders.
- Produces: `Glm5TransformerBlock`, `Glm5Model`, `make_glm5_attention_config()`, `build_glm5_layers()`, and `glm5_configs["debugmodel"]`.

- [ ] **Step 1: Write failing architecture and mask tests**

Add tests that call `glm5_configs["debugmodel"]()` and assert every approved debug value, one dense layer followed by three MoE layers, and one indexer on every attention. Add mask tests:

```python
def test_dense_mask_enforces_causality_and_document_boundaries(self):
    model = _build_debug_model()
    positions_BL = torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.long)
    mask_B1LL = model.get_attention_masks(positions_BL)
    min_value = torch.finfo(mask_B1LL.dtype).min
    self.assertEqual(mask_B1LL.shape, (1, 1, 5, 5))
    self.assertEqual(mask_B1LL[0, 0, 4, 3].item(), 0.0)
    self.assertEqual(mask_B1LL[0, 0, 4, 1].item(), min_value)
    self.assertEqual(mask_B1LL[0, 0, 1, 2].item(), min_value)

def test_debug_model_forward_shape(self):
    model = _build_debug_model()
    tokens_BL = torch.randint(0, 2048, (2, 12))
    positions_BL = torch.arange(12).expand(2, -1)
    logits_BLV = model(tokens_BL, positions=positions_BL)
    self.assertEqual(logits_BLV.shape, (2, 12, 2048))
```

`_build_debug_model()` must call the config factory, build, `init_states()`, and `eval()`.

- [ ] **Step 2: Run the model tests and verify failure**

```bash
/home/h50064611/torchtitan/.miniconda3/envs/torchtitan-dev/bin/python -m pytest tests/unit_tests/test_glm5.py::TestGlm5Model -v
```

Expected: FAIL because the block, decoder, and debug config are missing.

- [ ] **Step 3: Implement the block and decoder**

Implement `Glm5TransformerBlock` with the standard pre-norm residual flow:

```python
def forward(self, x_BLD, attention_masks, positions=None):
    x_BLD = x_BLD + self.attention(
        self.attention_norm(x_BLD), attention_masks, positions
    )
    normalized_BLD = self.ffn_norm(x_BLD)
    ffn_output_BLD = (
        self.moe(normalized_BLD)
        if self.moe_enabled
        else self.feed_forward(normalized_BLD)
    )
    return x_BLD + ffn_output_BLD
```

Implement `Glm5Model.Config.update_from_config()` by first calling `validate_glm5_parallelism(config.parallelism)` from Task 5's final interface and then `Decoder.Config.update_from_config()`. Until Task 5 creates that module, define the validation import inside the method so importing and unit-building the model does not require the file.

Implement FLOP accounting as:

```python
nparams, base_flops = get_moe_model_nparams_and_flops(
    self,
    model,
    attention.n_heads,
    attention.qk_head_dim + attention.v_head_dim,
    seq_len,
)
dsa_flops = (
    2
    * len(self.layers)
    * attention.indexer.n_heads
    * attention.indexer.head_dim
    * seq_len
)
return nparams, base_flops + dsa_flops
```

Override `Glm5Model.get_attention_masks()` using document IDs computed by `(positions_BL == 0).cumsum(dim=1)`, a lower-triangular causal predicate, and the token embedding weight dtype. Return `0` for allowed pairs and `torch.finfo(dtype).min` otherwise. Override `forward()` to create default positions and the dense mask when the caller does not supply them, then delegate to `Decoder.forward()`.

- [ ] **Step 4: Implement the config builders and exact debug flavor**

In `glm5/__init__.py`, define initializer dictionaries for Linear, LayerNorm, RMSNorm, Embedding, output projections, and grouped experts. Define:

```python
def make_glm5_attention_config(
    *,
    layer_id: int,
    dim: int,
    n_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    attention_dropout: float,
    rope: ComplexRoPE.Config,
) -> Glm5Attention.Config:
```

The builder must create every projection with its exact in/out dimensions, copy the RoPE config with `dataclasses.replace`, use `LayerNorm.Config(eps=1e-6)` for indexer `k_norm`, use `RMSNorm.Config(eps=1e-6)` for Q/KV low-rank norms to match HF, and set `inner_attention=FlexAttention.Config()` as the masked-backend protocol marker.

Define `build_glm5_layers()` with the approved architecture fields. For layer IDs below `n_dense_layers`, use `make_ffn_config(dim, dense_hidden_dim)`. For later layers, use:

```python
make_moe_config(
    num_experts=num_experts,
    router=make_router_config(
        dim=dim,
        num_experts=num_experts,
        top_k=router_top_k,
        score_func="sigmoid",
        num_expert_groups=router_num_expert_groups,
        num_limited_groups=router_num_limited_groups,
        route_scale=router_route_scale,
        route_norm=True,
        gate_param_init=_depth_init(layer_id),
    ),
    routed_experts=make_routed_experts_config(
        dim=dim,
        hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        top_k=router_top_k,
        param_init=_depth_experts_init(layer_id),
        comm_backend="standard",
    ),
    shared_experts=make_ffn_config(
        dim=dim,
        hidden_dim=moe_hidden_dim * num_shared_experts,
        w1_param_init=_LINEAR_INIT,
        w2w3_param_init=_depth_init(layer_id),
    ),
    load_balance_coeff=1e-3,
)
```

Validate dense-layer count, group divisibility, group top-k, and expert top-k before building layers. Define `_debugmodel()` with every exact value from Global Constraints and `ComplexRoPE.Config(dim=32, max_seq_len=128, theta=1_000_000, scaling="none")`. Export:

```python
glm5_configs = {"debugmodel": _debugmodel}
```

- [ ] **Step 5: Run all CPU component/model tests**

```bash
/home/h50064611/torchtitan/.miniconda3/envs/torchtitan-dev/bin/python -m pytest tests/unit_tests/test_glm5.py -v
```

Expected: all tests currently in `test_glm5.py` PASS.

- [ ] **Step 6: Commit the complete single-device model core**

```bash
git add torchtitan/models/glm5/model.py torchtitan/models/glm5/__init__.py tests/unit_tests/test_glm5.py
git commit -m "feat: build GLM-5 debug decoder"
```

---

### Task 4: Add the strict bidirectional HF state-dict adapter

**Files:**
- Create: `torchtitan/models/glm5/state_dict_adapter.py`
- Modify: `torchtitan/models/glm5/__init__.py`
- Modify: `tests/unit_tests/test_glm5.py`

**Interfaces:**
- Consumes: a plain single-device GLM-5 state dict and Transformers 5.14.1 GLM-MoE-DSA keys.
- Produces: `Glm5StateDictAdapter.from_hf()` and `.to_hf()` with value-preserving round-trip for all supported tensors.

- [ ] **Step 1: Write failing adapter tests**

Add `TestGlm5StateDictAdapter` with these tests:

```python
def test_fused_expert_gate_up_order(self):
    config = glm5_configs["debugmodel"]()
    adapter = Glm5StateDictAdapter(config, hf_assets_path=None)
    E, F, D = 8, 256, 256
    gate_EFD = torch.arange(E * F * D).reshape(E, F, D)
    up_EFD = gate_EFD + gate_EFD.numel()
    down_EDF = torch.arange(E * D * F).reshape(E, D, F)
    hf_state = {
        "model.layers.1.mlp.experts.gate_up_proj": torch.cat(
            (gate_EFD, up_EFD), dim=1
        ),
        "model.layers.1.mlp.experts.down_proj": down_EDF,
    }
    titan_state = adapter.from_hf(hf_state)
    self.assertTrue(torch.equal(
        titan_state["layers.1.moe.routed_experts.inner_experts.w1_EFD"],
        gate_EFD,
    ))
    self.assertTrue(torch.equal(
        titan_state["layers.1.moe.routed_experts.inner_experts.w3_EFD"],
        up_EFD,
    ))
    self.assertTrue(torch.equal(
        titan_state["layers.1.moe.routed_experts.inner_experts.w2_EDF"],
        down_EDF,
    ))

def test_full_state_dict_roundtrip(self):
    config = glm5_configs["debugmodel"]()
    model = config.build()
    model.init_states()
    adapter = Glm5StateDictAdapter(config, hf_assets_path=None)
    original = model.state_dict()
    restored = adapter.from_hf(adapter.to_hf(original))
    self.assertEqual(set(restored), set(original))
    for key in original:
        self.assertTrue(torch.equal(restored[key], original[key]), key)

def test_unknown_hf_key_is_rejected(self):
    adapter = Glm5StateDictAdapter(
        glm5_configs["debugmodel"](), hf_assets_path=None
    )
    with self.assertRaisesRegex(KeyError, "unmapped HF key"):
        adapter.from_hf({"model.layers.0.unexpected.weight": torch.ones(1)})
```

Also test direct indexer mappings, FP32 preservation of `e_score_correction_bias`, exact warning/skip of `model.layers.4.*` as the configured next MTP namespace, and rejection of `model.layers.5.*`.

- [ ] **Step 2: Run adapter tests and verify failure**

```bash
/home/h50064611/torchtitan/.miniconda3/envs/torchtitan-dev/bin/python -m pytest tests/unit_tests/test_glm5.py::TestGlm5StateDictAdapter -v
```

Expected: FAIL because `Glm5StateDictAdapter` is missing.

- [ ] **Step 3: Implement exact key maps and fused conversion**

Subclass `MoEStateDictAdapter`, call `self._validate_hf_rope_config(ComplexRoPE.Config)` in `from_hf()`, and define direct suffix maps covering:

```python
HF_TO_TITAN_LAYER = {
    "self_attn.q_a_proj.weight": "attention.wq_a.weight",
    "self_attn.q_a_layernorm.weight": "attention.q_norm.weight",
    "self_attn.q_b_proj.weight": "attention.wq_b.weight",
    "self_attn.kv_a_proj_with_mqa.weight": "attention.wkv_a.weight",
    "self_attn.kv_a_layernorm.weight": "attention.kv_norm.weight",
    "self_attn.kv_b_proj.weight": "attention.wkv_b.weight",
    "self_attn.o_proj.weight": "attention.wo.weight",
    "self_attn.indexer.wq_b.weight": "attention.indexer.wq_b.weight",
    "self_attn.indexer.wk.weight": "attention.indexer.wk.weight",
    "self_attn.indexer.k_norm.weight": "attention.indexer.k_norm.weight",
    "self_attn.indexer.k_norm.bias": "attention.indexer.k_norm.bias",
    "self_attn.indexer.weights_proj.weight": "attention.indexer.weights_proj.weight",
    "mlp.gate_proj.weight": "feed_forward.w1.weight",
    "mlp.up_proj.weight": "feed_forward.w3.weight",
    "mlp.down_proj.weight": "feed_forward.w2.weight",
    "mlp.gate.weight": "moe.router.gate.weight",
    "mlp.gate.e_score_correction_bias": "moe.expert_bias_E",
    "mlp.shared_experts.gate_proj.weight": "moe.shared_experts.w1.weight",
    "mlp.shared_experts.up_proj.weight": "moe.shared_experts.w3.weight",
    "mlp.shared_experts.down_proj.weight": "moe.shared_experts.w2.weight",
    "input_layernorm.weight": "attention_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
}
```

Top-level maps are `model.embed_tokens.weight -> tok_embeddings.weight`, `model.norm.weight -> norm.weight`, and `lm_head.weight -> lm_head.weight`.

For `from_hf()`, match `model.layers.(\d+).(suffix)`. Split `mlp.experts.gate_up_proj` equally on dimension 1 into gate then up; map `mlp.experts.down_proj` directly. Reject a fused first dimension different from configured experts or an odd second dimension. For `to_hf()`, gather `w1_EFD` and `w3_EFD` per layer and emit only after both are present; emit `w2_EDF` directly. Raise `KeyError` for unknown or incomplete mappings.

Recognize only `model.rotary_emb.inv_freq` and `model.rotary_emb.original_inv_freq` as recomputable ignored HF buffers. Warn and skip only keys whose layer index equals `len(model_config.layers)`; all later indices remain errors.

- [ ] **Step 4: Run adapter and full CPU tests**

```bash
/home/h50064611/torchtitan/.miniconda3/envs/torchtitan-dev/bin/python -m pytest tests/unit_tests/test_glm5.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit the adapter**

```bash
git add torchtitan/models/glm5/state_dict_adapter.py torchtitan/models/glm5/__init__.py tests/unit_tests/test_glm5.py
git commit -m "feat: add GLM-5 checkpoint adapter"
```

---

### Task 5: Wire model registration and the single-device training path

**Files:**
- Create: `torchtitan/models/glm5/sharding.py`
- Create: `torchtitan/models/glm5/parallelize.py`
- Create: `torchtitan/models/glm5/config_registry.py`
- Modify: `torchtitan/models/glm5/__init__.py`
- Modify: `torchtitan/models/__init__.py`
- Modify: `tests/unit_tests/test_config_manager.py`
- Modify: `tests/unit_tests/test_glm5.py`

**Interfaces:**
- Consumes: `glm5_configs["debugmodel"]`, Trainer config types, `apply_compile`, and activation-checkpoint config.
- Produces: `model_registry("debugmodel")`, `glm5_debugmodel()`, and a `parallelize_glm5()` function that rejects every multi-rank layout.

- [ ] **Step 1: Write failing registration and rejection tests**

Add beside the existing DeepSeek-V3 CLI test:

```python
def test_glm5_config(self):
    config_manager = ConfigManager()
    config = config_manager.parse_args(
        ["--module", "glm5", "--config", "glm5_debugmodel"]
    )
    assert config.model_spec.name == "glm5"
    assert config.model_spec.flavor == "debugmodel"
```

In `test_glm5.py`, assert `model_registry("debugmodel")` returns `Glm5Model.Config`, `Glm5StateDictAdapter`, no pipeline function, and the MoE load-balancing post-optimizer hook. Construct `ParallelismConfig(tensor_parallel_degree=2)` and verify `validate_glm5_parallelism()` raises `NotImplementedError` naming TP. Test equivalent rejection for CP, PP, EP, DP replicate, DP shard, and `spmd_backend="full_dtensor"`.

- [ ] **Step 2: Run registration tests and verify failure**

```bash
/home/h50064611/torchtitan/.miniconda3/envs/torchtitan-dev/bin/python -m pytest tests/unit_tests/test_config_manager.py::TestConfigManager::test_glm5_config tests/unit_tests/test_glm5.py::TestGlm5Registration -v
```

Expected: FAIL because the registry modules do not exist.

- [ ] **Step 3: Implement explicit single-device validation**

In `sharding.py`, define:

```python
def validate_glm5_parallelism(
    parallelism: ParallelismConfig,
    parallel_dims: ParallelDims | None = None,
) -> None:
```

Collect enabled names from degrees greater than one. Treat data-parallel shard degree `-1` as unresolved and valid only until `parallel_dims` is available. Reject a provided `parallel_dims.world_size != 1`, reject `spmd_backend != "default"`, and include all offending names in one `NotImplementedError` message.

- [ ] **Step 4: Implement single-device parallelization hooks**

In `parallelize.py`, use the same keyword-only signature as other model parallelize functions. Call `validate_glm5_parallelism(parallelism, parallel_dims)`, then:

```python
if ac_config is not None:
    ac_config.build(dump_folder=dump_folder).apply(model)
if compile_config.enable and "model" in compile_config.components:
    apply_compile(model, compile_config)
return model
```

Do not call TP, CP, PP, EP, or FSDP helpers.

- [ ] **Step 5: Implement model and trainer registries**

Define `model_registry(flavor: str = "debugmodel") -> ModelSpec`:

```python
return ModelSpec(
    name="glm5",
    flavor=flavor,
    model=glm5_configs[flavor](),
    parallelize_fn=parallelize_glm5,
    pipelining_fn=None,
    post_optimizer_build_fn=register_moe_load_balancing_hook,
    state_dict_adapter=Glm5StateDictAdapter,
)
```

Add `"glm5"` to `_supported_models`. In `config_registry.py`, define `glm5_debugmodel()` with common cross-entropy loss, `./tests/assets/tokenizer`, `c4_test`, AdamW `8e-4`, 2 local batches, sequence length 128, 10 steps, metrics every step, checkpoint interval 10, and default single-device `ParallelismConfig`. Do not enable compile or activation checkpointing by default.

- [ ] **Step 6: Run registration and config-manager tests**

```bash
/home/h50064611/torchtitan/.miniconda3/envs/torchtitan-dev/bin/python -m pytest tests/unit_tests/test_config_manager.py tests/unit_tests/test_glm5.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit runtime wiring**

```bash
git add torchtitan/models/glm5 torchtitan/models/__init__.py tests/unit_tests/test_config_manager.py tests/unit_tests/test_glm5.py
git commit -m "feat: register GLM-5 debug training config"
```

---

### Task 6: Prove CPU forward, loss, and backward behavior

**Files:**
- Modify: `tests/unit_tests/test_glm5.py`

**Interfaces:**
- Consumes: the registered debug model and its native state dict.
- Produces: CPU acceptance coverage for dense/MoE composition, loss, backward, indexer gradient exclusion, and parameter/FLOP reporting.

- [ ] **Step 1: Add the full CPU smoke test**

```python
def test_debug_model_cpu_forward_loss_backward(self):
    torch.manual_seed(29)
    config = glm5_configs["debugmodel"]()
    model = config.build()
    model.init_states()
    model.train()
    tokens_BL = torch.randint(0, config.vocab_size, (2, 16))
    positions_BL = torch.arange(16).expand(2, -1)
    labels_BL = torch.randint(0, config.vocab_size, (2, 16))
    logits_BLV = model(tokens_BL, positions=positions_BL)
    loss = F.cross_entropy(
        logits_BLV.float().flatten(0, 1), labels_BL.flatten()
    )
    loss.backward()

    self.assertTrue(torch.isfinite(loss))
    self.assertIsNotNone(model.tok_embeddings.weight.grad)
    self.assertIsNotNone(model.layers["0"].feed_forward.w1.weight.grad)
    self.assertIsNotNone(
        model.layers["1"].moe.routed_experts.inner_experts.w1_EFD.grad
    )
    self.assertTrue(
        all(
            parameter.grad is None
            for layer in model.layers.values()
            for parameter in layer.attention.indexer.parameters()
        )
    )
```

Add one test that calls `config.get_nparams_and_flops(model, seq_len=16)` and asserts both returned integers are positive and the parameter count equals `sum(p.numel() for p in model.parameters())`.

- [ ] **Step 2: Run the new test and record the first real failure**

```bash
/home/h50064611/torchtitan/.miniconda3/envs/torchtitan-dev/bin/python -m pytest tests/unit_tests/test_glm5.py::TestGlm5Model::test_debug_model_cpu_forward_loss_backward -v
```

Expected before any correction: either PASS or one concrete model integration failure. If it fails, use `superpowers:systematic-debugging`, fix only the demonstrated root cause, and keep the test unchanged.

- [ ] **Step 3: Run all focused CPU tests**

```bash
/home/h50064611/torchtitan/.miniconda3/envs/torchtitan-dev/bin/python -m pytest tests/unit_tests/test_glm5.py tests/unit_tests/test_config_manager.py -v
```

Expected: PASS with no CUDA requirement.

- [ ] **Step 4: Commit CPU acceptance coverage and any demonstrated fix**

```bash
git add tests/unit_tests/test_glm5.py torchtitan/models/glm5
git commit -m "test: cover GLM-5 CPU training step"
```

---

### Task 7: Add Transformers fixtures and FP32 component parity

**Files:**
- Create: `tests/unit_tests/test_glm5_parity.py`
- Modify: `torchtitan/models/glm5/model.py` only if the parity test demonstrates a model mismatch.

**Interfaces:**
- Consumes: Transformers `GlmMoeDsaConfig`/`GlmMoeDsaForCausalLM`, `Glm5StateDictAdapter`, and one CUDA device.
- Produces: exact top-k/router selection parity and FP32 indexer/attention output parity.

- [ ] **Step 1: Create matching HF and TorchTitan fixture builders**

At module import, use normal imports inside a `try` and retain the import exception. In `setUpClass`, raise `unittest.SkipTest` when Transformers GLM-MoE-DSA is unavailable or CUDA is unavailable. Build this exact HF config:

```python
GlmMoeDsaConfig(
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
```

Build both models in FP32, convert HF state through `Glm5StateDictAdapter.from_hf()`, load with `strict=True`, move to CUDA, and set `eval()`.

- [ ] **Step 2: Write exact indexer and router parity tests**

Use seed 41, `B=2`, `L=16`, identical normalized hidden states and positions. Obtain HF position embeddings from `hf_model.model.rotary_emb`. Obtain `q_resid` from the matching HF and TorchTitan attention low-rank paths. Call each indexer directly with the same dense causal mask and assert `torch.equal()` for top-k indices.

For the first MoE layer, call HF `mlp.gate()` and TorchTitan `moe.router()`, reshape the HF results to `[B, L, K]`, and assert exact expert indices plus `torch.testing.assert_close()` on route weights at `rtol=1e-6, atol=1e-7`.

- [ ] **Step 3: Write the FP32 attention parity test**

Call layer 0's HF `self_attn()` and TorchTitan `attention()` with the same normalized hidden tensor, positions, and causal mask. Compare outputs with:

```python
torch.testing.assert_close(
    titan_output_BLD,
    hf_output_BLD,
    rtol=1e-4,
    atol=1e-5,
)
```

The HF attention returns a tuple; use its first item. Do not compare internal rotated Q/K layouts because the equivalent interleaved rotation may use a different orthonormal output ordering; compare top-k decisions and final attention outputs.

Add a dense-block parity test using layer 0. Pass the same unnormalized hidden
states, dense causal mask, positions, and HF position embeddings into the two
blocks. Compare the TorchTitan tensor against the first item of the HF block
tuple at `rtol=1e-4, atol=1e-5`. This covers both residual paths and the common
dense FFN without introducing BF16 grouped-expert arithmetic.

- [ ] **Step 4: Run component parity and verify failure before correction**

```bash
CUDA_VISIBLE_DEVICES=0 /home/h50064611/torchtitan/.miniconda3/envs/torchtitan-dev/bin/python -m pytest tests/unit_tests/test_glm5_parity.py -k "indexer or router or attention" -v
```

Expected: the first run either passes or exposes a concrete formula/dtype/mask mismatch. For any mismatch, use `superpowers:systematic-debugging`, compare the first diverging intermediate, add a regression assertion, and make only the minimal model correction.

- [ ] **Step 5: Re-run component parity and CPU regression tests**

```bash
CUDA_VISIBLE_DEVICES=0 /home/h50064611/torchtitan/.miniconda3/envs/torchtitan-dev/bin/python -m pytest tests/unit_tests/test_glm5_parity.py -k "indexer or router or attention" -v
/home/h50064611/torchtitan/.miniconda3/envs/torchtitan-dev/bin/python -m pytest tests/unit_tests/test_glm5.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit component parity**

```bash
git add tests/unit_tests/test_glm5_parity.py torchtitan/models/glm5/model.py
git commit -m "test: align GLM-5 components with Transformers"
```

---

### Task 8: Prove one-CUDA BF16 end-to-end output and gradient parity

**Files:**
- Modify: `tests/unit_tests/test_glm5_parity.py`
- Modify: GLM-5 source only when a failing parity assertion demonstrates the need.

**Interfaces:**
- Consumes: matching FP32 models and the formal adapter from Task 7.
- Produces: BF16 logits/loss/representative-gradient parity and expected indexer gradient absence.

- [ ] **Step 1: Add a BF16 model conversion helper**

Convert both models with `.to(device="cuda", dtype=torch.bfloat16)`. Assert every HF and TorchTitan `indexer.weights_proj.weight` remains FP32 after conversion. Use `B=1`, `L=16`, seed 53, identical tokens and `positions = arange(L)`.

- [ ] **Step 2: Add end-to-end logits and causal loss parity**

Run HF with `use_cache=False` and TorchTitan with its dense mask path. Compare logits at `rtol=5e-2, atol=5e-2`. Compute causal language-model loss consistently:

```python
labels_BL = tokens_BL.clone()
hf_loss = hf_model(
    input_ids=tokens_BL,
    position_ids=positions_BL,
    labels=labels_BL,
    use_cache=False,
).loss
titan_logits_BLV = titan_model(tokens_BL, positions=positions_BL)
titan_loss = F.cross_entropy(
    titan_logits_BLV[:, :-1].float().reshape(-1, 2048),
    labels_BL[:, 1:].reshape(-1),
)
torch.testing.assert_close(titan_loss, hf_loss, rtol=5e-2, atol=5e-2)
```

- [ ] **Step 3: Add representative gradient mappings**

Backpropagate both losses and compare:

- `model.embed_tokens.weight` against `tok_embeddings.weight`;
- layer 0 `q_a_proj.weight` against `attention.wq_a.weight`;
- layer 0 dense `gate_proj.weight` against `feed_forward.w1.weight`;
- layer 1 router `gate.weight` against `moe.router.gate.weight`;
- layer 1 HF fused expert `gate_up_proj.grad[:, :F]` against Titan `w1_EFD.grad`;
- layer 1 HF fused expert `gate_up_proj.grad[:, F:]` against Titan `w3_EFD.grad`;
- layer 1 `down_proj.grad` against Titan `w2_EDF.grad`;
- `lm_head.weight` against `lm_head.weight`.

Use `rtol=5e-2, atol=5e-2`. Assert every indexer parameter has `grad is None` in both models.

Before backward, also compare layer 1's MoE block output in BF16 at
`rtol=5e-2, atol=5e-2`, using the same hidden states, masks, positions, and HF
position embeddings. This supplies block-level coverage for the routed and
shared expert path.

- [ ] **Step 4: Run the full CUDA parity file**

```bash
CUDA_VISIBLE_DEVICES=0 /home/h50064611/torchtitan/.miniconda3/envs/torchtitan-dev/bin/python -m pytest tests/unit_tests/test_glm5_parity.py -v
```

Expected: PASS on one CUDA GPU, or SKIP with a precise reason when CUDA/Transformers is unavailable. A numerical assertion failure is not converted into a skip.

- [ ] **Step 5: Run CPU regressions after any numerical correction**

```bash
/home/h50064611/torchtitan/.miniconda3/envs/torchtitan-dev/bin/python -m pytest tests/unit_tests/test_glm5.py tests/unit_tests/test_config_manager.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit full parity coverage**

```bash
git add tests/unit_tests/test_glm5_parity.py torchtitan/models/glm5
git commit -m "test: verify GLM-5 single-GPU parity"
```

---

### Task 9: Document the model and run final repository checks

**Files:**
- Create: `torchtitan/models/glm5/README.md`
- Modify: GLM-5/test files only for failures demonstrated by final checks.

**Interfaces:**
- Consumes: the completed debug model, CLI config, tests, and explicit unsupported boundaries.
- Produces: user-facing commands and fresh completion evidence.

- [ ] **Step 1: Write the GLM-5 README**

Document:

- `--module glm5 --config glm5_debugmodel` invocation;
- the exact reduced dimensions and structural fidelity;
- Transformers `glm_moe_dsa` as the numerical reference;
- CPU forward/loss/backward and one-CUDA parity commands;
- use of common FFN/MoE/RoPE/norm modules;
- DSA eager path's quadratic memory/time limitation;
- unsupported KV cache, incremental decode, Flash-MLA, shared indexers, MTP, indexer auxiliary loss, TP, CP, PP, EP, FSDP, and HSDP;
- the staged roadmap from the approved design.

- [ ] **Step 2: Run focused tests**

```bash
/home/h50064611/torchtitan/.miniconda3/envs/torchtitan-dev/bin/python -m pytest tests/unit_tests/test_glm5.py tests/unit_tests/test_glm5_parity.py tests/unit_tests/test_config_manager.py -v
```

Expected: all CPU tests PASS; CUDA parity PASS when CUDA is available or SKIP only for unavailable CUDA/Transformers.

- [ ] **Step 3: Run affected common-model tests**

```bash
/home/h50064611/torchtitan/.miniconda3/envs/torchtitan-dev/bin/python -m pytest tests/unit_tests/test_fused_qkv.py tests/unit_tests/test_fused_swiglu.py tests/unit_tests/test_state_dict_keys.py -v
```

Expected: PASS, with existing CUDA-only skips permitted.

- [ ] **Step 4: Run static and formatting checks**

```bash
git diff --check
env PRE_COMMIT_HOME=/home/h50064611/torchtitan/.cache/pre-commit /home/h50064611/torchtitan/.miniconda3/envs/torchtitan-dev/bin/pre-commit run --all-files
```

Expected: PASS. If a formatter changes files, inspect the diff and repeat the focused tests before committing.

- [ ] **Step 5: Attempt the full unit suite**

```bash
/home/h50064611/torchtitan/.miniconda3/envs/torchtitan-dev/bin/python -m pytest tests/unit_tests -v
```

Expected: PASS where the environment supplies required optional dependencies and devices. Record unrelated dependency, network, multi-GPU, or hardware limitations exactly; do not describe them as GLM-5 failures.

- [ ] **Step 6: Review acceptance criteria and diff**

Run:

```bash
git status --short
git diff --stat origin/main...HEAD
git diff --check origin/main...HEAD
git log --oneline origin/main..HEAD
```

Confirm every acceptance criterion in `docs/superpowers/specs/2026-08-04-glm5-debug-model-design.md` has a corresponding passing test or explicit unsupported-mode assertion.

- [ ] **Step 7: Commit documentation and final check adjustments**

```bash
git add torchtitan/models/glm5/README.md torchtitan/models/glm5 tests/unit_tests/test_glm5.py tests/unit_tests/test_glm5_parity.py tests/unit_tests/test_config_manager.py torchtitan/models/__init__.py
git commit -m "docs: document GLM-5 debug model"
```

- [ ] **Step 8: Use verification-before-completion**

Invoke `superpowers:verification-before-completion`, rerun the focused CPU tests, CUDA parity, `git diff --check`, and pre-commit from fresh commands, then report exact pass/skip/failure counts and the final commit range.
