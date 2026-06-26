# Ornith-1.0-397B Micro Draft Model Architecture

Target giant: [`deepreinforce-ai/Ornith-1.0-397B`](https://huggingface.co/deepreinforce-ai/Ornith-1.0-397B)

## Giant topology (raw clay repository)

Ornith-1.0-397B is a **397B-parameter MoE** built on **Qwen3.5-MoE** (`Qwen3_5MoeForConditionalGeneration`). Key text-backbone constants from `config.json`:

| Parameter | Value |
|---|---|
| Hidden size | 4096 |
| Layers | 60 |
| Experts per layer | **512** |
| Active experts / token | **10** |
| MoE intermediate | 1024 |
| Shared expert / layer | yes (1024 intermediate) |
| Attention pattern | 3× `linear_attention` + 1× `full_attention` (repeating) |
| Context | 262,144 tokens |
| Vocab | 248,320 |
| Dtype | BF16 |

**Sparsity:** only `10 / 512 ≈ 1.95%` of routed experts fire per token per layer. Across 60 layers that is up to **600 expert shard touches** per forward — but each shard is only ~24 MB (BF16), so the working set is bounded by *which* experts, not the full 30,720 expert tensors on disk.

**Hybrid attention** means shard topology is not uniform:
- Layers 3, 7, 11, … → `full_attention` shards (GQA, 32 Q heads / 2 KV heads)
- All other layers → `linear_attention` shards (gated delta / linear conv, smaller footprint)

**Always-on pieces** (every forward, every layer):
- Shared expert FFN
- Router gate
- Layer norm
- Attention block (linear or full)

## Why a naive draft model fails

A flat MLP mapping `hidden → ℝ^{60×512}` has output dimension **30,720**. With a 4096-d input, a single linear layer alone is ~126M parameters *per layer head*, and cannot encode cross-layer routing structure.

The micro draft model must:
1. Stay resident in **hundreds of MB**, not GB
2. Forecast **512-way expert competition** independently at 60 depths
3. Exploit **depth correlation** (coding/reasoning routes cluster by layer band)
4. Condition on **agentic context** — tool calls, `<think>` reasoning blocks, code tokens

## Proposed architecture: `OrnithDraftSelector`

```
                    ┌─────────────────────────────────────┐
  Giant hidden ────►│ Input fusion (4096 + 128 → 2048)   │
  Token history ───►│ HistoryEncoder (GRU, 8K buckets)   │
                    └──────────────┬──────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
     Per-layer trunk ×60   Cross-layer Transformer   Attn-load head
     + expert head (→512)  (2L, 4H, over depth)      (→60)
              │                    │
              └────────┬───────────┘
                       ▼
              expert_logits (60, 512)
                       │
                       ▼
              ReassemblyBlueprint
```

### Module breakdown

| Module | Role | Approx params |
|---|---|---|
| `_HistoryEncoder` | Bucketed token IDs → 128-d GRU context | ~1.1M |
| `input_proj` | Fuse giant pooled hidden + history | ~9M |
| `layer_embed` + `layer_trunk` | Per-depth conditioning (layer index embedding) | ~0.5M + 60× trunk |
| `expert_heads` | 60 independent `Linear(2048 → 512)` routers | ~60M |
| `_CrossLayerRefiner` | 2-layer TransformerEncoder over depth axis | ~25M |
| `super_bucket_head` | Coarse 64-bin pre-filter per layer (optional) | ~8M |
| `attn_load_head` | Predict attention-shard prefetch priority | ~0.1M |

**Total: ~150–220M parameters → ~300–440 MB FP32, ~150–220 MB BF16.**

Fits the design constraint: permanently resident, well under 1 GB, leaving the 32 GB budget for assembled subgraph pieces.

### Blueprint emission (`OrnithMicroDraftModel.select_and_plan`)

For each layer `L`:

1. **Infrastructure (always `high_priority`):**
   - `layer_L/linear_attention` or `layer_L/full_attention`
   - `layer_L/norm`, `layer_L/router`
   - `layer_L/shared_expert`

2. **Routed experts:**
   - `topk(expert_logits[L], k=10+ε)` → candidate set
   - `prob ≥ τ` → `high_priority_pieces` (async exact fetch)
   - else → `low_priority_pieces` (int8 approx stand-in)

3. **Global:** `embed_weight`, `final_norm_weight`, `lm_head_weight`

4. **Reasoning boost:** if token history indicates an open `<think>` block (Ornith is a reasoning model), logits are scaled ×1.15 — empirical prior that reasoning phases activate deeper, wider expert subsets.

### Training strategy

| Phase | Data | Loss |
|---|---|---|
| Offline warm-start | Logged `(hidden, RealPath, history)` from giant with `output_router_logits=True` hooks | BCE on 60×512 expert activation grid |
| Router distillation | Giant router softmax → temperature-matched KL (future) | KL + BCE |
| Online adapt | `(blueprint, realized_path)` each SWS step | BCE, AdamW lr=5e-4, grad clip 1.0 |

**Trace collection** runs the giant (or SWS Phase 1 on-demand path) with hooks on each `Qwen3_5MoeSparseMoeBlock` gate. The micro draft never sees full expert weights — only hidden states and router top-k indices.

### Shard budget interaction

Estimated per-forward working set (BF16, all unique experts):

```
60 layers × 10 routed experts × ~24 MB   ≈  14.4 GB
60 layers ×  1 shared expert  × ~24 MB   ≈   1.4 GB
60 layers × attention infra    (varies)   ≈   4–8 GB
embed + lm_head + norms                  ≈   2 GB
─────────────────────────────────────────────────────
Total                                    ≈  22–26 GB
```

Within a **32 GB** budget with prediction-driven eviction, leaving ~6–10 GB headroom for prefetch speculation and CUDA KV — viable on a single high-RAM workstation; aligns with Ornith's own 8×80 GB serving recipe for the *full* model.

## Integration with SWS

```python
from sws.ornith_config import OrnithMoEConfig
from sws.ornith_draft import OrnithMicroDraftModel, propose_architecture_summary
from sws.hf_integration import shard_ornith_model, ORNITH_MODEL_ID

print(propose_architecture_summary())

cfg = OrnithMoEConfig()
draft = OrnithMicroDraftModel(cfg, device="cuda")
# draft.bind_tokenizer(tokenizer)  # enables <think> phase detection

# After trace collection:
draft.train_offline_router_distillation(traces, epochs=5)
```

## Open design questions

1. **Super-bucket head** — 64 coarse bins × 8 experts/bin reduces false negatives on cold experts; needs offline cluster centroids from router log PCA.
2. **MTP head** — Ornith has `mtp_num_hidden_layers=1`; draft may need to predict *next-two-token* expert union for multi-token prediction passes.
3. **Vision shards** — multimodal `vision_config` present; text-only SWS v1 ignores vision encoder shards (always cold).
4. **CUDA stream overlap** — prefetch `high_priority` expert shards during linear-attention compute (cheaper than full-attn windows).