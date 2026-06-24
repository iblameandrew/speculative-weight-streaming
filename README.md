
#  Speculative Weight Streaming (SWS)

<img width="1248" height="832" alt="image" src="https://github.com/user-attachments/assets/c34752bb-e39e-4d1d-b957-b378f49aa819" />


**Streaming a Giant Into a Small Room: A Mechanism for Running Massive LLMs in 32GB RAM**

**SWS** (Speculative Weight Streaming) is a dynamic model-recomposition system for running frontier-scale MoE models (700B+ parameters) on severely memory-constrained hardware (e.g., 32 GB RAM). A permanently-resident **micro draft model** acts as an intelligent sculptor: it selects raw weight pieces from NVMe storage and reorganizes them into a compact, fully executable sub-model that fits in RAM.

## Overview

A giant MoE model such as GLM 5.2 is never loaded whole. Instead, fine-grained weight shards (per-expert FFN, per-layer attention) live on NVMe as **raw clay**. A micro draft model examines the current hidden state and token history, predicts which pieces the next forward pass needs, and emits a **reassembly blueprint**. A dynamic assembler materializes those pieces into an executable subgraph; a strict verifier guarantees output equivalence to the reference giant.

This lifts speculative decoding's draft → verify → fallback loop from the token domain into **weight selection and model reassembly**.

## Core Insight

> The giant model is never instantiated in RAM. A micro draft model selects the raw clay, reorganizes it into a temporary executable instance, and a verifier keeps that assembly honest.

## Architecture

Four cooperating components:

### 1. NVMeWeightStore (repository of raw clay)
- Fine-grained `safetensors` shards: per-expert FFN, per-layer attention, routers.
- `fetch(shard_id)` — async exact retrieval; `extract_pieces(selection)` — batch extraction for reassembly.
- Lazy mmap; the full giant is never materialized.

### 2. MicroDraftModel (intelligent selector & sculptor)
- Compact permanently-resident network (hundreds of MB) that outputs a **reassembly blueprint**: which experts/layers to activate, with confidence scores.
- Produces `high_priority_pieces` (exact fetch + immediate assembly) and `low_priority_pieces` (approximated stand-ins).
- Learns to anticipate router decisions; adapts online from `(blueprint, realized_path)` pairs.

### 3. DynamicAssembler + PredictiveCache (reassembly engine)
- Takes the blueprint, fetches selected pieces, and assembles an executable `nn.Module` subgraph in RAM.
- Enforces a strict byte budget (default 32 GB); prediction-driven eviction of lowest-future-utility pieces.
- Hot-pins frequently reused infrastructure (lower layers, routers, attention).

### 4. Verifier + Reassembly Loop (safety & correctness)
- Executes the assembled sub-model; captures the true activation path from the reference router.
- On mismatch: `fallback_reassemble` — fetch additional exact pieces, rebuild, recompute.
- Guarantees functional equivalence (within tolerance) at the output boundary.

## Why It Excels for MoE Models (e.g., GLM 5.2)

MoE models activate only a handful of experts per token. The micro draft model exploits this sparsity by selecting and composing the minimal piece set, keeping the reassembled subgraph well within 32 GB even when the full model has 700B+ parameters on disk.

## Performance Economics

```
(selection_accuracy × bandwidth_saved) > (miss/reassembly_rate × (recompute + fetch_penalty))
```

Weight selection misses are far more costly than token rejections in speculative decoding (NVMe ≪ RAM bandwidth). High micro-model forecasting accuracy is essential.

## Limitations & Caveats

- Best suited for sparse architectures such as MoE; significantly less effective for dense models.
- Requires fast NVMe storage for practical performance.
- Includes an initial predictor warm-up period before reaching peak efficiency.
- Verification overhead must be carefully minimized.

## Getting Started

### Prerequisites
- Python 3.10+
- PyTorch, Hugging Face `transformers`, `safetensors`
- Fast NVMe SSD (any local disk works for the proof-of-concept)
- `psutil` for RSS benchmarks

```bash
pip install -r requirements.txt
```

### Run verification gates

Each phase has an independent gate script. Run all sequentially:

```bash
python benchmarks/run_all_gates.py
```

Or individually:

```bash
python benchmarks/phase0_gate.py   # sharding + lazy store
python benchmarks/phase1_gate.py   # LRU cache + on-demand
python benchmarks/phase2_gate.py   # micro draft model + prefetch
python benchmarks/phase3_gate.py   # approximation + verifier
python benchmarks/phase4_gate.py   # online adaptation + eviction
```

### Package layout

```
sws/
  store.py          # NVMeWeightStore — raw clay repository, fetch, extract_pieces
  micro_draft.py    # MicroDraftModel — selector + reassembly blueprint planner
  assembler.py      # DynamicAssembler — materialize executable subgraph from blueprint
  cache.py          # PredictiveCache — byte-budget enforcer inside assembler
  verifier.py       # Verifier — detect_miss, functional equivalence check
  streamer.py       # SpeculativeWeightStreamer — select → reassemble → verify → adapt
  predictor.py      # Backward-compat alias for MicroDraftModel
  hf_integration.py # Hugging Face MoE sharding + lazy linear proxies
benchmarks/         # phase gates + metrics (RSS, miss rate, stalls, tokens/sec, fidelity)
tests/              # unit tests
```

### Synthetic vs real MoE models

Gates run on a **tiny synthetic MoE** (no multi-GB download) to prove correctness on any machine. For real hardware validation, shard a genuine MoE:

```python
from sws.hf_integration import shard_hf_model, recommended_test_models

shard_hf_model(recommended_test_models()[0], output_dir="shards/")
```

Recommended open-weight MoE models (smallest first):
- `trl-internal-testing/tiny-Mixtral-8x7B-Instruct-v0.1` (CI / smoke tests)
- `Qwen/Qwen1.5-MoE-A2.7B`
- `mistralai/Mixtral-8x7B-v0.1`

### Honest caveat

SWS only wins for **genuinely sparse MoE** models where a small fraction of experts fire per token. On a dense model where every parameter participates in every forward pass, there is nothing to speculate about — the approach collapses to ordinary on-demand loading with verifier overhead.

### Roadmap
- CUDA stream prefetch overlapping attention compute
- vLLM / llama.cpp integration
- Stronger micro draft architectures (router distillation)
- int4/int2 reconstruction quality tuning
- Distributed multi-GPU shard placement

## Contributing

Contributions are welcome. Areas of focus include improving predictor accuracy, reducing verification latency, and extending support to new model families.

## References & Inspiration
- Speculative decoding techniques
- Mixture-of-Experts routing dynamics
- Predictive caching literature

---
