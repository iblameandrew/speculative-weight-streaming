
#  Speculative Weight Streaming (SWS)

<img width="1248" height="832" alt="image" src="https://github.com/user-attachments/assets/c34752bb-e39e-4d1d-b957-b378f49aa819" />


**Streaming a Giant Into a Small Room: A Mechanism for Running Massive LLMs in 32GB RAM**

**SWS** (Speculative Weight Streaming) is a novel technique inspired by speculative decoding that enables running extremely large language models (far exceeding available RAM) on modest hardware by treating RAM as a predictive cache rather than a fixed container.

## Overview

Modern frontier models, particularly Mixture-of-Experts (MoE) architectures, are sparse: only a small fraction of parameters are active for any given token. Traditional loading assumes the entire model must reside in memory, creating an artificial ceiling.

SWS reframes this problem by speculatively streaming and caching only the weights predicted to be needed, with a verify-and-correct mechanism to ensure correctness. The full model lives on fast storage (NVMe), while RAM (e.g., 32GB) holds a dynamic working set.

This approach draws directly from speculative decoding's draft-verify-fallback loop, but applied to model weights instead of tokens.

## Core Insight

> You don't fit the model into memory — you fit a good guess about the model into memory, and let a cheap verifier keep that guess honest.

## Architecture

SWS consists of four cooperating components:

### 1. The Predictor (Draft Stage)
- A small, always-resident model (hundreds of MB) that forecasts the activation footprint for upcoming forward passes.
- Predicts active layers, experts (in MoE), attention heads, and FFN blocks based on current hidden states and token history.
- Learns online from realized paths to improve over time.

### 2. The Materializer (Prefetch & Approximate)
- Prefetches high-confidence weight shards from disk asynchronously.
- For low-confidence or rare blocks, reconstructs cheap approximations (e.g., low-rank or heavily quantized stand-ins).

### 3. The Verifier (Safety Net)
- Checks if speculatively loaded weights matched the actual computation path.
- On hit: Proceed seamlessly.
- On miss: Fetch exact weights, recompute the layer, and update the cache.

### 4. The Eviction Policy (Cache Management)
- Maintains the working set within the RAM budget using prediction-driven priorities.
- Pins frequently used components (e.g., lower layers, hot experts); evicts low-probability shards.

## Why It Excels for MoE Models (e.g., GLM 5.2)

MoE models activate only a handful of experts per token. SWS exploits this sparsity by predicting router behavior one step ahead, keeping the active plus speculative buffer well within 32GB.

## Performance Economics

Success depends on predictor accuracy, model sparsity, and storage I/O bandwidth.

The governing principle is that the benefit from accurate predictions and saved bandwidth must outweigh the cost of occasional misses, which involve recomputation and disk fetches. Weight misses are more expensive than analogous token rejections in speculative decoding due to slower storage access, making high predictor accuracy essential.

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
python benchmarks/phase2_gate.py   # predictor + prefetch
python benchmarks/phase3_gate.py   # approximation + verifier
python benchmarks/phase4_gate.py   # online adaptation + eviction
```

### Package layout

```
sws/
  store.py          # NVMeWeightStore — safetensors shards, async fetch, approx reconstructions
  cache.py          # PredictiveCache — byte-budget working set, prediction-driven eviction
  predictor.py      # TinyFootprintPredictor — forecasts next-step expert activation
  streamer.py       # SpeculativeWeightStreamer — draft → materialize → verify → adapt loop
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
- Stronger predictor architectures (router distillation)
- int4/int2 reconstruction quality tuning
- Distributed multi-GPU shard placement

## Contributing

Contributions are welcome. Areas of focus include improving predictor accuracy, reducing verification latency, and extending support to new model families.

## References & Inspiration
- Speculative decoding techniques
- Mixture-of-Experts routing dynamics
- Predictive caching literature

---
