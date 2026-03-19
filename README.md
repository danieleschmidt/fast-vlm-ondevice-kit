# fast-vlm-ondevice

Lightweight Vision-Language Model (VLM) toolkit for edge deployment.

Efficient image-text processing designed for CPU-bound, memory-constrained environments — mobile, embedded, IoT, and on-device inference.

---

## What's in here

```
fast_vlm_ondevice/
├── models.py         # Core model components
├── quantization.py   # INT8 quantization simulation
└── benchmark.py      # Latency / memory / ops profiling
```

### `ImageEncoder`

3-layer ConvNet visual feature extractor.

- Conv2d (3→32) → BN → ReLU → MaxPool
- Conv2d (32→64) → BN → ReLU → MaxPool
- Conv2d (64→128) → BN → ReLU → AdaptiveAvgPool(4×4)
- Linear projection → fixed embedding

Outputs a fixed-size embedding regardless of input resolution. Also exposes `encode_as_tokens()` for spatial token output (16 tokens × D), used by cross-modal attention.

### `TextEncoder`

Small 2-layer transformer encoder for text queries.

- Token embedding + learned positional encoding
- 2× TransformerEncoderLayer (d_model=64, nhead=4, dim_feedforward=128)
- Mean pooling with optional padding mask

### `CrossModalAttention`

Cross-attention between visual and text modalities.

Visual tokens attend to text query tokens — image regions focus on relevant text concepts. Residual connection + LayerNorm for stability.

### `TinyVLM`

Full image-text matching model combining all components.

```
image → ImageEncoder → visual_tokens (B, 16, D_visual)
text  → TextEncoder  → text_tokens   (B, L, D_text)
CrossModalAttention(visual_tokens, text_tokens)
→ mean pool → concat → MLP → matching_score (B,)
```

Returns a scalar similarity score per (image, text) pair.

### `QuantizationSimulator`

Simulates INT8 symmetric per-tensor quantization.

- Clips and rounds weights to 8-bit precision
- Measures per-layer weight error
- Reports SNR (signal-to-noise ratio) of weight approximation
- `compare_outputs()` measures output accuracy degradation

Useful for estimating quantization sensitivity before real hardware deployment (CoreML, ONNX Runtime INT8, TFLite, etc.).

### `EdgeBenchmark`

Profiles model inference on CPU.

- Latency: mean, p50, p95, p99 (ms)
- Memory: process RSS delta (MB)
- Parameter count and size (MB, FP32)
- Per-layer parameter breakdown

---

## Quick start

```python
import torch
from fast_vlm_ondevice import TinyVLM, QuantizationSimulator, EdgeBenchmark

# Build model
model = TinyVLM(visual_dim=128, text_dim=64, vocab_size=256, max_seq_len=32)

# Run inference
images = torch.randn(4, 3, 64, 64)
tokens = torch.randint(0, 256, (4, 16))
scores = model(images, tokens)  # (4,) matching scores

# Simulate INT8 quantization
sim = QuantizationSimulator(bits=8)
q_model, report = sim.quantize(model)
print(report)

# Compare outputs
diff = sim.compare_outputs(model, q_model, (images, tokens))
print(f"Cosine similarity: {diff['cosine_similarity']:.6f}")

# Benchmark
bench = EdgeBenchmark(warmup_runs=5, benchmark_runs=30)
result = bench.run(model, (images[:1], tokens[:1]), name="TinyVLM-FP32")
print(result)
```

## Demo

```bash
python demo.py
```

Example output (315K params, CPU):
```
Model: TinyVLM | 315,553 parameters

QuantizationReport (FP32 → INT8)
  Parameters quantized: 315,105 / 315,105
  Mean weight error:    0.000784
  Weight SNR:           47.02 dB

BenchmarkResult: TinyVLM-FP32
  Parameters:      315,553 (1.20 MB FP32)
  Latency (CPU):   1.04 ms mean  [p50=0.86  p95=1.53  p99=1.60]

Output cosine similarity (FP32 vs INT8sim): 0.999997
```

---

## Install

```bash
pip install -e .
```

Requirements: Python ≥3.10, PyTorch ≥2.0, psutil.

---

## Tests

```bash
pytest tests/ -v
```

43 tests covering all components: shape/dtype correctness, gradient flow, quantization math, benchmark output structure.

---

## Design goals

- **Minimal dependencies**: PyTorch + psutil only
- **No pretrained weights**: architecture demo, easily replaced with real weights
- **CPU-first**: everything measurable without GPU
- **Composable**: use components independently or as a full pipeline
- **Quantization-aware**: built for INT8/INT4 transition to CoreML/ONNX/TFLite

---

## On-device deployment path

This toolkit is a research/prototyping harness. For production deployment:

1. Train TinyVLM (or substitute a real VLM backbone)
2. Assess quantization sensitivity with `QuantizationSimulator`
3. Export to target runtime:
   - **Apple Neural Engine**: CoreML via `coremltools`
   - **Android**: TFLite or ONNX Runtime Mobile
   - **General edge**: ONNX export + ORT
4. Benchmark on device with `EdgeBenchmark` pattern

---

## License

MIT
