#!/usr/bin/env python3
"""
fast-vlm-ondevice demo: TinyVLM on synthetic image-text pairs.

Demonstrates:
  - FP32 forward pass on synthetic (image, text) pairs
  - INT8 quantization simulation
  - EdgeBenchmark: latency/memory comparison FP32 vs INT8
"""

import torch

from fast_vlm_ondevice import (
    TinyVLM,
    QuantizationSimulator,
    EdgeBenchmark,
)


def make_synthetic_batch(batch_size: int = 4):
    """Generate random synthetic image-text pairs."""
    images = torch.randn(batch_size, 3, 64, 64)
    token_ids = torch.randint(0, 256, (batch_size, 16))
    return images, token_ids


def run_matching_demo(model: torch.nn.Module, label: str):
    """Run TinyVLM on a batch and print matching scores."""
    images, tokens = make_synthetic_batch(batch_size=4)
    model.eval()
    with torch.no_grad():
        scores = model(images, tokens)
    print(f"\n{label} — matching scores (4 image-text pairs):")
    for i, s in enumerate(scores):
        print(f"  pair {i}: {s.item():+.4f}")


def main():
    print("=" * 60)
    print(" fast-vlm-ondevice: TinyVLM Demo")
    print("=" * 60)

    # --- Build model ---
    model_fp32 = TinyVLM(visual_dim=128, text_dim=64, vocab_size=256, max_seq_len=32)
    total_params = sum(p.numel() for p in model_fp32.parameters())
    print(f"\nModel: TinyVLM | {total_params:,} parameters")

    # --- FP32 forward pass ---
    run_matching_demo(model_fp32, "FP32")

    # --- INT8 Quantization ---
    print("\n" + "-" * 40)
    print("Simulating INT8 quantization...")
    sim = QuantizationSimulator(bits=8)
    model_int8, report = sim.quantize(model_fp32)
    print(report)

    # --- Compare outputs FP32 vs INT8 ---
    images, tokens = make_synthetic_batch(batch_size=8)
    diff_stats = sim.compare_outputs(model_fp32, model_int8, (images, tokens))
    print("\nOutput diff (FP32 vs INT8):")
    for k, v in diff_stats.items():
        print(f"  {k}: {v:.6f}")

    # --- EdgeBenchmark ---
    print("\n" + "-" * 40)
    print("Running EdgeBenchmark (CPU)...")
    bench = EdgeBenchmark(warmup_runs=5, benchmark_runs=30)

    images_bench, tokens_bench = make_synthetic_batch(batch_size=1)

    results = bench.compare({
        "TinyVLM-FP32": (model_fp32, (images_bench, tokens_bench)),
        "TinyVLM-INT8sim": (model_int8, (images_bench, tokens_bench)),
    })

    for name, result in results.items():
        print(f"\n{result}")

    # --- Summary ---
    fp32_lat = results["TinyVLM-FP32"].latency_mean_ms
    int8_lat = results["TinyVLM-INT8sim"].latency_mean_ms
    speedup = fp32_lat / int8_lat if int8_lat > 0 else 0

    print("\n" + "=" * 60)
    print(" Summary")
    print("=" * 60)
    print(f"  FP32 latency:    {fp32_lat:.2f} ms")
    print(f"  INT8sim latency: {int8_lat:.2f} ms")
    print(f"  Speedup:         {speedup:.2f}x")
    print(f"  Mean output diff:{diff_stats['mean_abs_diff']:.6f}")
    print(f"  Output cosine:   {diff_stats['cosine_similarity']:.6f}")
    print("\nNote: INT8 speedup on CPU depends on backend BLAS support.")
    print("Simulated quantization shows accuracy cost; real INT8 requires")
    print("hardware-level ops (CoreML, ONNX Runtime, TFLite, etc.)")


if __name__ == "__main__":
    main()
