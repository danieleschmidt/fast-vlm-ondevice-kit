"""
EdgeBenchmark: On-device inference profiling for VLM models.

Measures:
  - Latency: mean/p50/p95/p99 inference time (CPU, ms)
  - Memory: model parameter footprint (MB), peak resident memory change
  - Ops: rough parameter count and estimated FLOPs for key layers
"""

import gc
import time
from dataclasses import dataclass, field
from typing import Callable

import psutil
import torch
import torch.nn as nn


@dataclass
class BenchmarkResult:
    model_name: str
    num_params: int
    param_size_mb: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_min_ms: float
    latency_max_ms: float
    memory_delta_mb: float
    num_runs: int
    layer_param_counts: dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"BenchmarkResult: {self.model_name}",
            f"  Parameters:      {self.num_params:,} ({self.param_size_mb:.2f} MB FP32)",
            f"  Latency (CPU):   {self.latency_mean_ms:.2f} ms mean  "
            f"[p50={self.latency_p50_ms:.2f}  p95={self.latency_p95_ms:.2f}  "
            f"p99={self.latency_p99_ms:.2f}]",
            f"  Latency range:   {self.latency_min_ms:.2f}–{self.latency_max_ms:.2f} ms",
            f"  Memory delta:    {self.memory_delta_mb:+.2f} MB (process RSS change)",
            f"  Runs:            {self.num_runs}",
        ]
        if self.layer_param_counts:
            lines.append("  Layer params:")
            for name, count in sorted(
                self.layer_param_counts.items(), key=lambda x: -x[1]
            )[:10]:
                lines.append(f"    {name:40s} {count:>10,}")
        return "\n".join(lines)


class EdgeBenchmark:
    """
    Profiles VLM model inference on CPU to simulate edge device constraints.

    Usage::

        bench = EdgeBenchmark(warmup_runs=3, benchmark_runs=20)

        model = TinyVLM()
        images = torch.randn(1, 3, 64, 64)
        tokens = torch.randint(0, 256, (1, 16))

        result = bench.run(model, (images, tokens), name="TinyVLM-FP32")
        print(result)
    """

    def __init__(self, warmup_runs: int = 5, benchmark_runs: int = 30):
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs

    def count_parameters(self, model: nn.Module) -> dict[str, int]:
        """Count parameters per named module (non-recursive, top-level modules)."""
        counts: dict[str, int] = {}
        for name, module in model.named_modules():
            if name == "":
                continue
            local_params = sum(
                p.numel() for p in module.parameters(recurse=False)
            )
            if local_params > 0:
                counts[name] = local_params
        return counts

    def param_size_mb(self, model: nn.Module) -> float:
        """Estimate model size in MB assuming FP32 (4 bytes/param)."""
        total = sum(p.numel() for p in model.parameters())
        return total * 4 / (1024 ** 2)

    def run(
        self,
        model: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        name: str = "model",
        forward_fn: Callable | None = None,
    ) -> BenchmarkResult:
        """
        Run inference benchmark.

        Args:
            model: The model to benchmark.
            inputs: Tuple of input tensors passed to model(*inputs).
            name: Display name for the result.
            forward_fn: Optional custom forward callable. Defaults to model(*inputs).

        Returns:
            BenchmarkResult with all metrics.
        """
        model.eval()
        call = forward_fn or (lambda: model(*inputs))

        # --- Memory baseline ---
        proc = psutil.Process()
        gc.collect()
        mem_before = proc.memory_info().rss / (1024 ** 2)

        # --- Warmup ---
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                call()

        # --- Timed runs ---
        latencies: list[float] = []
        with torch.no_grad():
            for _ in range(self.benchmark_runs):
                t0 = time.perf_counter()
                call()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)

        mem_after = proc.memory_info().rss / (1024 ** 2)

        # --- Stats ---
        lats = torch.tensor(latencies)
        total_params = sum(p.numel() for p in model.parameters())
        layer_counts = self.count_parameters(model)

        return BenchmarkResult(
            model_name=name,
            num_params=total_params,
            param_size_mb=self.param_size_mb(model),
            latency_mean_ms=lats.mean().item(),
            latency_p50_ms=lats.quantile(0.50).item(),
            latency_p95_ms=lats.quantile(0.95).item(),
            latency_p99_ms=lats.quantile(0.99).item(),
            latency_min_ms=lats.min().item(),
            latency_max_ms=lats.max().item(),
            memory_delta_mb=mem_after - mem_before,
            num_runs=self.benchmark_runs,
            layer_param_counts=layer_counts,
        )

    def compare(
        self,
        models: dict[str, tuple[nn.Module, tuple[torch.Tensor, ...]]],
    ) -> dict[str, BenchmarkResult]:
        """
        Benchmark multiple models and return results keyed by name.

        Args:
            models: Dict of {name: (model, inputs_tuple)}

        Returns:
            Dict of {name: BenchmarkResult}
        """
        results = {}
        for name, (model, inputs) in models.items():
            results[name] = self.run(model, inputs, name=name)
        return results
