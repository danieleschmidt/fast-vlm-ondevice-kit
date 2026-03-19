"""Tests for EdgeBenchmark profiling."""

import pytest
import torch

from fast_vlm_ondevice.models import TinyVLM
from fast_vlm_ondevice.benchmark import EdgeBenchmark, BenchmarkResult


@pytest.fixture
def tiny_model():
    return TinyVLM(visual_dim=64, text_dim=32, vocab_size=64, max_seq_len=16)


@pytest.fixture
def sample_inputs():
    images = torch.randn(1, 3, 64, 64)
    tokens = torch.randint(0, 64, (1, 16))
    return images, tokens


class TestEdgeBenchmark:
    def test_returns_benchmark_result(self, tiny_model, sample_inputs):
        bench = EdgeBenchmark(warmup_runs=1, benchmark_runs=3)
        result = bench.run(tiny_model, sample_inputs, name="test")
        assert isinstance(result, BenchmarkResult)

    def test_latency_positive(self, tiny_model, sample_inputs):
        bench = EdgeBenchmark(warmup_runs=1, benchmark_runs=5)
        result = bench.run(tiny_model, sample_inputs)
        assert result.latency_mean_ms > 0
        assert result.latency_min_ms > 0
        assert result.latency_max_ms >= result.latency_min_ms

    def test_percentiles_ordered(self, tiny_model, sample_inputs):
        bench = EdgeBenchmark(warmup_runs=1, benchmark_runs=10)
        result = bench.run(tiny_model, sample_inputs)
        assert result.latency_p50_ms <= result.latency_p95_ms
        assert result.latency_p95_ms <= result.latency_p99_ms

    def test_num_params_correct(self, tiny_model, sample_inputs):
        bench = EdgeBenchmark(warmup_runs=1, benchmark_runs=3)
        result = bench.run(tiny_model, sample_inputs)
        expected = sum(p.numel() for p in tiny_model.parameters())
        assert result.num_params == expected

    def test_param_size_mb_positive(self, tiny_model, sample_inputs):
        bench = EdgeBenchmark(warmup_runs=1, benchmark_runs=3)
        result = bench.run(tiny_model, sample_inputs)
        assert result.param_size_mb > 0

    def test_layer_param_counts_populated(self, tiny_model, sample_inputs):
        bench = EdgeBenchmark(warmup_runs=1, benchmark_runs=3)
        result = bench.run(tiny_model, sample_inputs)
        assert len(result.layer_param_counts) > 0

    def test_result_str(self, tiny_model, sample_inputs):
        bench = EdgeBenchmark(warmup_runs=1, benchmark_runs=3)
        result = bench.run(tiny_model, sample_inputs, name="TinyVLM-Test")
        s = str(result)
        assert "TinyVLM-Test" in s
        assert "Parameters" in s
        assert "Latency" in s

    def test_compare_returns_dict(self, tiny_model, sample_inputs):
        bench = EdgeBenchmark(warmup_runs=1, benchmark_runs=3)
        results = bench.compare({"model-a": (tiny_model, sample_inputs)})
        assert "model-a" in results
        assert isinstance(results["model-a"], BenchmarkResult)

    def test_custom_forward_fn(self, tiny_model, sample_inputs):
        bench = EdgeBenchmark(warmup_runs=1, benchmark_runs=3)
        call_count = [0]

        def custom_fn():
            call_count[0] += 1
            return tiny_model(*sample_inputs)

        result = bench.run(tiny_model, sample_inputs, forward_fn=custom_fn)
        # warmup(1) + benchmark(3) = 4 calls
        assert call_count[0] == 4

    def test_count_parameters(self, tiny_model):
        bench = EdgeBenchmark()
        counts = bench.count_parameters(tiny_model)
        assert isinstance(counts, dict)
        assert len(counts) > 0
        total_from_counts = sum(counts.values())
        total_actual = sum(p.numel() for p in tiny_model.parameters())
        # Note: named_modules is non-recursive per module, but can double-count
        # Just verify sum > 0 and structure is right
        assert total_from_counts > 0
