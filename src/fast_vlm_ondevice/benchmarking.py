"""
Performance benchmarking utilities for FastVLM models.

Provides comprehensive performance measurement and analysis.
"""

import time
import threading
import statistics
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from pathlib import Path

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    import coremltools as ct
    import numpy as np
    from PIL import Image
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


@dataclass
class BenchmarkMetrics:
    """Container for benchmark metrics."""
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_fps: float
    peak_memory_mb: float
    avg_memory_mb: float
    total_time_s: float
    iterations: int
    warmup_iterations: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "throughput_fps": round(self.throughput_fps, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "avg_memory_mb": round(self.avg_memory_mb, 2),
            "total_time_s": round(self.total_time_s, 2),
            "iterations": self.iterations,
            "warmup_iterations": self.warmup_iterations
        }


class MemoryMonitor:
    """Memory usage monitor for benchmarking."""
    
    def __init__(self):
        self.monitoring = False
        self.memory_samples = []
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start memory monitoring."""
        self.monitoring = True
        self.memory_samples = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            
    def _monitor_loop(self):
        """Memory monitoring loop."""
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
        else:
            process = None
        while self.monitoring:
            try:
                if PSUTIL_AVAILABLE and process:
                    memory_mb = process.memory_info().rss / (1024**2)
                    self.memory_samples.append(memory_mb)
                else:
                    # Fallback: estimate memory usage
                    memory_mb = 100.0  # Default estimate
                    self.memory_samples.append(memory_mb)
                time.sleep(0.01)  # Sample every 10ms
            except:
                break
                
    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        return max(self.memory_samples) if self.memory_samples else 0.0
        
    def get_avg_memory(self) -> float:
        """Get average memory usage in MB."""
        return statistics.mean(self.memory_samples) if self.memory_samples else 0.0


class PerformanceBenchmark:
    """Performance benchmarking for FastVLM models."""
    
    def __init__(self, model_path: str):
        """Initialize benchmark.
        
        Args:
            model_path: Path to model file
        """
        self.model_path = Path(model_path)
        self.model = None
        self.model_type = self._detect_model_type()
        self.memory_monitor = MemoryMonitor()
        
    def _detect_model_type(self) -> str:
        """Detect model type from file extension."""
        if self.model_path.suffix == '.pth':
            return "pytorch"
        elif self.model_path.name.endswith('.mlpackage'):
            return "coreml"
        else:
            return "unknown"
    
    def run_benchmark(
        self,
        iterations: int = 100,
        warmup: int = 10,
        batch_sizes: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Run comprehensive performance benchmark.
        
        Args:
            iterations: Number of benchmark iterations
            warmup: Number of warmup iterations
            batch_sizes: List of batch sizes to test (if supported)
            
        Returns:
            Benchmark results dictionary
        """
        if not DEPS_AVAILABLE:
            return {
                "error": "Required dependencies not available",
                "success": False
            }
        
        results = {
            "success": False,
            "model_path": str(self.model_path),
            "model_type": self.model_type,
            "benchmarks": {}
        }
        
        try:
            # Load model
            self._load_model()
            
            # Single inference benchmark
            single_results = self._benchmark_single_inference(iterations, warmup)
            results["benchmarks"]["single_inference"] = single_results
            
            # Batch inference benchmark (if supported)
            if batch_sizes and self.model_type == "pytorch":
                batch_results = {}
                for batch_size in batch_sizes:
                    batch_result = self._benchmark_batch_inference(
                        batch_size, iterations // 2, warmup // 2
                    )
                    batch_results[f"batch_{batch_size}"] = batch_result
                results["benchmarks"]["batch_inference"] = batch_results
            
            # Memory stress test
            memory_results = self._benchmark_memory_usage(50, 5)
            results["benchmarks"]["memory_stress"] = memory_results
            
            # Cold start benchmark
            cold_start_results = self._benchmark_cold_start(10)
            results["benchmarks"]["cold_start"] = cold_start_results
            
            results["success"] = True
            
            # Generate summary
            main_metrics = single_results
            results["summary"] = {
                "avg_latency_ms": main_metrics["avg_latency_ms"],
                "p95_latency_ms": main_metrics["p95_latency_ms"],
                "throughput_fps": main_metrics["throughput_fps"],
                "peak_memory_mb": memory_results["peak_memory_mb"],
                "cold_start_ms": cold_start_results["avg_latency_ms"]
            }
            
            return results
            
        except Exception as e:
            results["error"] = str(e)
            return results
    
    def _load_model(self):
        """Load model for benchmarking."""
        if self.model_type == "pytorch":
            self.model = torch.load(self.model_path, map_location='cpu')
            if hasattr(self.model, 'eval'):
                self.model.eval()
                
        elif self.model_type == "coreml":
            self.model = ct.models.MLModel(str(self.model_path))
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _benchmark_single_inference(
        self,
        iterations: int,
        warmup: int
    ) -> Dict[str, Any]:
        """Benchmark single inference performance."""
        latencies = []
        
        # Start memory monitoring
        self.memory_monitor.start_monitoring()
        
        total_start = time.time()
        
        try:
            # Warmup iterations
            for _ in range(warmup):
                self._run_inference()
            
            # Benchmark iterations
            for _ in range(iterations):
                start_time = time.time()
                self._run_inference()
                latency_ms = (time.time() - start_time) * 1000
                latencies.append(latency_ms)
            
            total_time = time.time() - total_start
            
        finally:
            self.memory_monitor.stop_monitoring()
        
        # Calculate metrics
        metrics = BenchmarkMetrics(
            avg_latency_ms=statistics.mean(latencies),
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            p50_latency_ms=statistics.median(latencies),
            p95_latency_ms=np.percentile(latencies, 95),
            p99_latency_ms=np.percentile(latencies, 99),
            throughput_fps=iterations / (total_time - warmup * statistics.mean(latencies[:warmup]) / 1000),
            peak_memory_mb=self.memory_monitor.get_peak_memory(),
            avg_memory_mb=self.memory_monitor.get_avg_memory(),
            total_time_s=total_time,
            iterations=iterations,
            warmup_iterations=warmup
        )
        
        return metrics.to_dict()
    
    def _benchmark_batch_inference(
        self,
        batch_size: int,
        iterations: int,
        warmup: int
    ) -> Dict[str, Any]:
        """Benchmark batch inference performance."""
        latencies = []
        
        self.memory_monitor.start_monitoring()
        total_start = time.time()
        
        try:
            # Warmup
            for _ in range(warmup):
                self._run_batch_inference(batch_size)
            
            # Benchmark
            for _ in range(iterations):
                start_time = time.time()
                self._run_batch_inference(batch_size)
                latency_ms = (time.time() - start_time) * 1000
                latencies.append(latency_ms)
            
            total_time = time.time() - total_start
            
        finally:
            self.memory_monitor.stop_monitoring()
        
        # Calculate per-sample metrics
        per_sample_latencies = [lat / batch_size for lat in latencies]
        
        metrics = BenchmarkMetrics(
            avg_latency_ms=statistics.mean(per_sample_latencies),
            min_latency_ms=min(per_sample_latencies),
            max_latency_ms=max(per_sample_latencies),
            p50_latency_ms=statistics.median(per_sample_latencies),
            p95_latency_ms=np.percentile(per_sample_latencies, 95),
            p99_latency_ms=np.percentile(per_sample_latencies, 99),
            throughput_fps=(iterations * batch_size) / total_time,
            peak_memory_mb=self.memory_monitor.get_peak_memory(),
            avg_memory_mb=self.memory_monitor.get_avg_memory(),
            total_time_s=total_time,
            iterations=iterations,
            warmup_iterations=warmup
        )
        
        result = metrics.to_dict()
        result["batch_size"] = batch_size
        result["batch_latency_ms"] = statistics.mean(latencies)
        return result
    
    def _benchmark_memory_usage(
        self,
        iterations: int,
        warmup: int
    ) -> Dict[str, Any]:
        """Benchmark memory usage patterns."""
        memory_samples = []
        
        for i in range(warmup + iterations):
            self.memory_monitor.start_monitoring()
            self._run_inference()
            self.memory_monitor.stop_monitoring()
            
            if i >= warmup:  # Skip warmup iterations
                peak_mem = self.memory_monitor.get_peak_memory()
                memory_samples.append(peak_mem)
        
        return {
            "peak_memory_mb": max(memory_samples),
            "avg_memory_mb": statistics.mean(memory_samples),
            "min_memory_mb": min(memory_samples),
            "memory_std_mb": statistics.stdev(memory_samples) if len(memory_samples) > 1 else 0.0,
            "iterations": iterations
        }
    
    def _benchmark_cold_start(self, iterations: int) -> Dict[str, Any]:
        """Benchmark cold start performance."""
        latencies = []
        
        for _ in range(iterations):
            # Reload model to simulate cold start
            self._load_model()
            
            start_time = time.time()
            self._run_inference()
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)
        
        return {
            "avg_latency_ms": statistics.mean(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "iterations": iterations
        }
    
    def _run_inference(self):
        """Run single model inference."""
        if self.model_type == "pytorch":
            dummy_image = torch.randn(1, 3, 336, 336)
            dummy_input_ids = torch.randint(0, 30522, (1, 77))
            dummy_attention_mask = torch.ones(1, 77)
            
            with torch.no_grad():
                if hasattr(self.model, 'forward'):
                    _ = self.model(dummy_image, dummy_input_ids, dummy_attention_mask)
                    
        elif self.model_type == "coreml":
            # Create dummy inputs based on model spec
            spec = self.model.get_spec()
            input_dict = {}
            
            for input_desc in spec.description.input:
                if input_desc.type.HasField('imageType'):
                    width = input_desc.type.imageType.width
                    height = input_desc.type.imageType.height
                    dummy_image = Image.new('RGB', (width, height), color='red')
                    input_dict[input_desc.name] = dummy_image
                    
                elif input_desc.type.HasField('multiArrayType'):
                    shape = list(input_desc.type.multiArrayType.shape)
                    dummy_array = np.random.randn(*shape).astype(np.float32)
                    input_dict[input_desc.name] = dummy_array
            
            _ = self.model.predict(input_dict)
    
    def _run_batch_inference(self, batch_size: int):
        """Run batch model inference (PyTorch only)."""
        if self.model_type == "pytorch":
            dummy_image = torch.randn(batch_size, 3, 336, 336)
            dummy_input_ids = torch.randint(0, 30522, (batch_size, 77))
            dummy_attention_mask = torch.ones(batch_size, 77)
            
            with torch.no_grad():
                if hasattr(self.model, 'forward'):
                    _ = self.model(dummy_image, dummy_input_ids, dummy_attention_mask)


def compare_models(model_paths: List[str], iterations: int = 50) -> Dict[str, Any]:
    """Compare performance of multiple models.
    
    Args:
        model_paths: List of model file paths
        iterations: Number of benchmark iterations per model
        
    Returns:
        Comparison results
    """
    results = {
        "models": {},
        "comparison": {}
    }
    
    # Benchmark each model
    for model_path in model_paths:
        try:
            benchmark = PerformanceBenchmark(model_path)
            model_results = benchmark.run_benchmark(iterations=iterations, warmup=10)
            model_name = Path(model_path).stem
            results["models"][model_name] = model_results
        except Exception as e:
            results["models"][Path(model_path).stem] = {"error": str(e)}
    
    # Generate comparison
    valid_models = {
        name: result for name, result in results["models"].items()
        if result.get("success", False)
    }
    
    if len(valid_models) >= 2:
        # Find best performing model for each metric
        metrics = ["avg_latency_ms", "p95_latency_ms", "throughput_fps", "peak_memory_mb"]
        
        for metric in metrics:
            model_values = {}
            for model_name, model_result in valid_models.items():
                if "summary" in model_result and metric in model_result["summary"]:
                    model_values[model_name] = model_result["summary"][metric]
            
            if model_values:
                if metric in ["avg_latency_ms", "p95_latency_ms", "peak_memory_mb"]:
                    # Lower is better
                    best_model = min(model_values, key=model_values.get)
                else:
                    # Higher is better
                    best_model = max(model_values, key=model_values.get)
                
                results["comparison"][f"best_{metric}"] = {
                    "model": best_model,
                    "value": model_values[best_model],
                    "all_values": model_values
                }
    
    return results