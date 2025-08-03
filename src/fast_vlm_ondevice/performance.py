"""
Performance monitoring and optimization utilities for FastVLM models.

Handles profiling, benchmarking, and performance optimization.
"""

import logging
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import statistics
import json
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for model inference."""
    
    latency_ms: float
    memory_peak_mb: float
    memory_current_mb: float
    cpu_percent: float
    inference_count: int
    throughput_fps: float
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "latency_ms": self.latency_ms,
            "memory_peak_mb": self.memory_peak_mb,
            "memory_current_mb": self.memory_current_mb,
            "cpu_percent": self.cpu_percent,
            "inference_count": self.inference_count,
            "throughput_fps": self.throughput_fps,
            "error_rate": self.error_rate,
            "cache_hit_rate": self.cache_hit_rate
        }


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark."""
    
    model_name: str
    test_name: str
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput_fps: float
    memory_usage_mb: float
    error_count: int
    total_runs: int
    test_duration_s: float
    timestamp: str
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return 1.0 - (self.error_count / self.total_runs) if self.total_runs > 0 else 0.0


class PerformanceProfiler:
    """Real-time performance profiling for FastVLM inference."""
    
    def __init__(self, window_size: int = 100):
        """Initialize profiler.
        
        Args:
            window_size: Number of recent measurements to keep
        """
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.error_count = 0
        self.total_inferences = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        self._monitoring = False
        self._monitor_thread = None
        self._system_metrics = {}
    
    @contextmanager
    def profile_inference(self):
        """Context manager for profiling a single inference."""
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        try:
            yield
            success = True
        except Exception as e:
            self.error_count += 1
            success = False
            raise
        finally:
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            
            latency_ms = (end_time - start_time) * 1000
            
            self.latencies.append(latency_ms)
            self.memory_usage.append(end_memory)
            self.timestamps.append(time.time())
            self.total_inferences += 1
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self.cache_misses += 1
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        if not self.latencies:
            return PerformanceMetrics(
                latency_ms=0.0,
                memory_peak_mb=0.0,
                memory_current_mb=0.0,
                cpu_percent=0.0,
                inference_count=0,
                throughput_fps=0.0
            )
        
        current_memory = self._get_memory_usage()
        peak_memory = max(self.memory_usage) if self.memory_usage else current_memory
        avg_latency = statistics.mean(self.latencies)
        
        # Calculate throughput over the last second
        recent_timestamps = [t for t in self.timestamps if time.time() - t < 1.0]
        throughput = len(recent_timestamps)
        
        # Calculate error rate
        error_rate = self.error_count / self.total_inferences if self.total_inferences > 0 else 0.0
        
        # Calculate cache hit rate
        total_cache_ops = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_cache_ops if total_cache_ops > 0 else 0.0
        
        return PerformanceMetrics(
            latency_ms=avg_latency,
            memory_peak_mb=peak_memory,
            memory_current_mb=current_memory,
            cpu_percent=psutil.cpu_percent(),
            inference_count=self.total_inferences,
            throughput_fps=throughput,
            error_rate=error_rate,
            cache_hit_rate=cache_hit_rate
        )
    
    def get_latency_percentiles(self) -> Dict[str, float]:
        """Get latency percentiles."""
        if not self.latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        
        sorted_latencies = sorted(self.latencies)
        n = len(sorted_latencies)
        
        return {
            "p50": sorted_latencies[int(n * 0.5)],
            "p95": sorted_latencies[int(n * 0.95)],
            "p99": sorted_latencies[int(n * 0.99)]
        }
    
    def start_system_monitoring(self):
        """Start background system monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_system)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop_system_monitoring(self):
        """Stop background system monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
    
    def _monitor_system(self):
        """Background system monitoring loop."""
        while self._monitoring:
            self._system_metrics = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_available_mb": psutil.virtual_memory().available / (1024 * 1024),
                "timestamp": time.time()
            }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def reset(self):
        """Reset all metrics."""
        self.latencies.clear()
        self.memory_usage.clear()
        self.timestamps.clear()
        self.error_count = 0
        self.total_inferences = 0
        self.cache_hits = 0
        self.cache_misses = 0


class BenchmarkSuite:
    """Comprehensive benchmarking suite for FastVLM models."""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.results = []
    
    def run_latency_benchmark(
        self,
        inference_func: Callable,
        test_data: List[Any],
        warmup_runs: int = 10,
        test_runs: int = 100,
        test_name: str = "latency_test"
    ) -> BenchmarkResult:
        """Run latency benchmark.
        
        Args:
            inference_func: Function to benchmark
            test_data: Test data samples
            warmup_runs: Number of warmup runs
            test_runs: Number of test runs
            test_name: Name of the test
            
        Returns:
            Benchmark results
        """
        logger.info(f"Running latency benchmark: {test_name}")
        
        # Warmup
        logger.info(f"Warming up with {warmup_runs} runs...")
        for i in range(warmup_runs):
            data = test_data[i % len(test_data)]
            try:
                inference_func(data)
            except Exception as e:
                logger.warning(f"Warmup run {i} failed: {e}")
        
        # Reset profiler
        self.profiler.reset()
        
        # Actual benchmark
        logger.info(f"Running {test_runs} test iterations...")
        start_time = time.time()
        latencies = []
        error_count = 0
        
        for i in range(test_runs):
            data = test_data[i % len(test_data)]
            
            with self.profiler.profile_inference():
                try:
                    run_start = time.perf_counter()
                    inference_func(data)
                    run_end = time.perf_counter()
                    latencies.append((run_end - run_start) * 1000)
                except Exception as e:
                    error_count += 1
                    logger.debug(f"Test run {i} failed: {e}")
        
        end_time = time.time()
        test_duration = end_time - start_time
        
        # Calculate statistics
        if latencies:
            avg_latency = statistics.mean(latencies)
            sorted_latencies = sorted(latencies)
            n = len(sorted_latencies)
            
            p50 = sorted_latencies[int(n * 0.5)]
            p95 = sorted_latencies[int(n * 0.95)]
            p99 = sorted_latencies[int(n * 0.99)]
            min_latency = min(latencies)
            max_latency = max(latencies)
        else:
            avg_latency = p50 = p95 = p99 = min_latency = max_latency = 0.0
        
        throughput = test_runs / test_duration
        current_metrics = self.profiler.get_current_metrics()
        
        result = BenchmarkResult(
            model_name="FastVLM",
            test_name=test_name,
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            throughput_fps=throughput,
            memory_usage_mb=current_metrics.memory_peak_mb,
            error_count=error_count,
            total_runs=test_runs,
            test_duration_s=test_duration,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        self.results.append(result)
        logger.info(f"Benchmark completed: {avg_latency:.1f}ms avg, {throughput:.1f} FPS")
        
        return result
    
    def run_memory_benchmark(
        self,
        inference_func: Callable,
        test_data: List[Any],
        test_runs: int = 50
    ) -> Dict[str, float]:
        """Run memory usage benchmark."""
        logger.info("Running memory benchmark...")
        
        self.profiler.reset()
        peak_memory = 0.0
        memory_samples = []
        
        for i in range(test_runs):
            data = test_data[i % len(test_data)]
            
            with self.profiler.profile_inference():
                inference_func(data)
                current_memory = self.profiler._get_memory_usage()
                memory_samples.append(current_memory)
                peak_memory = max(peak_memory, current_memory)
        
        return {
            "peak_memory_mb": peak_memory,
            "avg_memory_mb": statistics.mean(memory_samples),
            "memory_std_mb": statistics.stdev(memory_samples) if len(memory_samples) > 1 else 0.0
        }
    
    def run_stress_test(
        self,
        inference_func: Callable,
        test_data: List[Any],
        duration_seconds: int = 60,
        concurrent_threads: int = 1
    ) -> Dict[str, Any]:
        """Run stress test with sustained load."""
        logger.info(f"Running {duration_seconds}s stress test with {concurrent_threads} threads...")
        
        self.profiler.reset()
        self.profiler.start_system_monitoring()
        
        results = {
            "threads": concurrent_threads,
            "duration_s": duration_seconds,
            "total_inferences": 0,
            "errors": 0,
            "avg_latency_ms": 0.0,
            "throughput_fps": 0.0
        }
        
        def worker_thread():
            start_time = time.time()
            while time.time() - start_time < duration_seconds:
                data = test_data[results["total_inferences"] % len(test_data)]
                
                with self.profiler.profile_inference():
                    try:
                        inference_func(data)
                        results["total_inferences"] += 1
                    except Exception:
                        results["errors"] += 1
        
        # Run worker threads
        threads = []
        start_time = time.time()
        
        for _ in range(concurrent_threads):
            thread = threading.Thread(target=worker_thread)
            thread.start()
            threads.append(thread)
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        actual_duration = end_time - start_time
        
        self.profiler.stop_system_monitoring()
        
        # Calculate final metrics
        results["actual_duration_s"] = actual_duration
        results["throughput_fps"] = results["total_inferences"] / actual_duration
        
        metrics = self.profiler.get_current_metrics()
        results["avg_latency_ms"] = metrics.latency_ms
        results["peak_memory_mb"] = metrics.memory_peak_mb
        results["error_rate"] = results["errors"] / results["total_inferences"] if results["total_inferences"] > 0 else 0.0
        
        logger.info(f"Stress test completed: {results['throughput_fps']:.1f} FPS average")
        
        return results
    
    def generate_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        report = {
            "summary": {
                "total_benchmarks": len(self.results),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "results": [
                {
                    "test_name": r.test_name,
                    "avg_latency_ms": r.avg_latency_ms,
                    "p95_latency_ms": r.p95_latency_ms,
                    "throughput_fps": r.throughput_fps,
                    "memory_usage_mb": r.memory_usage_mb,
                    "success_rate": r.success_rate,
                    "timestamp": r.timestamp
                }
                for r in self.results
            ]
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Benchmark report saved to {output_path}")
        
        return report


class PerformanceOptimizer:
    """Automatic performance optimization for FastVLM models."""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.optimization_history = []
    
    def suggest_optimizations(self, metrics: PerformanceMetrics) -> List[str]:
        """Suggest performance optimizations based on metrics."""
        suggestions = []
        
        # Latency optimizations
        if metrics.latency_ms > 300:
            suggestions.append("Consider using FastVLM-Tiny model for better latency")
            suggestions.append("Enable aggressive quantization (INT4)")
            suggestions.append("Reduce input image resolution")
        
        # Memory optimizations  
        if metrics.memory_peak_mb > 1000:
            suggestions.append("Enable memory optimization in Core ML configuration")
            suggestions.append("Reduce batch size or use streaming inference")
            suggestions.append("Clear inference cache more frequently")
        
        # Cache optimizations
        if metrics.cache_hit_rate < 0.3:
            suggestions.append("Increase cache size for better hit rate")
            suggestions.append("Implement smarter cache eviction policy")
        
        # Error rate optimizations
        if metrics.error_rate > 0.05:
            suggestions.append("Add input validation and error handling")
            suggestions.append("Implement graceful degradation for edge cases")
        
        return suggestions
    
    def auto_optimize_config(self, target_latency_ms: float = 250) -> Dict[str, Any]:
        """Automatically suggest optimal configuration."""
        config = {
            "model_variant": "fast-vlm-base",
            "quantization": "int4",
            "image_size": (336, 336),
            "compute_units": "ALL",
            "cache_size": 100,
            "enable_optimizations": True
        }
        
        # Adjust based on target latency
        if target_latency_ms < 150:
            config.update({
                "model_variant": "fast-vlm-tiny",
                "image_size": (224, 224),
                "quantization": "int4"
            })
        elif target_latency_ms > 400:
            config.update({
                "model_variant": "fast-vlm-large",
                "quantization": "int8"
            })
        
        return config