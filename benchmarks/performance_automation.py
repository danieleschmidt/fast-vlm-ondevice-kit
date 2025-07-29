#!/usr/bin/env python3
"""
Automated Performance Benchmarking for Fast VLM On-Device Kit

This module provides comprehensive performance testing automation
for maturing SDLC environments.
"""

import json
import time
import psutil
import asyncio
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Performance metrics collection"""
    test_name: str
    latency_ms: float
    memory_mb: float
    cpu_percent: float
    throughput_ops_sec: float
    accuracy_score: Optional[float] = None
    energy_consumption_mwh: Optional[float] = None
    

@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    iterations: int = 100
    warmup_iterations: int = 10
    timeout_seconds: int = 300
    measure_energy: bool = False
    collect_detailed_metrics: bool = True


class PerformanceBenchmark:
    """Main benchmark orchestrator"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkMetrics] = []
        
    async def run_inference_benchmark(self, model_path: str) -> BenchmarkMetrics:
        """Benchmark model inference performance"""
        logger.info(f"Running inference benchmark for {model_path}")
        
        # Warmup phase
        for _ in range(self.config.warmup_iterations):
            await self._simulate_inference()
            
        # Measurement phase
        latencies = []
        memory_readings = []
        cpu_readings = []
        
        start_time = time.perf_counter()
        
        for _ in range(self.config.iterations):
            inference_start = time.perf_counter()
            await self._simulate_inference()
            inference_end = time.perf_counter()
            
            latencies.append((inference_end - inference_start) * 1000)
            memory_readings.append(psutil.virtual_memory().used / 1024 / 1024)
            cpu_readings.append(psutil.cpu_percent(interval=0.1))
            
        end_time = time.perf_counter()
        total_time = end_time - start_time
        throughput = self.config.iterations / total_time
        
        return BenchmarkMetrics(
            test_name=f"inference_{Path(model_path).stem}",
            latency_ms=sum(latencies) / len(latencies),
            memory_mb=sum(memory_readings) / len(memory_readings),
            cpu_percent=sum(cpu_readings) / len(cpu_readings),
            throughput_ops_sec=throughput
        )
    
    async def run_memory_benchmark(self) -> BenchmarkMetrics:
        """Benchmark memory usage patterns"""
        logger.info("Running memory benchmark")
        
        initial_memory = psutil.virtual_memory().used
        peak_memory = initial_memory
        
        # Simulate memory-intensive operations
        for i in range(self.config.iterations):
            await self._simulate_memory_operation(i)
            current_memory = psutil.virtual_memory().used
            peak_memory = max(peak_memory, current_memory)
            
        memory_delta_mb = (peak_memory - initial_memory) / 1024 / 1024
        
        return BenchmarkMetrics(
            test_name="memory_usage",
            latency_ms=0.0,
            memory_mb=memory_delta_mb,
            cpu_percent=psutil.cpu_percent(),
            throughput_ops_sec=self.config.iterations / 10.0  # Simulated
        )
    
    async def run_concurrent_benchmark(self, num_workers: int = 4) -> BenchmarkMetrics:
        """Benchmark concurrent processing performance"""
        logger.info(f"Running concurrent benchmark with {num_workers} workers")
        
        start_time = time.perf_counter()
        
        tasks = []
        for _ in range(num_workers):
            task = asyncio.create_task(self._concurrent_worker())
            tasks.append(task)
            
        await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        return BenchmarkMetrics(
            test_name=f"concurrent_{num_workers}_workers",
            latency_ms=duration * 1000,
            memory_mb=psutil.virtual_memory().used / 1024 / 1024,
            cpu_percent=psutil.cpu_percent(),
            throughput_ops_sec=num_workers * self.config.iterations / duration
        )
    
    async def run_comprehensive_suite(self, model_paths: List[str]) -> Dict[str, List[BenchmarkMetrics]]:
        """Run complete benchmark suite"""
        logger.info("Starting comprehensive benchmark suite")
        
        results = {
            "inference": [],
            "memory": [],
            "concurrent": [],
            "system": []
        }
        
        # Inference benchmarks for each model
        for model_path in model_paths:
            metrics = await self.run_inference_benchmark(model_path)
            results["inference"].append(metrics)
            
        # Memory benchmark
        memory_metrics = await self.run_memory_benchmark()
        results["memory"].append(memory_metrics)
        
        # Concurrent processing benchmarks
        for workers in [1, 2, 4, 8]:
            concurrent_metrics = await self.run_concurrent_benchmark(workers)
            results["concurrent"].append(concurrent_metrics)
            
        # System resource benchmark
        system_metrics = await self._run_system_benchmark()
        results["system"].append(system_metrics)
        
        return results
    
    async def _simulate_inference(self):
        """Simulate model inference operation"""
        # Simulate computation time
        await asyncio.sleep(0.01)
        
        # Simulate memory allocation
        dummy_data = bytearray(1024 * 100)  # 100KB
        del dummy_data
    
    async def _simulate_memory_operation(self, iteration: int):
        """Simulate memory-intensive operation"""
        # Gradually increase memory usage
        size = 1024 * (iteration + 1)
        dummy_data = bytearray(size)
        await asyncio.sleep(0.001)
        del dummy_data
    
    async def _concurrent_worker(self):
        """Worker for concurrent benchmark"""
        for _ in range(self.config.iterations // 4):
            await self._simulate_inference()
    
    async def _run_system_benchmark(self) -> BenchmarkMetrics:
        """Benchmark overall system performance"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        
        return BenchmarkMetrics(
            test_name="system_resources",
            latency_ms=0.0,
            memory_mb=memory_info.used / 1024 / 1024,
            cpu_percent=cpu_percent,
            throughput_ops_sec=0.0
        )


class BenchmarkReporter:
    """Generate benchmark reports"""
    
    def __init__(self, output_dir: Path = Path("benchmark-results")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_json_report(self, results: Dict[str, List[BenchmarkMetrics]], 
                           filename: str = "benchmark-results.json"):
        """Generate JSON benchmark report"""
        serializable_results = {}
        for category, metrics_list in results.items():
            serializable_results[category] = [asdict(m) for m in metrics_list]
        
        report_path = self.output_dir / filename
        with open(report_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"JSON report saved to {report_path}")
    
    def generate_markdown_report(self, results: Dict[str, List[BenchmarkMetrics]], 
                               filename: str = "benchmark-report.md"):
        """Generate Markdown benchmark report"""
        report_path = self.output_dir / filename
        
        with open(report_path, 'w') as f:
            f.write("# Fast VLM Performance Benchmark Report\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for category, metrics_list in results.items():
                f.write(f"## {category.title()} Benchmarks\n\n")
                
                if metrics_list:
                    f.write("| Test Name | Latency (ms) | Memory (MB) | CPU (%) | Throughput (ops/sec) |\n")
                    f.write("|-----------|--------------|-------------|---------|---------------------|\n")
                    
                    for metrics in metrics_list:
                        f.write(f"| {metrics.test_name} | {metrics.latency_ms:.2f} | "
                               f"{metrics.memory_mb:.1f} | {metrics.cpu_percent:.1f} | "
                               f"{metrics.throughput_ops_sec:.2f} |\n")
                    
                    f.write("\n")
        
        logger.info(f"Markdown report saved to {report_path}")
    
    def generate_ci_metrics(self, results: Dict[str, List[BenchmarkMetrics]], 
                          filename: str = "ci-metrics.json"):
        """Generate CI-friendly metrics for automated analysis"""
        ci_metrics = {
            "timestamp": time.time(),
            "summary": {
                "total_tests": sum(len(metrics) for metrics in results.values()),
                "avg_latency_ms": 0.0,
                "peak_memory_mb": 0.0,
                "avg_cpu_percent": 0.0,
                "total_throughput": 0.0
            },
            "thresholds": {
                "max_latency_ms": 1000.0,
                "max_memory_mb": 2048.0,
                "max_cpu_percent": 80.0,
                "min_throughput": 10.0
            },
            "status": "PASS"
        }
        
        all_metrics = []
        for metrics_list in results.values():
            all_metrics.extend(metrics_list)
        
        if all_metrics:
            ci_metrics["summary"]["avg_latency_ms"] = sum(m.latency_ms for m in all_metrics) / len(all_metrics)
            ci_metrics["summary"]["peak_memory_mb"] = max(m.memory_mb for m in all_metrics)
            ci_metrics["summary"]["avg_cpu_percent"] = sum(m.cpu_percent for m in all_metrics) / len(all_metrics)
            ci_metrics["summary"]["total_throughput"] = sum(m.throughput_ops_sec for m in all_metrics)
            
            # Check thresholds
            thresholds = ci_metrics["thresholds"]
            summary = ci_metrics["summary"]
            
            if (summary["avg_latency_ms"] > thresholds["max_latency_ms"] or
                summary["peak_memory_mb"] > thresholds["max_memory_mb"] or
                summary["avg_cpu_percent"] > thresholds["max_cpu_percent"] or
                summary["total_throughput"] < thresholds["min_throughput"]):
                ci_metrics["status"] = "FAIL"
        
        report_path = self.output_dir / filename
        with open(report_path, 'w') as f:
            json.dump(ci_metrics, f, indent=2)
        
        logger.info(f"CI metrics saved to {report_path}")
        return ci_metrics["status"]


async def main():
    """Main benchmark execution"""
    config = BenchmarkConfig(
        iterations=50,
        warmup_iterations=5,
        timeout_seconds=300,
        collect_detailed_metrics=True
    )
    
    benchmark = PerformanceBenchmark(config)
    reporter = BenchmarkReporter()
    
    # Simulate model paths (in real implementation, these would be actual models)
    model_paths = [
        "models/fast-vlm-tiny.mlpackage",
        "models/fast-vlm-base.mlpackage",
        "models/fast-vlm-large.mlpackage"
    ]
    
    try:
        results = await benchmark.run_comprehensive_suite(model_paths)
        
        # Generate reports
        reporter.generate_json_report(results)
        reporter.generate_markdown_report(results)
        status = reporter.generate_ci_metrics(results)
        
        logger.info(f"Benchmark completed with status: {status}")
        
        # Exit with appropriate code for CI
        return 0 if status == "PASS" else 1
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)