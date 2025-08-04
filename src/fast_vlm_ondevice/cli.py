"""
Command-line interface for FastVLM On-Device Kit.

Provides conversion, testing, and benchmarking utilities.
"""

import argparse
import sys
import logging
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import torch
    from PIL import Image
    import numpy as np
except ImportError as e:
    logging.warning(f"Some dependencies not available: {e}")

from .converter import FastVLMConverter
from .quantization import QuantizationConfig
from .health import HealthChecker

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def convert_model(args) -> int:
    """Convert PyTorch model to Core ML."""
    try:
        converter = FastVLMConverter()
        
        # Load PyTorch model
        print(f"Loading model from {args.input}")
        model = converter.load_pytorch_model(args.input)
        
        # Convert to Core ML
        print(f"Converting to Core ML with {args.quantization} quantization...")
        coreml_model = converter.convert_to_coreml(
            model=model,
            quantization=args.quantization,
            compute_units=args.compute_units,
            image_size=tuple(args.image_size),
            max_seq_length=args.max_seq_length
        )
        
        # Save model
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        coreml_model.save(str(output_path))
        
        # Report results
        size_mb = converter.get_model_size_mb()
        print(f"✅ Conversion complete!")
        print(f"   Output: {output_path}")
        print(f"   Size: {size_mb:.1f}MB")
        print(f"   Quantization: {args.quantization}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        logger.exception("Conversion error")
        return 1


def test_model(args) -> int:
    """Test converted model with sample inputs."""
    try:
        from .testing import ModelTester
        
        tester = ModelTester(args.model)
        results = tester.run_basic_tests()
        
        print("🧪 Model Test Results:")
        print(f"   Latency: {results['latency_ms']:.1f}ms")
        print(f"   Memory: {results['memory_mb']:.1f}MB")
        print(f"   Status: {'✅ PASS' if results['success'] else '❌ FAIL'}")
        
        return 0 if results['success'] else 1
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        logger.exception("Testing error")
        return 1


def benchmark_model(args) -> int:
    """Benchmark model performance."""
    try:
        from .benchmarking import PerformanceBenchmark
        
        benchmark = PerformanceBenchmark(args.model)
        results = benchmark.run_benchmark(
            iterations=args.iterations,
            warmup=args.warmup
        )
        
        print("📊 Benchmark Results:")
        print(f"   Average Latency: {results['avg_latency_ms']:.1f}ms")
        print(f"   P95 Latency: {results['p95_latency_ms']:.1f}ms")
        print(f"   Peak Memory: {results['peak_memory_mb']:.1f}MB")
        print(f"   Throughput: {results['throughput_fps']:.1f} FPS")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"   Results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Benchmarking failed: {e}")
        logger.exception("Benchmarking error")
        return 1


def health_check(args) -> int:
    """Perform system health checks."""
    try:
        checker = HealthChecker()
        health_status = checker.check_all()
        
        print("🏥 System Health Check:")
        for component, status in health_status.items():
            icon = "✅" if status['healthy'] else "❌"
            print(f"   {icon} {component}: {status['message']}")
        
        overall_healthy = all(s['healthy'] for s in health_status.values())
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(health_status, f, indent=2)
        
        return 0 if overall_healthy else 1
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        logger.exception("Health check error")
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="FastVLM On-Device Kit CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert PyTorch model to Core ML")
    convert_parser.add_argument("input", help="Input PyTorch model path (.pth)")
    convert_parser.add_argument("output", help="Output Core ML model path (.mlpackage)")
    convert_parser.add_argument(
        "--quantization",
        choices=["fp32", "fp16", "int8", "int4"],
        default="int4",
        help="Quantization type"
    )
    convert_parser.add_argument(
        "--compute-units",
        choices=["ALL", "CPU_AND_GPU", "CPU_ONLY"],
        default="ALL",
        help="Target compute units"
    )
    convert_parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[336, 336],
        help="Input image size (height width)"
    )
    convert_parser.add_argument(
        "--max-seq-length",
        type=int,
        default=77,
        help="Maximum text sequence length"
    )
    convert_parser.set_defaults(func=convert_model)
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test converted model")
    test_parser.add_argument("model", help="Core ML model path (.mlpackage)")
    test_parser.set_defaults(func=test_model)
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark model performance")
    benchmark_parser.add_argument("model", help="Core ML model path (.mlpackage)")
    benchmark_parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations"
    )
    benchmark_parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations"
    )
    benchmark_parser.add_argument(
        "--output", "-o",
        help="Output JSON file for results"
    )
    benchmark_parser.set_defaults(func=benchmark_model)
    
    # Health command
    health_parser = subparsers.add_parser("health", help="System health check")
    health_parser.add_argument(
        "--output", "-o",
        help="Output JSON file for health status"
    )
    health_parser.set_defaults(func=health_check)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Execute command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())