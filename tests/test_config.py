"""
Test configuration and shared settings for the FastVLM test suite.
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path for testing
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Test configuration constants
TEST_CONFIG = {
    # Performance thresholds
    "max_inference_latency_ms": 250,
    "max_memory_usage_mb": 1000,
    "min_accuracy_threshold": 0.70,
    "max_model_size_mb": 500,
    
    # Quantization settings
    "default_quantization": "mixed",
    "supported_quantizations": ["int4", "int8", "fp16", "mixed"],
    
    # Model variants
    "model_variants": {
        "tiny": {"size_mb": 98, "latency_ms": 124, "accuracy": 0.683},
        "base": {"size_mb": 412, "latency_ms": 187, "accuracy": 0.712},
        "large": {"size_mb": 892, "latency_ms": 243, "accuracy": 0.748}
    },
    
    # Test data settings
    "sample_image_size": (336, 336, 3),
    "sample_batch_size": 4,
    "test_questions_count": 10,
    
    # Benchmark settings
    "benchmark_iterations": 100,
    "benchmark_warmup": 10,
    "performance_test_timeout": 300,  # 5 minutes
    
    # Platform settings
    "min_ios_version": "17.0",
    "min_macos_version": "14.0",
    "required_xcode_version": "15.0",
    
    # CI/CD settings
    "ci_test_timeout": 1800,  # 30 minutes
    "ci_memory_limit_mb": 8000,  # 8GB
    "ci_parallel_jobs": 4,
}

# Environment-specific settings
def get_test_environment():
    """Determine the current test environment."""
    if os.getenv("GITHUB_ACTIONS"):
        return "ci"
    elif os.getenv("DOCKER_CONTAINER"):
        return "docker"
    elif sys.platform == "darwin":
        return "macos"
    elif sys.platform.startswith("linux"):
        return "linux"
    else:
        return "unknown"

# Adjust configuration based on environment
ENVIRONMENT = get_test_environment()

if ENVIRONMENT == "ci":
    # Stricter settings for CI
    TEST_CONFIG.update({
        "max_inference_latency_ms": 300,  # Allow more time in CI
        "benchmark_iterations": 50,       # Fewer iterations in CI
        "performance_test_timeout": 600,  # 10 minutes in CI
    })

elif ENVIRONMENT == "docker":
    # Docker-specific adjustments
    TEST_CONFIG.update({
        "max_memory_usage_mb": 2000,      # Higher memory limit in Docker
        "benchmark_iterations": 25,       # Fewer iterations in Docker
    })

# Export configuration
__all__ = ["TEST_CONFIG", "ENVIRONMENT", "get_test_environment"]