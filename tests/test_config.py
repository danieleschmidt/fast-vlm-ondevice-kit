"""
Test configuration and utilities for FastVLM On-Device Kit.
Provides centralized configuration for test execution.
"""

import os
from pathlib import Path
from typing import Dict, Any, List


class TestConfig:
    """Centralized test configuration."""
    
    # Test data paths
    TEST_DATA_DIR = Path(__file__).parent / "data"
    FIXTURES_DIR = Path(__file__).parent / "fixtures"
    
    # Model test configurations
    MODEL_VARIANTS = ["tiny", "base", "large"]
    QUANTIZATION_STRATEGIES = ["int4", "int8", "fp16", "mixed"]
    TARGET_DEVICES = ["iPhone14", "iPhone15Pro", "iPadProM2"]
    
    # Performance test thresholds
    PERFORMANCE_THRESHOLDS = {
        "iPhone14": {
            "latency_ms": 300,
            "memory_mb": 600,
            "accuracy_drop": 0.05
        },
        "iPhone15Pro": {
            "latency_ms": 200,
            "memory_mb": 900,
            "accuracy_drop": 0.03
        },
        "iPadProM2": {
            "latency_ms": 150,
            "memory_mb": 1200,
            "accuracy_drop": 0.02
        }
    }
    
    # Test execution settings
    SLOW_TEST_TIMEOUT = 300  # 5 minutes
    BENCHMARK_ITERATIONS = 10
    WARMUP_ITERATIONS = 3
    
    # Coverage requirements
    MIN_COVERAGE_PERCENT = 85
    
    # Test markers configuration
    PYTEST_MARKERS = {
        "unit": "Unit tests - fast, isolated tests",
        "integration": "Integration tests - test component interactions", 
        "e2e": "End-to-end tests - full workflow testing",
        "performance": "Performance tests - benchmark and timing",
        "slow": "Slow tests - may take several minutes",
        "ios": "iOS specific tests", 
        "macos": "macOS specific tests",
        "gpu": "Tests requiring GPU/Neural Engine",
        "model": "Tests requiring model files",
        "security": "Security and vulnerability tests"
    }
    
    @classmethod
    def get_test_data_path(cls, filename: str) -> Path:
        """Get path to test data file."""
        return cls.TEST_DATA_DIR / filename
    
    @classmethod
    def get_fixture_path(cls, filename: str) -> Path:
        """Get path to test fixture file."""
        return cls.FIXTURES_DIR / filename
    
    @classmethod
    def should_run_slow_tests(cls) -> bool:
        """Check if slow tests should be run."""
        return os.getenv("RUN_SLOW_TESTS", "false").lower() == "true"
    
    @classmethod
    def should_run_gpu_tests(cls) -> bool:
        """Check if GPU tests should be run."""
        return os.getenv("RUN_GPU_TESTS", "false").lower() == "true"
    
    @classmethod
    def should_run_model_tests(cls) -> bool:
        """Check if model tests should be run."""
        return os.getenv("RUN_MODEL_TESTS", "false").lower() == "true"
    
    @classmethod
    def get_test_model_path(cls, variant: str = "base") -> str:
        """Get path to test model for the given variant."""
        model_dir = os.getenv("TEST_MODEL_DIR", "tests/models")
        return f"{model_dir}/FastVLM-{variant}.mlpackage"
    
    @classmethod
    def get_benchmark_config(cls) -> Dict[str, Any]:
        """Get benchmark configuration."""
        return {
            "iterations": cls.BENCHMARK_ITERATIONS,
            "warmup": cls.WARMUP_ITERATIONS,
            "timeout": cls.SLOW_TEST_TIMEOUT,
            "devices": cls.TARGET_DEVICES,
            "thresholds": cls.PERFORMANCE_THRESHOLDS
        }


class CITestConfig(TestConfig):
    """Test configuration for CI environment."""
    
    # Reduced iterations for CI speed
    BENCHMARK_ITERATIONS = 3
    WARMUP_ITERATIONS = 1
    SLOW_TEST_TIMEOUT = 120  # 2 minutes
    
    # Relaxed coverage for CI
    MIN_COVERAGE_PERCENT = 80


class LocalTestConfig(TestConfig):
    """Test configuration for local development."""
    
    # More thorough testing locally
    BENCHMARK_ITERATIONS = 20
    WARMUP_ITERATIONS = 5
    SLOW_TEST_TIMEOUT = 600  # 10 minutes
    
    # Strict coverage locally
    MIN_COVERAGE_PERCENT = 90


def get_test_config() -> TestConfig:
    """Get appropriate test configuration based on environment."""
    if os.getenv("CI", "false").lower() == "true":
        return CITestConfig()
    else:
        return LocalTestConfig()


# Test environment detection
IS_CI = os.getenv("CI", "false").lower() == "true"
IS_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
IS_LOCAL_DEV = not IS_CI

# Test data validation
def validate_test_environment():
    """Validate test environment setup."""
    config = get_test_config()
    issues = []
    
    # Check test data directory
    if not config.TEST_DATA_DIR.exists():
        issues.append(f"Test data directory missing: {config.TEST_DATA_DIR}")
    
    # Check for required environment variables in CI
    if IS_CI:
        required_vars = ["PYTHONPATH"]
        for var in required_vars:
            if not os.getenv(var):
                issues.append(f"Required CI environment variable missing: {var}")
    
    # Check model files if model tests are enabled
    if config.should_run_model_tests():
        for variant in config.MODEL_VARIANTS:
            model_path = Path(config.get_test_model_path(variant))
            if not model_path.exists():
                issues.append(f"Test model missing: {model_path}")
    
    return issues


# Export configuration
TEST_CONFIG = get_test_config()

__all__ = [
    'TestConfig',
    'CITestConfig', 
    'LocalTestConfig',
    'get_test_config',
    'TEST_CONFIG',
    'IS_CI',
    'IS_GITHUB_ACTIONS',
    'IS_LOCAL_DEV',
    'validate_test_environment'
]