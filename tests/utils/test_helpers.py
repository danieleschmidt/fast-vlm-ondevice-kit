"""
Test helper utilities for FastVLM testing infrastructure.
"""

import time
import psutil
import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from unittest.mock import MagicMock, patch
from contextlib import contextmanager
import tempfile
import logging

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Performance profiling utilities for testing."""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.measurements = []
    
    def start(self):
        """Start performance measurement."""
        self.start_time = time.perf_counter()
        self.start_memory = self._get_memory_usage()
    
    def stop(self) -> Dict[str, float]:
        """Stop measurement and return results."""
        if self.start_time is None:
            raise RuntimeError("Profiler not started")
        
        end_time = time.perf_counter()
        end_memory = self._get_memory_usage()
        
        result = {
            "duration_ms": (end_time - self.start_time) * 1000,
            "memory_peak_mb": end_memory,
            "memory_delta_mb": end_memory - self.start_memory
        }
        
        self.measurements.append(result)
        return result
    
    @staticmethod
    def _get_memory_usage() -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    @contextmanager
    def measure(self):
        """Context manager for performance measurement."""
        self.start()
        try:
            yield
        finally:
            self.stop()


class ModelTestData:
    """Generator for test model data."""
    
    @staticmethod
    def create_sample_image(size: Tuple[int, int, int] = (336, 336, 3)) -> np.ndarray:
        """Create a sample RGB image."""
        return np.random.randint(0, 255, size, dtype=np.uint8)
    
    @staticmethod
    def create_batch_images(batch_size: int, size: Tuple[int, int, int] = (336, 336, 3)) -> np.ndarray:
        """Create a batch of sample images."""
        return np.random.randint(0, 255, (batch_size,) + size, dtype=np.uint8)
    
    @staticmethod
    def create_sample_questions(count: int = 1) -> List[str]:
        """Create sample questions for testing."""
        questions = [
            "What objects are in this image?",
            "What is the color of the main object?",
            "How many people are visible?",
            "What is the weather like?",
            "What activity is taking place?",
            "What is the setting or location?",
            "What time of day is it?",
            "What emotions are visible?",
            "What brands or text are visible?",
            "What is the dominant color scheme?"
        ]
        return questions[:count] if count <= len(questions) else questions * (count // len(questions) + 1)[:count]
    
    @staticmethod
    def create_mock_model_weights() -> Dict[str, np.ndarray]:
        """Create mock model weights for testing."""
        return {
            "vision_encoder.weight": np.random.randn(768, 3, 16, 16).astype(np.float32),
            "text_encoder.weight": np.random.randn(512, 77, 768).astype(np.float32),
            "fusion.weight": np.random.randn(512, 768).astype(np.float32),
            "decoder.weight": np.random.randn(1000, 512).astype(np.float32),
        }


class MockModelFactory:
    """Factory for creating mock models and components."""
    
    @staticmethod
    def create_pytorch_model():
        """Create a mock PyTorch model."""
        model = MagicMock()
        model.eval.return_value = model
        model.parameters.return_value = []
        model.state_dict.return_value = ModelTestData.create_mock_model_weights()
        model.forward.return_value = np.random.randn(1, 1000).astype(np.float32)
        return model
    
    @staticmethod
    def create_coreml_model():
        """Create a mock Core ML model."""
        model = MagicMock()
        model.save = MagicMock()
        model.predict = MagicMock(return_value={"output": np.random.randn(1, 1000)})
        return model
    
    @staticmethod
    def create_quantization_config():
        """Create a mock quantization configuration."""
        config = MagicMock()
        config.vision_encoder = "int4"
        config.text_encoder = "int8"
        config.fusion_layers = "fp16"
        config.decoder = "int4"
        config.calibration_samples = 1000
        return config


class TestDataManager:
    """Manager for test data and fixtures."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path(__file__).parent.parent / "fixtures"
        self.base_dir.mkdir(exist_ok=True)
    
    def get_sample_image_path(self) -> Path:
        """Get path to sample test image."""
        image_path = self.base_dir / "sample_image.jpg"
        if not image_path.exists():
            self._create_sample_image(image_path)
        return image_path
    
    def get_sample_questions_path(self) -> Path:
        """Get path to sample questions file."""
        questions_path = self.base_dir / "sample_questions.json"
        if not questions_path.exists():
            self._create_sample_questions(questions_path)
        return questions_path
    
    def get_mock_checkpoint_path(self) -> Path:
        """Get path to mock model checkpoint."""
        checkpoint_path = self.base_dir / "mock_checkpoint.pth"
        if not checkpoint_path.exists():
            self._create_mock_checkpoint(checkpoint_path)
        return checkpoint_path
    
    def _create_sample_image(self, path: Path):
        """Create a sample test image file."""
        try:
            from PIL import Image
            image = Image.fromarray(ModelTestData.create_sample_image())
            image.save(path)
        except ImportError:
            # Fallback: create a simple placeholder file
            path.write_bytes(b"fake_image_data")
    
    def _create_sample_questions(self, path: Path):
        """Create sample questions file."""
        questions = {
            "questions": ModelTestData.create_sample_questions(10),
            "expected_answers": [
                "Sample answer for testing purposes" 
                for _ in range(10)
            ]
        }
        path.write_text(json.dumps(questions, indent=2))
    
    def _create_mock_checkpoint(self, path: Path):
        """Create a mock model checkpoint file."""
        # Create a simple fake checkpoint file
        mock_data = {
            "model_state_dict": "mock_weights",
            "metadata": {
                "model_type": "fast_vlm_base",
                "version": "1.0.0",
                "quantization": "fp32"
            }
        }
        path.write_text(json.dumps(mock_data))


class AssertionHelpers:
    """Custom assertion helpers for FastVLM testing."""
    
    @staticmethod
    def assert_model_output_shape(output: np.ndarray, expected_shape: Tuple[int, ...]):
        """Assert model output has expected shape."""
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    @staticmethod
    def assert_latency_under_threshold(duration_ms: float, threshold_ms: float):
        """Assert operation completed under latency threshold."""
        assert duration_ms < threshold_ms, f"Latency {duration_ms:.2f}ms exceeds threshold {threshold_ms}ms"
    
    @staticmethod
    def assert_memory_under_threshold(memory_mb: float, threshold_mb: float):
        """Assert memory usage under threshold."""
        assert memory_mb < threshold_mb, f"Memory usage {memory_mb:.2f}MB exceeds threshold {threshold_mb}MB"
    
    @staticmethod
    def assert_accuracy_above_threshold(accuracy: float, threshold: float):
        """Assert accuracy above minimum threshold."""
        assert accuracy >= threshold, f"Accuracy {accuracy:.3f} below threshold {threshold:.3f}"
    
    @staticmethod
    def assert_quantization_quality(original_accuracy: float, quantized_accuracy: float, max_drop: float = 0.05):
        """Assert quantization doesn't degrade accuracy too much."""
        accuracy_drop = original_accuracy - quantized_accuracy
        assert accuracy_drop <= max_drop, f"Accuracy drop {accuracy_drop:.3f} exceeds threshold {max_drop:.3f}"


class TestEnvironment:
    """Test environment setup and configuration."""
    
    @staticmethod
    def is_gpu_available() -> bool:
        """Check if GPU is available for testing."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    @staticmethod
    def is_coreml_available() -> bool:
        """Check if Core ML is available."""
        try:
            import coremltools
            return True
        except ImportError:
            return False
    
    @staticmethod
    def is_ios_simulator_available() -> bool:
        """Check if iOS simulator is available."""
        import subprocess
        try:
            result = subprocess.run(
                ["xcrun", "simctl", "list", "devices"], 
                capture_output=True, 
                text=True
            )
            return result.returncode == 0 and "iPhone" in result.stdout
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    @staticmethod
    def get_available_memory_mb() -> float:
        """Get available system memory in MB."""
        return psutil.virtual_memory().available / 1024 / 1024
    
    @staticmethod
    def skip_if_insufficient_memory(required_mb: float):
        """Skip test if insufficient memory available."""
        import pytest
        available = TestEnvironment.get_available_memory_mb()
        if available < required_mb:
            pytest.skip(f"Insufficient memory: {available:.0f}MB < {required_mb:.0f}MB")


@contextmanager
def temporary_environment_variable(key: str, value: str):
    """Temporarily set an environment variable."""
    import os
    old_value = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value


@contextmanager
def capture_logs(logger_name: str = None, level: int = logging.INFO):
    """Capture logs for testing."""
    import logging
    from io import StringIO
    
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(level)
    
    logger_obj = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    logger_obj.addHandler(handler)
    logger_obj.setLevel(level)
    
    try:
        yield log_capture
    finally:
        logger_obj.removeHandler(handler)


def parametrize_model_variants():
    """Pytest parametrize decorator for model variants."""
    import pytest
    return pytest.mark.parametrize("model_variant", [
        "fast-vlm-tiny",
        "fast-vlm-base", 
        "fast-vlm-large"
    ])


def parametrize_quantization_strategies():
    """Pytest parametrize decorator for quantization strategies."""
    import pytest
    return pytest.mark.parametrize("quantization", [
        "int4",
        "int8",
        "fp16",
        "mixed"
    ])


def parametrize_device_targets():
    """Pytest parametrize decorator for device targets."""
    import pytest
    return pytest.mark.parametrize("device", [
        "iPhone-15-Pro",
        "iPhone-14",
        "iPad-Pro-M2",
        "Mac-Studio-M2"
    ])


class BenchmarkRunner:
    """Utility for running and collecting benchmark results."""
    
    def __init__(self):
        self.results = []
    
    def run_benchmark(self, 
                     test_function: Callable,
                     iterations: int = 10,
                     warmup: int = 3) -> Dict[str, Any]:
        """Run benchmark with multiple iterations."""
        
        # Warmup runs
        for _ in range(warmup):
            test_function()
        
        # Measured runs
        times = []
        memories = []
        
        for _ in range(iterations):
            profiler = PerformanceProfiler()
            profiler.start()
            test_function()
            result = profiler.stop()
            
            times.append(result["duration_ms"])
            memories.append(result["memory_peak_mb"])
        
        benchmark_result = {
            "iterations": iterations,
            "mean_time_ms": np.mean(times),
            "std_time_ms": np.std(times),
            "min_time_ms": np.min(times),
            "max_time_ms": np.max(times),
            "p95_time_ms": np.percentile(times, 95),
            "mean_memory_mb": np.mean(memories),
            "peak_memory_mb": np.max(memories)
        }
        
        self.results.append(benchmark_result)
        return benchmark_result
    
    def save_results(self, filepath: Path):
        """Save benchmark results to file."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)


# Global instances for easy access
test_data_manager = TestDataManager()
performance_profiler = PerformanceProfiler()
benchmark_runner = BenchmarkRunner()