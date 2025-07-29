"""
Pytest configuration and shared fixtures for Fast VLM tests.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

# Test data constants
SAMPLE_IMAGE_SIZE = (336, 336, 3)
SAMPLE_QUESTION = "What objects are in this image?"
SAMPLE_ANSWER = "A cat and a dog are visible in the image."


@pytest.fixture
def sample_image():
    """Generate a sample image tensor for testing."""
    return np.random.randint(0, 255, SAMPLE_IMAGE_SIZE, dtype=np.uint8)


@pytest.fixture
def sample_question():
    """Provide a sample question for testing."""
    return SAMPLE_QUESTION


@pytest.fixture
def sample_answer():
    """Provide a sample answer for testing."""
    return SAMPLE_ANSWER


@pytest.fixture
def temp_checkpoint_path():
    """Create a temporary checkpoint file path."""
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_model_path():
    """Create a temporary model output path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "test_model.mlpackage"
        yield str(model_path)


@pytest.fixture
def mock_pytorch_model():
    """Create a mock PyTorch model for testing."""
    model = MagicMock()
    model.eval.return_value = model
    model.parameters.return_value = []
    return model


@pytest.fixture
def mock_coreml_model():
    """Create a mock Core ML model for testing."""
    model = MagicMock()
    model.save = MagicMock()
    return model


@pytest.fixture(scope="session")
def test_data_dir():
    """Get the test data directory path."""
    return Path(__file__).parent / "data"


@pytest.fixture
def performance_benchmark():
    """Fixture for performance testing utilities."""
    class BenchmarkHelper:
        @staticmethod
        def measure_time(func, *args, **kwargs):
            """Measure execution time of a function."""
            import time
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            return result, (end - start) * 1000  # Return result and time in ms
        
        @staticmethod
        def measure_memory():
            """Get current memory usage."""
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
    
    return BenchmarkHelper()


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", 
        "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", 
        "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", 
        "performance: mark test as a performance benchmark"
    )
    config.addinivalue_line(
        "markers", 
        "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", 
        "ios: mark test as iOS specific"
    )
    config.addinivalue_line(
        "markers", 
        "macos: mark test as macOS specific"
    )
    config.addinivalue_line(
        "markers", 
        "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", 
        "model: mark test as requiring model files"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on file location."""
    for item in items:
        # Add unit marker to tests in unit test directories
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add performance marker to benchmark tests
        if "benchmark" in str(item.fspath) or "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)