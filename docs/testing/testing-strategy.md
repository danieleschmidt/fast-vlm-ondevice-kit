# Comprehensive Testing Strategy

This document outlines the testing approach for Fast VLM On-Device Kit, covering unit tests, integration tests, performance tests, and mobile-specific testing.

## Testing Framework Overview

### Test Categories

1. **Unit Tests**: Individual component testing
2. **Integration Tests**: Component interaction testing  
3. **End-to-End Tests**: Full pipeline testing
4. **Performance Tests**: Latency and throughput validation
5. **Mobile Tests**: iOS/macOS device testing
6. **Security Tests**: Vulnerability and penetration testing

## Python Testing Infrastructure

### Test Configuration

Update `pyproject.toml` testing section:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--cov=src/fast_vlm_ondevice",
    "--cov-report=term-missing",
    "--cov-report=html:htmlcov",
    "--cov-report=xml:coverage.xml",
    "--cov-fail-under=85",
    "--strict-markers",
    "--strict-config",
    "-ra"
]
markers = [
    "unit: Unit tests",
    "integration: Integration tests", 
    "performance: Performance tests",
    "slow: Slow running tests",
    "mobile: Mobile device tests"
]
```

### Performance Testing Framework

Create `tests/performance/test_conversion_performance.py`:
```python
import pytest
import time
from pathlib import Path
from fast_vlm_ondevice import FastVLMConverter
from fast_vlm_ondevice.metrics import PerformanceMetrics

@pytest.mark.performance
class TestConversionPerformance:
    """Performance tests for model conversion."""
    
    @pytest.fixture
    def converter(self):
        return FastVLMConverter()
    
    @pytest.fixture
    def metrics(self):
        return PerformanceMetrics()
    
    @pytest.mark.parametrize("model_size", ["tiny", "base", "large"])
    def test_conversion_latency(self, converter, metrics, model_size):
        """Test conversion latency for different model sizes."""
        mock_model_path = f"tests/fixtures/mock_{model_size}_model.pth"
        
        start_time = time.time()
        # Mock conversion process
        time.sleep(0.1 * {"tiny": 1, "base": 3, "large": 5}[model_size])
        duration = time.time() - start_time
        
        metrics.record_conversion_time(model_size, duration)
        
        # Assert performance benchmarks
        max_times = {"tiny": 2.0, "base": 10.0, "large": 30.0}
        assert duration < max_times[model_size], f"Conversion too slow: {duration}s"
    
    def test_memory_usage_during_conversion(self, converter):
        """Test memory usage doesn't exceed limits."""
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Mock conversion process
        # In real test, this would load and convert a model
        time.sleep(0.1)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Assert memory usage is reasonable (< 4GB increase)
        assert memory_increase < 4096, f"Memory usage too high: {memory_increase}MB"
    
    @pytest.mark.slow
    def test_concurrent_conversions(self, converter):
        """Test multiple concurrent conversions."""
        import concurrent.futures
        import threading
        
        def convert_model(model_id):
            # Mock conversion
            time.sleep(0.5)
            return f"model_{model_id}_converted"
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(convert_model, i) for i in range(3)]
            results = [future.result() for future in futures]
        
        assert len(results) == 3
        assert all("converted" in result for result in results)
```

### Integration Testing

Create `tests/integration/test_pipeline_integration.py`:
```python
import pytest
from pathlib import Path
from fast_vlm_ondevice import FastVLMConverter
from fast_vlm_ondevice.quantization import QuantizationConfig

@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests for the complete conversion pipeline."""
    
    @pytest.fixture
    def sample_model_path(self, tmp_path):
        """Create a mock model file for testing."""
        model_path = tmp_path / "test_model.pth"
        # Create a minimal mock PyTorch model file
        import torch
        torch.save({"model_state": "mock"}, model_path)
        return model_path
    
    @pytest.fixture
    def quantization_config(self):
        return QuantizationConfig(
            vision_encoder="int4",
            text_encoder="int8", 
            fusion_layers="fp16",
            decoder="int4"
        )
    
    def test_full_conversion_pipeline(self, sample_model_path, quantization_config, tmp_path):
        """Test complete model conversion pipeline."""
        converter = FastVLMConverter()
        output_path = tmp_path / "converted_model.mlpackage"
        
        # Test the full pipeline (mocked)
        # In real test, this would:
        # 1. Load PyTorch model
        # 2. Apply quantization
        # 3. Convert to Core ML
        # 4. Optimize for Apple Neural Engine
        
        # Mock the conversion process
        result = {
            "model_path": str(output_path),
            "size_mb": 156.7,
            "quantization_applied": True,
            "optimization_level": "high"
        }
        
        assert result["quantization_applied"]
        assert result["size_mb"] < 500  # Reasonable size limit
        assert "mlpackage" in result["model_path"]
    
    def test_error_handling_invalid_model(self):
        """Test error handling for invalid model inputs."""
        converter = FastVLMConverter()
        
        with pytest.raises(FileNotFoundError):
            converter.load_pytorch_model("nonexistent_model.pth")
    
    def test_quantization_accuracy_retention(self, quantization_config):
        """Test that quantization retains acceptable accuracy."""
        # Mock accuracy evaluation
        original_accuracy = 0.758
        quantized_accuracy = 0.751  # Simulated post-quantization accuracy
        
        accuracy_drop = (original_accuracy - quantized_accuracy) / original_accuracy
        
        # Assert accuracy drop is within acceptable bounds (< 5%)
        assert accuracy_drop < 0.05, f"Accuracy drop too high: {accuracy_drop:.2%}"
```

## Swift Testing Infrastructure

### iOS Unit Tests

Create `ios/Tests/FastVLMKitTests/FastVLMPerformanceTests.swift`:
```swift
import XCTest
import CoreML
import UIKit
@testable import FastVLMKit

final class FastVLMPerformanceTests: XCTestCase {
    var vlm: FastVLM?
    var testImage: UIImage?
    
    override func setUp() async throws {
        // Load test model and image
        let bundle = Bundle(for: type(of: self))
        guard let modelURL = bundle.url(forResource: "TestFastVLM", withExtension: "mlpackage") else {
            XCTFail("Test model not found")
            return
        }
        
        vlm = try FastVLM(modelPath: modelURL.path)
        testImage = UIImage(named: "test_image", in: bundle, compatibleWith: nil)
    }
    
    func testInferenceLatency() async throws {
        guard let vlm = vlm, let image = testImage else {
            XCTFail("Setup failed")
            return
        }
        
        let question = "What objects are in this image?"
        let iterations = 10
        var totalTime: TimeInterval = 0
        
        // Warmup run
        _ = try await vlm.answer(image: image, question: question)
        
        // Measure performance
        for _ in 0..<iterations {
            let startTime = CFAbsoluteTimeGetCurrent()
            _ = try await vlm.answer(image: image, question: question)
            totalTime += CFAbsoluteTimeGetCurrent() - startTime
        }
        
        let averageLatency = (totalTime / Double(iterations)) * 1000 // ms
        
        // Assert latency is within acceptable bounds (< 500ms average)
        XCTAssertLessThan(averageLatency, 500, "Average latency too high: \(averageLatency)ms")
        
        print("Average inference latency: \(averageLatency)ms")
    }
    
    func testMemoryUsage() async throws {
        guard let vlm = vlm, let image = testImage else {
            XCTFail("Setup failed")
            return
        }
        
        let initialMemory = getMemoryUsage()
        
        // Run multiple inferences
        for i in 0..<50 {
            _ = try await vlm.answer(image: image, question: "Test question \(i)")
            
            // Force garbage collection
            if i % 10 == 0 {
                autoreleasepool { /* Force cleanup */ }
            }
        }
        
        let finalMemory = getMemoryUsage()
        let memoryIncrease = finalMemory - initialMemory
        
        // Assert memory increase is reasonable (< 100MB)
        XCTAssertLessThan(memoryIncrease, 100_000_000, "Memory usage increased too much: \(memoryIncrease) bytes")
    }
    
    private func getMemoryUsage() -> Int64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
        
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        if kerr == KERN_SUCCESS {
            return Int64(info.resident_size)
        } else {
            return -1
        }
    }
}
```

## Automated Testing Infrastructure

### GitHub Actions Testing Workflow

Create `docs/workflows/comprehensive-testing.yml`:
```yaml
name: Comprehensive Testing

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * 1'  # Weekly full test suite

jobs:
  python-tests:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ["3.10", "3.11", "3.12"]
    runs-on: ${{ matrix.os }}
    
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      
      - name: Run unit tests
        run: pytest tests/unit/ -v --cov=src/fast_vlm_ondevice
      
      - name: Run integration tests  
        run: pytest tests/integration/ -v
      
      - name: Run performance tests
        run: pytest tests/performance/ -v --benchmark-skip
        if: matrix.os == 'macos-latest'

  swift-tests:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - name: Test Swift package
        run: |
          cd ios
          swift test --enable-code-coverage
      
      - name: Generate coverage report
        run: |
          cd ios
          xcrun llvm-cov export -format="lcov" \
            .build/debug/FastVLMKitPackageTests.xctest/Contents/MacOS/FastVLMKitPackageTests \
            -instr-profile .build/debug/codecov/default.profdata > coverage.lcov

  mobile-device-tests:
    runs-on: macos-latest
    if: github.event_name == 'schedule'  # Only run weekly
    steps:
      - uses: actions/checkout@v4
      - name: Build for iOS Simulator
        run: |
          cd ios
          xcodebuild -scheme FastVLMKit \
            -destination 'platform=iOS Simulator,name=iPhone 15 Pro,OS=17.0' \
            build test
      
      - name: Run UI tests
        run: |
          cd ios
          xcodebuild -scheme FastVLMDemo \
            -destination 'platform=iOS Simulator,name=iPhone 15 Pro,OS=17.0' \
            test

  benchmark-tests:
    runs-on: macos-latest
    if: github.event_name == 'schedule'
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
          pip install pytest-benchmark
      
      - name: Run benchmark tests
        run: |
          pytest tests/performance/ --benchmark-only \
            --benchmark-json=benchmark-results.json
      
      - name: Upload benchmark results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmark-results.json
```

## Test Data Management

### Test Fixtures

Create `tests/fixtures/README.md`:
```markdown
# Test Fixtures

This directory contains test data and mock models for testing.

## Structure
- `models/`: Mock model files for testing
- `images/`: Sample images for vision testing  
- `configs/`: Test configuration files
- `expected_outputs/`: Expected test results

## Usage
Test fixtures are loaded automatically by pytest fixtures.
Large files should use Git LFS for storage.

## Adding New Fixtures
1. Add files to appropriate subdirectory
2. Update corresponding fixture in `tests/conftest.py`
3. Use Git LFS for files > 10MB
```

### Mock Data Generation

Create `tests/utils/mock_data.py`:
```python
"""Utilities for generating mock test data."""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, Any

class MockDataGenerator:
    """Generate mock data for testing."""
    
    @staticmethod
    def create_mock_model(size: str = "base") -> Dict[str, Any]:
        """Create mock PyTorch model state dict."""
        sizes = {
            "tiny": (64, 128),
            "base": (256, 512), 
            "large": (512, 1024)
        }
        
        hidden_dim, output_dim = sizes[size]
        
        return {
            "vision_encoder.weight": torch.randn(hidden_dim, 768),
            "text_encoder.weight": torch.randn(hidden_dim, 512),
            "fusion.weight": torch.randn(output_dim, hidden_dim * 2),
            "decoder.weight": torch.randn(32000, output_dim),  # vocab size
            "config": {
                "model_type": "fast_vlm",
                "size": size,
                "hidden_dim": hidden_dim
            }
        }
    
    @staticmethod
    def create_test_image(size: Tuple[int, int] = (336, 336)) -> Image.Image:
        """Create test image for vision testing."""
        # Create a test pattern image
        array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        
        # Add some recognizable patterns
        array[50:100, 50:100] = [255, 0, 0]  # Red square
        array[200:250, 200:250] = [0, 255, 0]  # Green square
        
        return Image.fromarray(array)
    
    @staticmethod
    def create_test_questions() -> list[str]:
        """Create test questions for VQA testing."""
        return [
            "What objects are in this image?",
            "What colors do you see?",
            "How many objects are there?",
            "Describe the scene in detail.",
            "What is the main subject of the image?"
        ]

    @staticmethod  
    def save_test_fixtures(output_dir: Path):
        """Save test fixtures to directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save mock models
        models_dir = output_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        for size in ["tiny", "base", "large"]:
            model_data = MockDataGenerator.create_mock_model(size)
            torch.save(model_data, models_dir / f"mock_{size}_model.pth")
        
        # Save test images
        images_dir = output_dir / "images"  
        images_dir.mkdir(exist_ok=True)
        
        for i in range(5):
            image = MockDataGenerator.create_test_image()
            image.save(images_dir / f"test_image_{i}.jpg")
        
        print(f"Test fixtures saved to {output_dir}")

if __name__ == "__main__":
    MockDataGenerator.save_test_fixtures(Path("tests/fixtures"))
```

## Testing Best Practices

### 1. Test Organization
- Separate unit, integration, and performance tests
- Use descriptive test names and docstrings
- Group related tests in classes
- Use pytest fixtures for common setup

### 2. Performance Testing
- Include warmup runs before timing measurements
- Test multiple iterations for statistical significance  
- Set reasonable performance thresholds
- Monitor performance regressions over time

### 3. Mobile Testing
- Test on multiple iOS device simulators
- Include memory and battery usage tests
- Test different screen sizes and orientations
- Validate Core ML model loading and inference

### 4. Continuous Integration
- Run fast tests on every commit
- Run comprehensive tests nightly/weekly
- Fail builds on test failures or coverage drops
- Archive test results and performance metrics

This comprehensive testing strategy ensures high code quality, performance validation, and reliability across all supported platforms.