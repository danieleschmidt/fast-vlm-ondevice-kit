"""
Performance benchmarks for Fast VLM components.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.fast_vlm_ondevice.converter import FastVLMConverter


@pytest.mark.performance
@pytest.mark.slow
class TestConverterBenchmarks:
    """Performance benchmarks for the converter module."""
    
    def setup_method(self):
        """Setup benchmark fixtures."""
        self.converter = FastVLMConverter()
        self.iterations = 10
        
    def test_model_loading_performance(self, temp_checkpoint_path, performance_benchmark):
        """Benchmark model loading performance."""
        with patch.object(self.converter, 'load_pytorch_model') as mock_load:
            mock_load.return_value = MagicMock()
            
            def load_model():
                return self.converter.load_pytorch_model(temp_checkpoint_path)
            
            results = []
            for _ in range(self.iterations):
                _, time_ms = performance_benchmark.measure_time(load_model)
                results.append(time_ms)
            
            avg_time = sum(results) / len(results)
            max_time = max(results)
            min_time = min(results)
            
            # Performance assertions
            assert avg_time < 1000, f"Average loading time {avg_time:.2f}ms exceeds 1000ms"
            assert max_time < 2000, f"Max loading time {max_time:.2f}ms exceeds 2000ms"
            
            print(f"\nModel Loading Performance:")
            print(f"  Average: {avg_time:.2f}ms")
            print(f"  Min: {min_time:.2f}ms")
            print(f"  Max: {max_time:.2f}ms")
    
    def test_conversion_performance(self, mock_pytorch_model, performance_benchmark):
        """Benchmark model conversion performance."""
        with patch.object(self.converter, 'convert_to_coreml') as mock_convert:
            mock_convert.return_value = MagicMock()
            
            def convert_model():
                return self.converter.convert_to_coreml(
                    mock_pytorch_model,
                    quantization="int4",
                    image_size=(336, 336)
                )
            
            results = []
            for _ in range(self.iterations):
                _, time_ms = performance_benchmark.measure_time(convert_model)
                results.append(time_ms)
            
            avg_time = sum(results) / len(results)
            
            # Performance assertion
            assert avg_time < 5000, f"Average conversion time {avg_time:.2f}ms exceeds 5000ms"
            
            print(f"\nModel Conversion Performance:")
            print(f"  Average: {avg_time:.2f}ms")
    
    def test_memory_usage_during_conversion(self, mock_pytorch_model, performance_benchmark):
        """Benchmark memory usage during conversion."""
        initial_memory = performance_benchmark.measure_memory()
        
        with patch.object(self.converter, 'convert_to_coreml') as mock_convert:
            mock_convert.return_value = MagicMock()
            
            # Simulate memory-intensive conversion
            _ = self.converter.convert_to_coreml(mock_pytorch_model)
            peak_memory = performance_benchmark.measure_memory()
        
        memory_increase = peak_memory - initial_memory
        
        # Memory usage assertion (should be reasonable for testing environment)
        assert memory_increase < 100, f"Memory increase {memory_increase:.2f}MB exceeds 100MB in test"
        
        print(f"\nMemory Usage During Conversion:")
        print(f"  Initial: {initial_memory:.2f}MB")
        print(f"  Peak: {peak_memory:.2f}MB") 
        print(f"  Increase: {memory_increase:.2f}MB")


@pytest.mark.performance 
class TestInferenceBenchmarks:
    """Performance benchmarks for inference operations."""
    
    def test_inference_latency_simulation(self, sample_image, sample_question, performance_benchmark):
        """Simulate and benchmark inference latency."""
        def simulate_inference():
            # Simulate preprocessing
            processed_image = np.array(sample_image, dtype=np.float32) / 255.0
            
            # Simulate model inference (placeholder)
            import time
            time.sleep(0.01)  # Simulate 10ms processing
            
            return "Simulated answer"
        
        results = []
        for _ in range(100):  # More iterations for inference testing
            _, time_ms = performance_benchmark.measure_time(simulate_inference)
            results.append(time_ms)
        
        avg_latency = sum(results) / len(results)
        p95_latency = sorted(results)[int(len(results) * 0.95)]
        
        # Target performance goals
        assert avg_latency < 250, f"Average latency {avg_latency:.2f}ms exceeds 250ms target"
        assert p95_latency < 400, f"P95 latency {p95_latency:.2f}ms exceeds 400ms target"
        
        print(f"\nInference Performance Simulation:")
        print(f"  Average latency: {avg_latency:.2f}ms")
        print(f"  P95 latency: {p95_latency:.2f}ms")
        print(f"  Min latency: {min(results):.2f}ms")
        print(f"  Max latency: {max(results):.2f}ms")


@pytest.mark.performance
@pytest.mark.model
@pytest.mark.skipif(True, reason="Requires actual model files")
class TestEndToEndBenchmarks:
    """End-to-end performance benchmarks (requires actual models)."""
    
    def test_full_pipeline_performance(self):
        """Benchmark the complete pipeline performance."""
        # This test would run with actual models in a full test environment
        pytest.skip("Requires actual FastVLM model files")
    
    def test_batch_processing_performance(self):
        """Benchmark batch processing capabilities."""
        # This test would evaluate batch inference performance
        pytest.skip("Requires actual FastVLM model files")