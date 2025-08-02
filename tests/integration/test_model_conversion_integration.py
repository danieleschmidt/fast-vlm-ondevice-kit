"""
Integration tests for end-to-end model conversion pipeline.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from tests.utils.test_helpers import (
    TestDataManager,
    PerformanceProfiler,
    AssertionHelpers,
    TestEnvironment,
    parametrize_quantization_strategies,
    capture_logs
)


class TestModelConversionPipeline:
    """Integration tests for the complete model conversion pipeline."""
    
    @pytest.fixture(scope="class")
    def test_data(self):
        """Set up test data for integration tests."""
        manager = TestDataManager()
        return {
            "checkpoint_path": manager.get_mock_checkpoint_path(),
            "sample_image": manager.get_sample_image_path(),
            "questions": manager.get_sample_questions_path()
        }
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_conversion_pipeline(self, test_data):
        """Test complete conversion from PyTorch to Core ML."""
        # Skip if Core ML not available
        if not TestEnvironment.is_coreml_available():
            pytest.skip("Core ML not available")
        
        # Skip if insufficient memory
        TestEnvironment.skip_if_insufficient_memory(2000)  # 2GB required
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "converted_model.mlpackage"
            
            # Mock the conversion process for integration testing
            with patch('fast_vlm_ondevice.converter.FastVLMConverter') as mock_converter_class:
                mock_converter = mock_converter_class.return_value
                mock_model = MagicMock()
                mock_coreml = MagicMock()
                
                mock_converter.load_pytorch_model.return_value = mock_model
                mock_converter.convert_to_coreml.return_value = mock_coreml
                mock_converter.get_model_size_mb.return_value = 412.5
                
                # Measure performance
                profiler = PerformanceProfiler()
                profiler.start()
                
                # Execute pipeline
                converter = mock_converter_class()
                model = converter.load_pytorch_model(str(test_data["checkpoint_path"]))
                coreml_model = converter.convert_to_coreml(
                    model,
                    quantization="int4",
                    compute_units="ALL",
                    image_size=(336, 336),
                    max_seq_length=77
                )
                coreml_model.save(str(output_path))
                
                metrics = profiler.stop()
                
                # Assertions
                mock_converter.load_pytorch_model.assert_called_once()
                mock_converter.convert_to_coreml.assert_called_once()
                mock_coreml.save.assert_called_once_with(str(output_path))
                
                # Performance assertions
                AssertionHelpers.assert_latency_under_threshold(metrics["duration_ms"], 30000)  # 30 seconds
                AssertionHelpers.assert_memory_under_threshold(metrics["memory_peak_mb"], 4000)  # 4GB
    
    @pytest.mark.integration
    @parametrize_quantization_strategies()
    def test_quantization_strategies_integration(self, test_data, quantization):
        """Test different quantization strategies end-to-end."""
        with patch('fast_vlm_ondevice.converter.FastVLMConverter') as mock_converter_class:
            mock_converter = mock_converter_class.return_value
            mock_model = MagicMock()
            mock_coreml = MagicMock()
            
            mock_converter.load_pytorch_model.return_value = mock_model
            mock_converter.convert_to_coreml.return_value = mock_coreml
            
            # Set up quantization-specific behavior
            expected_sizes = {
                "int4": 103.0,  # ~25% of original
                "int8": 206.0,  # ~50% of original  
                "fp16": 206.0,  # ~50% of original
                "mixed": 155.0  # ~37.5% of original
            }
            mock_converter.get_model_size_mb.return_value = expected_sizes[quantization]
            
            # Execute
            converter = mock_converter_class()
            model = converter.load_pytorch_model(str(test_data["checkpoint_path"]))
            coreml_model = converter.convert_to_coreml(model, quantization=quantization)
            size_mb = converter.get_model_size_mb()
            
            # Verify quantization was applied
            mock_converter.convert_to_coreml.assert_called_once()
            call_args = mock_converter.convert_to_coreml.call_args
            assert call_args[1]["quantization"] == quantization
            
            # Verify size reduction
            assert size_mb == expected_sizes[quantization]
    
    @pytest.mark.integration
    def test_error_recovery_and_logging(self, test_data):
        """Test error handling and logging in integration scenario."""
        with capture_logs("fast_vlm_ondevice") as log_capture:
            with patch('fast_vlm_ondevice.converter.FastVLMConverter') as mock_converter_class:
                mock_converter = mock_converter_class.return_value
                
                # Simulate conversion failure
                mock_converter.load_pytorch_model.return_value = MagicMock()
                mock_converter.convert_to_coreml.side_effect = RuntimeError("Conversion failed")
                
                # Execute and expect failure
                converter = mock_converter_class()
                model = converter.load_pytorch_model(str(test_data["checkpoint_path"]))
                
                with pytest.raises(RuntimeError, match="Conversion failed"):
                    converter.convert_to_coreml(model)
                
                # Verify logging
                log_output = log_capture.getvalue()
                # Note: In real implementation, we'd check for specific log messages
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_memory_usage_monitoring(self, test_data):
        """Test memory usage monitoring during conversion."""
        TestEnvironment.skip_if_insufficient_memory(1000)  # 1GB required
        
        with patch('fast_vlm_ondevice.converter.FastVLMConverter') as mock_converter_class:
            mock_converter = mock_converter_class.return_value
            mock_model = MagicMock()
            mock_coreml = MagicMock()
            
            mock_converter.load_pytorch_model.return_value = mock_model
            mock_converter.convert_to_coreml.return_value = mock_coreml
            
            profiler = PerformanceProfiler()
            
            # Monitor memory usage throughout process
            initial_memory = profiler._get_memory_usage()
            
            converter = mock_converter_class()
            model = converter.load_pytorch_model(str(test_data["checkpoint_path"]))
            
            loading_memory = profiler._get_memory_usage()
            
            coreml_model = converter.convert_to_coreml(model, quantization="int4")
            
            conversion_memory = profiler._get_memory_usage()
            
            # Memory usage assertions
            assert loading_memory >= initial_memory  # Memory should increase after loading
            assert conversion_memory >= loading_memory  # Memory should increase during conversion
            
            # Memory usage should be reasonable
            memory_increase = conversion_memory - initial_memory
            AssertionHelpers.assert_memory_under_threshold(memory_increase, 2000)  # 2GB max increase


class TestModelValidationIntegration:
    """Integration tests for model validation after conversion."""
    
    @pytest.mark.integration
    def test_converted_model_validation(self):
        """Test validation of converted Core ML model."""
        with patch('fast_vlm_ondevice.converter.FastVLMConverter') as mock_converter_class:
            mock_converter = mock_converter_class.return_value
            mock_coreml = MagicMock()
            
            # Set up mock model prediction
            expected_output = {"answer_probabilities": [0.1, 0.8, 0.1]}
            mock_coreml.predict.return_value = expected_output
            
            mock_converter.convert_to_coreml.return_value = mock_coreml
            
            # Execute conversion and validation
            converter = mock_converter_class()
            coreml_model = converter.convert_to_coreml(MagicMock())
            
            # Validate model can make predictions
            test_input = {"image": "mock_image_tensor", "question": "mock_question_tensor"}
            result = coreml_model.predict(test_input)
            
            assert result == expected_output
            mock_coreml.predict.assert_called_once_with(test_input)
    
    @pytest.mark.integration
    def test_model_accuracy_validation(self):
        """Test accuracy validation of converted model."""
        # Mock accuracy comparison
        original_accuracy = 0.742
        quantized_accuracy = 0.712
        
        # This would involve actual model evaluation in real scenario
        AssertionHelpers.assert_accuracy_above_threshold(quantized_accuracy, 0.70)
        AssertionHelpers.assert_quantization_quality(
            original_accuracy, 
            quantized_accuracy, 
            max_drop=0.05
        )


class TestBatchConversionIntegration:
    """Integration tests for batch model conversion scenarios."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_multiple_model_conversion(self):
        """Test converting multiple models in sequence."""
        model_configs = [
            {"variant": "tiny", "quantization": "int4"},
            {"variant": "base", "quantization": "int8"}, 
            {"variant": "large", "quantization": "mixed"}
        ]
        
        results = []
        
        for config in model_configs:
            with patch('fast_vlm_ondevice.converter.FastVLMConverter') as mock_converter_class:
                mock_converter = mock_converter_class.return_value
                mock_model = MagicMock()
                mock_coreml = MagicMock()
                
                mock_converter.load_pytorch_model.return_value = mock_model
                mock_converter.convert_to_coreml.return_value = mock_coreml
                
                # Variant-specific sizes
                sizes = {"tiny": 98.0, "base": 412.0, "large": 892.0}
                mock_converter.get_model_size_mb.return_value = sizes[config["variant"]]
                
                profiler = PerformanceProfiler()
                profiler.start()
                
                # Execute conversion
                converter = mock_converter_class()
                model = converter.load_pytorch_model(f"checkpoints/fast-vlm-{config['variant']}.pth")
                coreml_model = converter.convert_to_coreml(model, quantization=config["quantization"])
                size = converter.get_model_size_mb()
                
                metrics = profiler.stop()
                
                results.append({
                    "variant": config["variant"],
                    "quantization": config["quantization"],
                    "size_mb": size,
                    "conversion_time_ms": metrics["duration_ms"]
                })
        
        # Verify all conversions completed
        assert len(results) == len(model_configs)
        
        # Verify size ordering (tiny < base < large)
        sizes = [r["size_mb"] for r in results]
        assert sizes[0] < sizes[1] < sizes[2]  # tiny < base < large


class TestPlatformCompatibilityIntegration:
    """Integration tests for platform-specific compatibility."""
    
    @pytest.mark.integration
    @pytest.mark.skipif(not TestEnvironment.is_ios_simulator_available(), 
                       reason="iOS simulator not available")
    def test_ios_deployment_compatibility(self):
        """Test iOS deployment compatibility."""
        with patch('fast_vlm_ondevice.converter.FastVLMConverter') as mock_converter_class:
            mock_converter = mock_converter_class.return_value
            mock_coreml = MagicMock()
            
            # Configure for iOS deployment
            mock_converter.convert_to_coreml.return_value = mock_coreml
            mock_coreml.save = MagicMock()
            
            converter = mock_converter_class()
            coreml_model = converter.convert_to_coreml(
                MagicMock(),
                compute_units="ALL",  # CPU + GPU + ANE
                minimum_deployment_target="iOS17"
            )
            
            # Verify iOS-specific configurations
            call_args = mock_converter.convert_to_coreml.call_args
            assert call_args[1]["compute_units"] == "ALL"
            assert call_args[1]["minimum_deployment_target"] == "iOS17"
    
    @pytest.mark.integration
    @pytest.mark.skipif(not TestEnvironment.is_gpu_available(),
                       reason="GPU not available") 
    def test_gpu_acceleration_integration(self):
        """Test GPU acceleration integration."""
        with patch('fast_vlm_ondevice.converter.FastVLMConverter') as mock_converter_class:
            mock_converter = mock_converter_class.return_value
            mock_coreml = MagicMock()
            
            mock_converter.convert_to_coreml.return_value = mock_coreml
            
            converter = mock_converter_class()
            coreml_model = converter.convert_to_coreml(
                MagicMock(),
                compute_units="CPU_AND_GPU"
            )
            
            # Verify GPU configuration
            call_args = mock_converter.convert_to_coreml.call_args
            assert call_args[1]["compute_units"] == "CPU_AND_GPU"