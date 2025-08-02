"""
Integration tests for the model conversion pipeline.
Tests the complete flow from PyTorch to Core ML conversion.
"""

import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from fast_vlm_ondevice.converter import FastVLMConverter
from fast_vlm_ondevice.quantization import QuantizationConfig


class TestConversionPipeline:
    """Test the complete model conversion pipeline."""
    
    @pytest.fixture
    def converter(self):
        """Create a FastVLM converter instance."""
        return FastVLMConverter()
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.mark.integration
    def test_pytorch_to_coreml_conversion(self, converter, mock_fastvlm_model, 
                                        temp_output_dir):
        """Test complete PyTorch to Core ML conversion."""
        output_path = temp_output_dir / "test_model.mlpackage"
        
        with patch('coremltools.convert') as mock_convert:
            # Mock Core ML conversion
            mock_coreml_model = MagicMock()
            mock_convert.return_value = mock_coreml_model
            
            # Perform conversion
            result = converter.convert_to_coreml(
                model=mock_fastvlm_model,
                output_path=str(output_path),
                quantization="int4",
                compute_units="ALL"
            )
            
            # Verify conversion was attempted
            mock_convert.assert_called_once()
            assert result is not None
    
    @pytest.mark.integration
    def test_quantization_pipeline(self, converter, mock_fastvlm_model, 
                                 sample_quantization_config):
        """Test quantization pipeline integration."""
        with patch('fast_vlm_ondevice.quantization.apply_quantization') as mock_quant:
            mock_quant.return_value = mock_fastvlm_model
            
            # Apply quantization
            quantized_model = converter.apply_quantization(
                model=mock_fastvlm_model,
                config=sample_quantization_config
            )
            
            # Verify quantization was applied
            mock_quant.assert_called_once_with(mock_fastvlm_model, sample_quantization_config)
            assert quantized_model is not None
    
    @pytest.mark.integration
    def test_end_to_end_conversion(self, converter, mock_checkpoint_file, temp_output_dir):
        """Test end-to-end conversion from checkpoint to Core ML."""
        output_path = temp_output_dir / "e2e_model.mlpackage"
        
        with patch('torch.load') as mock_load, \
             patch('fast_vlm_ondevice.converter.FastVLMConverter.load_pytorch_model') as mock_load_model, \
             patch('coremltools.convert') as mock_convert:
            
            # Mock checkpoint loading
            mock_checkpoint = {
                "model_state_dict": {"layer.weight": torch.randn(10, 10)},
                "config": {"model_type": "fast-vlm-base"}
            }
            mock_load.return_value = mock_checkpoint
            
            # Mock model loading
            mock_model = MagicMock()
            mock_load_model.return_value = mock_model
            
            # Mock Core ML conversion
            mock_coreml_model = MagicMock()
            mock_convert.return_value = mock_coreml_model
            
            # Perform end-to-end conversion
            result = converter.convert_checkpoint_to_coreml(
                checkpoint_path=mock_checkpoint_file,
                output_path=str(output_path),
                quantization="mixed"
            )
            
            # Verify all steps were called
            mock_load.assert_called_once_with(mock_checkpoint_file, map_location='cpu')
            mock_load_model.assert_called_once()
            mock_convert.assert_called_once()
            assert result is not None
    
    @pytest.mark.integration 
    @pytest.mark.model
    def test_model_validation_pipeline(self, converter, mock_fastvlm_model, 
                                     sample_image_tensor, sample_text_tokens):
        """Test model validation after conversion."""
        with patch.object(converter, 'validate_model_output') as mock_validate:
            mock_validate.return_value = True
            
            # Test model validation
            is_valid = converter.validate_converted_model(
                pytorch_model=mock_fastvlm_model,
                coreml_model=MagicMock(),
                test_inputs={
                    "image": sample_image_tensor,
                    "text_tokens": sample_text_tokens
                }
            )
            
            mock_validate.assert_called_once()
            assert is_valid is True
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_performance_optimization_pipeline(self, converter, mock_fastvlm_model):
        """Test performance optimization pipeline."""
        with patch.object(converter, 'optimize_for_device') as mock_optimize:
            mock_optimize.return_value = mock_fastvlm_model
            
            # Test optimization
            optimized_model = converter.optimize_for_apple_neural_engine(
                model=mock_fastvlm_model,
                target_device="iPhone15Pro"
            )
            
            mock_optimize.assert_called_once()
            assert optimized_model is not None
    
    @pytest.mark.integration
    def test_batch_conversion_pipeline(self, converter, temp_output_dir):
        """Test batch conversion of multiple models."""
        model_configs = [
            {"name": "tiny", "quantization": "int4"},
            {"name": "base", "quantization": "mixed"},
            {"name": "large", "quantization": "int8"}
        ]
        
        with patch.object(converter, 'convert_to_coreml') as mock_convert:
            mock_convert.return_value = MagicMock()
            
            # Test batch conversion
            results = converter.batch_convert_models(
                checkpoint_dir="checkpoints/",
                output_dir=str(temp_output_dir),
                model_configs=model_configs
            )
            
            # Verify all models were processed
            assert len(results) == len(model_configs)
            assert mock_convert.call_count == len(model_configs)
    
    @pytest.mark.integration
    def test_error_handling_pipeline(self, converter, mock_fastvlm_model, temp_output_dir):
        """Test error handling in conversion pipeline."""
        output_path = temp_output_dir / "error_test.mlpackage"
        
        with patch('coremltools.convert') as mock_convert:
            # Simulate conversion error
            mock_convert.side_effect = Exception("Core ML conversion failed")
            
            # Test error handling
            with pytest.raises(Exception) as exc_info:
                converter.convert_to_coreml(
                    model=mock_fastvlm_model,
                    output_path=str(output_path)
                )
            
            assert "Core ML conversion failed" in str(exc_info.value)
    
    @pytest.mark.integration
    @pytest.mark.performance
    def test_conversion_performance_benchmark(self, converter, mock_fastvlm_model, 
                                            performance_benchmark, temp_output_dir):
        """Benchmark conversion performance."""
        output_path = temp_output_dir / "benchmark_model.mlpackage"
        
        with patch('coremltools.convert') as mock_convert:
            mock_convert.return_value = MagicMock()
            
            # Measure conversion time
            result, conversion_time = performance_benchmark.measure_time(
                converter.convert_to_coreml,
                model=mock_fastvlm_model,
                output_path=str(output_path)
            )
            
            # Verify performance metrics
            assert conversion_time > 0  # Should take some time
            assert result is not None
            
            # Log performance for analysis
            print(f"Conversion time: {conversion_time:.2f}ms")
    
    @pytest.mark.integration
    def test_memory_usage_during_conversion(self, converter, mock_fastvlm_model, 
                                          performance_benchmark, temp_output_dir):
        """Test memory usage during conversion."""
        output_path = temp_output_dir / "memory_test.mlpackage"
        
        with patch('coremltools.convert') as mock_convert:
            mock_convert.return_value = MagicMock()
            
            # Measure memory before conversion
            memory_before = performance_benchmark.measure_memory()
            
            # Perform conversion
            converter.convert_to_coreml(
                model=mock_fastvlm_model,
                output_path=str(output_path)
            )
            
            # Measure memory after conversion
            memory_after = performance_benchmark.measure_memory()
            
            # Verify memory usage is reasonable
            memory_increase = memory_after - memory_before
            assert memory_increase >= 0  # Memory should not decrease
            
            print(f"Memory increase during conversion: {memory_increase:.2f}MB")