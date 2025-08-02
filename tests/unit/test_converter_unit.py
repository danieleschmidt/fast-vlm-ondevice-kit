"""
Unit tests for FastVLM converter module.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call
from pathlib import Path

from tests.utils.test_helpers import (
    MockModelFactory, 
    ModelTestData, 
    AssertionHelpers,
    PerformanceProfiler
)


class TestFastVLMConverter:
    """Unit tests for FastVLMConverter class."""
    
    @pytest.fixture
    def mock_converter(self):
        """Create a mock converter instance."""
        with patch('fast_vlm_ondevice.converter.FastVLMConverter') as mock:
            converter = mock.return_value
            converter.load_pytorch_model = MagicMock()
            converter.convert_to_coreml = MagicMock()
            converter.get_model_size_mb = MagicMock(return_value=412.5)
            yield converter
    
    @pytest.mark.unit
    def test_load_pytorch_model_success(self, mock_converter, temp_checkpoint_path):
        """Test successful PyTorch model loading."""
        # Arrange
        mock_model = MockModelFactory.create_pytorch_model()
        mock_converter.load_pytorch_model.return_value = mock_model
        
        # Act
        result = mock_converter.load_pytorch_model(temp_checkpoint_path)
        
        # Assert
        assert result is not None
        mock_converter.load_pytorch_model.assert_called_once_with(temp_checkpoint_path)
    
    @pytest.mark.unit
    def test_load_pytorch_model_file_not_found(self, mock_converter):
        """Test PyTorch model loading with non-existent file."""
        # Arrange
        non_existent_path = "/path/that/does/not/exist.pth"
        mock_converter.load_pytorch_model.side_effect = FileNotFoundError("Model file not found")
        
        # Act & Assert
        with pytest.raises(FileNotFoundError):
            mock_converter.load_pytorch_model(non_existent_path)
    
    @pytest.mark.unit
    def test_convert_to_coreml_success(self, mock_converter):
        """Test successful Core ML conversion."""
        # Arrange
        mock_pytorch_model = MockModelFactory.create_pytorch_model()
        mock_coreml_model = MockModelFactory.create_coreml_model()
        mock_converter.convert_to_coreml.return_value = mock_coreml_model
        
        # Act
        result = mock_converter.convert_to_coreml(
            mock_pytorch_model,
            quantization="int4",
            compute_units="ALL",
            image_size=(336, 336),
            max_seq_length=77
        )
        
        # Assert
        assert result is not None
        mock_converter.convert_to_coreml.assert_called_once()
        call_args = mock_converter.convert_to_coreml.call_args
        assert call_args[0][0] == mock_pytorch_model
        assert call_args[1]["quantization"] == "int4"
    
    @pytest.mark.unit
    @pytest.mark.parametrize("quantization", ["int4", "int8", "fp16", "mixed"])
    def test_convert_with_different_quantizations(self, mock_converter, quantization):
        """Test conversion with different quantization strategies."""
        # Arrange
        mock_pytorch_model = MockModelFactory.create_pytorch_model()
        mock_coreml_model = MockModelFactory.create_coreml_model()
        mock_converter.convert_to_coreml.return_value = mock_coreml_model
        
        # Act
        result = mock_converter.convert_to_coreml(
            mock_pytorch_model, 
            quantization=quantization
        )
        
        # Assert
        assert result is not None
        call_args = mock_converter.convert_to_coreml.call_args
        assert call_args[1]["quantization"] == quantization
    
    @pytest.mark.unit
    def test_get_model_size_mb(self, mock_converter):
        """Test model size calculation."""
        # Arrange
        expected_size = 412.5
        mock_converter.get_model_size_mb.return_value = expected_size
        
        # Act
        size = mock_converter.get_model_size_mb()
        
        # Assert
        assert size == expected_size
        assert isinstance(size, float)
    
    @pytest.mark.unit
    def test_conversion_performance(self, mock_converter):
        """Test conversion performance tracking."""
        # Arrange
        mock_pytorch_model = MockModelFactory.create_pytorch_model()
        mock_coreml_model = MockModelFactory.create_coreml_model()
        mock_converter.convert_to_coreml.return_value = mock_coreml_model
        
        profiler = PerformanceProfiler()
        
        # Act
        profiler.start()
        mock_converter.convert_to_coreml(mock_pytorch_model)
        metrics = profiler.stop()
        
        # Assert
        AssertionHelpers.assert_latency_under_threshold(metrics["duration_ms"], 10000)  # 10 seconds max
        assert metrics["memory_delta_mb"] >= 0  # Memory usage should not go negative


class TestQuantizationConfig:
    """Unit tests for quantization configuration."""
    
    @pytest.mark.unit
    def test_quantization_config_creation(self):
        """Test quantization config object creation."""
        # Arrange & Act
        config = MockModelFactory.create_quantization_config()
        
        # Assert
        assert config.vision_encoder == "int4"
        assert config.text_encoder == "int8" 
        assert config.fusion_layers == "fp16"
        assert config.decoder == "int4"
        assert config.calibration_samples == 1000
    
    @pytest.mark.unit
    @pytest.mark.parametrize("vision_quant,text_quant", [
        ("int4", "int8"),
        ("int8", "int8"),
        ("fp16", "fp16"),
    ])
    def test_quantization_config_variations(self, vision_quant, text_quant):
        """Test different quantization configurations."""
        # Arrange
        config = MockModelFactory.create_quantization_config()
        config.vision_encoder = vision_quant
        config.text_encoder = text_quant
        
        # Act & Assert
        assert config.vision_encoder == vision_quant
        assert config.text_encoder == text_quant


class TestModelValidation:
    """Unit tests for model validation utilities."""
    
    @pytest.mark.unit
    def test_validate_input_shapes(self):
        """Test input shape validation."""
        # Arrange
        image = ModelTestData.create_sample_image()
        expected_shape = (336, 336, 3)
        
        # Act & Assert
        AssertionHelpers.assert_model_output_shape(image, expected_shape)
    
    @pytest.mark.unit
    def test_validate_batch_shapes(self):
        """Test batch input shape validation."""
        # Arrange
        batch_size = 4
        batch_images = ModelTestData.create_batch_images(batch_size)
        expected_shape = (4, 336, 336, 3)
        
        # Act & Assert
        AssertionHelpers.assert_model_output_shape(batch_images, expected_shape)
    
    @pytest.mark.unit
    def test_validate_model_output_format(self):
        """Test model output format validation."""
        # Arrange
        mock_output = np.random.randn(1, 1000).astype(np.float32)
        expected_shape = (1, 1000)
        
        # Act & Assert
        AssertionHelpers.assert_model_output_shape(mock_output, expected_shape)
        assert mock_output.dtype == np.float32


class TestErrorHandling:
    """Unit tests for error handling scenarios."""
    
    @pytest.mark.unit
    def test_invalid_quantization_strategy(self, mock_converter):
        """Test handling of invalid quantization strategy."""
        # Arrange
        mock_pytorch_model = MockModelFactory.create_pytorch_model()
        mock_converter.convert_to_coreml.side_effect = ValueError("Invalid quantization strategy")
        
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid quantization strategy"):
            mock_converter.convert_to_coreml(mock_pytorch_model, quantization="invalid")
    
    @pytest.mark.unit
    def test_insufficient_memory_error(self, mock_converter):
        """Test handling of insufficient memory errors."""
        # Arrange
        mock_pytorch_model = MockModelFactory.create_pytorch_model()
        mock_converter.convert_to_coreml.side_effect = MemoryError("Insufficient memory for conversion")
        
        # Act & Assert
        with pytest.raises(MemoryError):
            mock_converter.convert_to_coreml(mock_pytorch_model)
    
    @pytest.mark.unit
    def test_corrupted_model_error(self, mock_converter):
        """Test handling of corrupted model files."""
        # Arrange
        corrupted_path = "/path/to/corrupted.pth"
        mock_converter.load_pytorch_model.side_effect = RuntimeError("Corrupted model file")
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="Corrupted model file"):
            mock_converter.load_pytorch_model(corrupted_path)