"""
Tests for quantization configuration.
"""

import pytest
from src.fast_vlm_ondevice.quantization import QuantizationConfig


class TestQuantizationConfig:
    """Test suite for QuantizationConfig class."""
    
    def test_default_config(self):
        """Test default quantization configuration."""
        config = QuantizationConfig()
        assert config.vision_encoder == "int4"
        assert config.text_encoder == "int8"
        assert config.fusion_layers == "fp16"
        assert config.decoder == "int4"
        assert config.calibration_samples == 1000
        
    def test_mobile_optimized_preset(self):
        """Test mobile-optimized preset."""
        config = QuantizationConfig.mobile_optimized()
        assert config.vision_encoder == "int4"
        assert config.text_encoder == "int4"
        assert config.fusion_layers == "int8"
        assert config.calibration_samples == 500
        
    def test_balanced_preset(self):
        """Test balanced preset."""
        config = QuantizationConfig.balanced()
        assert config.vision_encoder == "int4"
        assert config.text_encoder == "int8"
        assert config.fusion_layers == "fp16"
        assert config.calibration_samples == 1000
        
    def test_invalid_quantization_type(self):
        """Test validation of invalid quantization types."""
        with pytest.raises(ValueError, match="must be one of"):
            QuantizationConfig(vision_encoder="invalid")
            
    def test_invalid_calibration_samples(self):
        """Test validation of calibration samples."""
        with pytest.raises(ValueError, match="must be positive"):
            QuantizationConfig(calibration_samples=0)