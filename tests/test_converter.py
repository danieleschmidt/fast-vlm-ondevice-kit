"""
Tests for FastVLM converter functionality.
"""

import pytest
from src.fast_vlm_ondevice.converter import FastVLMConverter


class TestFastVLMConverter:
    """Test suite for FastVLMConverter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.converter = FastVLMConverter()
        
    def test_converter_initialization(self):
        """Test converter creates successfully."""
        assert self.converter is not None
        assert self.converter.model_size_mb == 0
        
    def test_load_pytorch_model_demo(self):
        """Test demo model loading."""
        result = self.converter.load_pytorch_model("dummy_path.pth")
        assert result is not None
        # Should create demo model when checkpoint doesn't exist
        assert hasattr(result, 'forward')
        
    @pytest.mark.slow
    def test_convert_to_coreml_demo(self):
        """Test Core ML conversion with demo model."""
        # Load demo model first
        model = self.converter.load_pytorch_model("dummy_path.pth")
        
        # Convert to CoreML - will work with demo model
        result = self.converter.convert_to_coreml(
            model=model,
            quantization="int4", 
            image_size=(336, 336)
        )
        assert result is not None
        assert self.converter.get_model_size_mb() > 0
        
    def test_get_model_size(self):
        """Test model size reporting."""
        size = self.converter.get_model_size_mb()
        assert isinstance(size, float)
        assert size >= 0