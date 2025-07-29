"""
Quantization configuration and utilities for FastVLM models.

Supports per-layer quantization strategies optimized for mobile deployment.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class QuantizationConfig:
    """Configuration for model quantization strategies."""
    
    vision_encoder: str = "int4"
    text_encoder: str = "int8"  
    fusion_layers: str = "fp16"
    decoder: str = "int4"
    calibration_samples: int = 1000
    
    def __post_init__(self):
        """Validate quantization settings."""
        valid_types = {"int4", "int8", "fp16", "fp32"}
        
        for field_name in ["vision_encoder", "text_encoder", "fusion_layers", "decoder"]:
            value = getattr(self, field_name)
            if value not in valid_types:
                raise ValueError(f"{field_name} must be one of {valid_types}")
                
        if self.calibration_samples <= 0:
            raise ValueError("calibration_samples must be positive")
            
    @classmethod
    def mobile_optimized(cls) -> "QuantizationConfig":
        """Preset for mobile-optimized quantization."""
        return cls(
            vision_encoder="int4",
            text_encoder="int4", 
            fusion_layers="int8",
            decoder="int4",
            calibration_samples=500
        )
        
    @classmethod
    def balanced(cls) -> "QuantizationConfig":
        """Preset for balanced accuracy/performance."""
        return cls(
            vision_encoder="int4",
            text_encoder="int8",
            fusion_layers="fp16", 
            decoder="int4",
            calibration_samples=1000
        )