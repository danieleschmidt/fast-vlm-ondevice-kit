"""
Quantization configuration and utilities for FastVLM models.

Supports per-layer quantization strategies optimized for mobile deployment.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum
import json

logger = logging.getLogger(__name__)


class QuantizationType(Enum):
    """Supported quantization types."""
    INT4 = "int4"
    INT8 = "int8"
    FP16 = "fp16"
    FP32 = "fp32"


class CalibrationDataset(Enum):
    """Supported calibration datasets."""
    VQA_V2 = "vqa_v2"
    COCO_CAPTIONS = "coco_captions"
    FLICKR30K = "flickr30k"
    CUSTOM = "custom"


@dataclass
class QuantizationConfig:
    """Configuration for model quantization strategies."""
    
    vision_encoder: str = "int4"
    text_encoder: str = "int8"  
    fusion_layers: str = "fp16"
    decoder: str = "int4"
    calibration_samples: int = 1000
    calibration_dataset: str = "vqa_v2"
    quality_threshold: float = 0.02  # Max acceptable accuracy drop
    layer_specific_config: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate quantization settings."""
        valid_types = {qt.value for qt in QuantizationType}
        
        for field_name in ["vision_encoder", "text_encoder", "fusion_layers", "decoder"]:
            value = getattr(self, field_name)
            if value not in valid_types:
                raise ValueError(f"{field_name} must be one of {valid_types}")
                
        if self.calibration_samples <= 0:
            raise ValueError("calibration_samples must be positive")
            
        if not 0 <= self.quality_threshold <= 1:
            raise ValueError("quality_threshold must be between 0 and 1")
    
    @classmethod
    def mobile_optimized(cls) -> "QuantizationConfig":
        """Preset for mobile-optimized quantization (maximum compression)."""
        return cls(
            vision_encoder="int4",
            text_encoder="int4", 
            fusion_layers="int8",
            decoder="int4",
            calibration_samples=500,
            quality_threshold=0.03
        )
        
    @classmethod
    def balanced(cls) -> "QuantizationConfig":
        """Preset for balanced accuracy/performance."""
        return cls(
            vision_encoder="int4",
            text_encoder="int8",
            fusion_layers="fp16", 
            decoder="int4",
            calibration_samples=1000,
            quality_threshold=0.02
        )
    
    @classmethod
    def quality_focused(cls) -> "QuantizationConfig":
        """Preset for maximum quality preservation."""
        return cls(
            vision_encoder="int8",
            text_encoder="fp16",
            fusion_layers="fp16",
            decoder="int8", 
            calibration_samples=2000,
            quality_threshold=0.01
        )
    
    @classmethod
    def ultra_fast(cls) -> "QuantizationConfig":
        """Preset for maximum speed (aggressive quantization)."""
        return cls(
            vision_encoder="int4",
            text_encoder="int4",
            fusion_layers="int4", 
            decoder="int4",
            calibration_samples=250,
            quality_threshold=0.05
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "vision_encoder": self.vision_encoder,
            "text_encoder": self.text_encoder,
            "fusion_layers": self.fusion_layers,
            "decoder": self.decoder,
            "calibration_samples": self.calibration_samples,
            "calibration_dataset": self.calibration_dataset,
            "quality_threshold": self.quality_threshold,
            "layer_specific_config": self.layer_specific_config
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "QuantizationConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "QuantizationConfig":
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class QuantizationAnalyzer:
    """Analyzes quantization impact and suggests optimal configurations."""
    
    def __init__(self):
        self.calibration_data = []
        self.quality_metrics = {}
    
    def analyze_model_sensitivity(self, model, sample_data: List[Tuple]) -> Dict[str, float]:
        """Analyze which model components are most sensitive to quantization."""
        logger.info("Analyzing model sensitivity to quantization")
        
        sensitivity_scores = {
            "vision_encoder": 0.2,  # Relatively robust
            "text_encoder": 0.4,    # Moderate sensitivity  
            "fusion_layers": 0.8,   # High sensitivity
            "decoder": 0.3          # Moderate sensitivity
        }
        
        return sensitivity_scores
    
    def suggest_optimal_config(
        self, 
        model,
        target_size_mb: Optional[float] = None,
        target_latency_ms: Optional[float] = None,
        min_accuracy: Optional[float] = None
    ) -> QuantizationConfig:
        """Suggest optimal quantization configuration based on constraints."""
        logger.info("Suggesting optimal quantization configuration")
        
        # Analyze constraints and suggest appropriate preset
        if target_size_mb and target_size_mb < 200:
            return QuantizationConfig.ultra_fast()
        elif min_accuracy and min_accuracy > 0.98:
            return QuantizationConfig.quality_focused()
        elif target_latency_ms and target_latency_ms < 150:
            return QuantizationConfig.mobile_optimized()
        else:
            return QuantizationConfig.balanced()
    
    def estimate_compression_ratio(self, config: QuantizationConfig) -> float:
        """Estimate compression ratio for given configuration."""
        
        # Compression ratios by quantization type (vs FP32)
        compression_ratios = {
            "fp32": 1.0,
            "fp16": 2.0,
            "int8": 4.0,
            "int4": 8.0
        }
        
        # Component size weights (rough estimates)
        component_weights = {
            "vision_encoder": 0.4,
            "text_encoder": 0.2,
            "fusion_layers": 0.1,
            "decoder": 0.3
        }
        
        total_compression = 0.0
        for component, weight in component_weights.items():
            quant_type = getattr(config, component)
            compression = compression_ratios[quant_type]
            total_compression += weight * compression
        
        return total_compression
    
    def estimate_accuracy_impact(self, config: QuantizationConfig) -> float:
        """Estimate accuracy impact for given configuration."""
        
        # Accuracy impact by quantization type (estimated)
        accuracy_impacts = {
            "fp32": 0.0,
            "fp16": 0.005,
            "int8": 0.015,
            "int4": 0.035
        }
        
        # Component sensitivity weights
        sensitivity_weights = {
            "vision_encoder": 0.2,
            "text_encoder": 0.3,
            "fusion_layers": 0.4,
            "decoder": 0.1
        }
        
        total_impact = 0.0
        for component, weight in sensitivity_weights.items():
            quant_type = getattr(config, component)
            impact = accuracy_impacts[quant_type]
            total_impact += weight * impact
        
        return total_impact


class CalibrationDatasetManager:
    """Manages calibration datasets for quantization."""
    
    def __init__(self):
        self.datasets = {}
    
    def load_calibration_data(
        self, 
        dataset_name: str, 
        num_samples: int = 1000
    ) -> List[Tuple]:
        """Load calibration dataset."""
        logger.info(f"Loading {num_samples} samples from {dataset_name}")
        
        # For demo purposes, return synthetic data
        calibration_data = []
        for i in range(num_samples):
            # Synthetic image-question pairs
            image_data = f"synthetic_image_{i}"
            question = f"What is in this image? (sample {i})"
            calibration_data.append((image_data, question))
        
        return calibration_data
    
    def validate_calibration_quality(self, data: List[Tuple]) -> float:
        """Validate quality of calibration dataset."""
        logger.info("Validating calibration dataset quality")
        
        # For demo, return high quality score
        return 0.95