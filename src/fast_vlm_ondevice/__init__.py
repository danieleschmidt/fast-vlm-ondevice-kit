"""
Fast VLM On-Device Kit

Production-ready Vision-Language Models for mobile devices.
Optimized for Apple Neural Engine with <250ms inference.
"""

__version__ = "1.0.0"
__author__ = "Daniel Schmidt"

from .converter import FastVLMConverter
from .quantization import QuantizationConfig

__all__ = ["FastVLMConverter", "QuantizationConfig"]