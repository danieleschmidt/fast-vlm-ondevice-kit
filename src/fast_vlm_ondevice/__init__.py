"""
fast-vlm-ondevice: Lightweight Vision-Language Model toolkit for edge deployment.

Core components:
  - ImageEncoder: 3-layer ConvNet visual feature extractor
  - TextEncoder: 2-layer transformer text encoder (64-dim)
  - CrossModalAttention: visual tokens attending to text queries
  - TinyVLM: full image-text matching model
  - QuantizationSimulator: INT8 quantization simulation
  - EdgeBenchmark: latency, memory, and ops profiling
"""

__version__ = "0.1.0"

from .models import ImageEncoder, TextEncoder, CrossModalAttention, TinyVLM
from .quantization import QuantizationSimulator
from .benchmark import EdgeBenchmark

__all__ = [
    "ImageEncoder",
    "TextEncoder",
    "CrossModalAttention",
    "TinyVLM",
    "QuantizationSimulator",
    "EdgeBenchmark",
]
