"""
Test fixtures package for FastVLM On-Device Kit.
Provides shared test utilities, mock objects, and test data.
"""

from .models import (
    MockFastVLMModel,
    MockCoreMLModel,
    ModelTestUtils
)

__all__ = [
    'MockFastVLMModel',
    'MockCoreMLModel', 
    'ModelTestUtils'
]