"""
Test fixtures for model-related testing.
Provides mock models, sample data, and testing utilities.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Dict, Any, Optional, Tuple


class MockFastVLMModel:
    """Mock FastVLM model for testing without actual model files."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            "vision_encoder_dim": 768,
            "text_encoder_dim": 512,
            "fusion_dim": 512,
            "vocab_size": 32000,
            "max_seq_length": 77,
            "image_size": 336
        }
        self.eval_mode = True
    
    def eval(self):
        self.eval_mode = True
        return self
    
    def forward(self, images: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        # Mock output - token probabilities
        return torch.randn(batch_size, self.config["max_seq_length"], self.config["vocab_size"])
    
    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        return torch.randn(batch_size, 49, self.config["vision_encoder_dim"])  # 7x7 patches
    
    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = text_tokens.shape
        return torch.randn(batch_size, seq_len, self.config["text_encoder_dim"])
    
    def parameters(self):
        # Mock parameters for testing quantization
        yield torch.randn(1000, 768)  # Vision encoder weights
        yield torch.randn(512, 512)  # Text encoder weights
        yield torch.randn(512, 768)  # Fusion weights
        yield torch.randn(32000, 512)  # Decoder weights


class MockCoreMLModel:
    """Mock Core ML model for testing conversion pipeline."""
    
    def __init__(self, spec=None):
        self.spec = spec or {}
        self.saved_path = None
    
    def save(self, path: str):
        self.saved_path = path
        # Create mock .mlpackage directory structure
        model_path = Path(path)
        model_path.mkdir(parents=True, exist_ok=True)
        (model_path / "Manifest.json").write_text('{"fileFormatVersion": "1.0.0"}')
        (model_path / "Data").mkdir(exist_ok=True)
        (model_path / "Metadata").mkdir(exist_ok=True)
    
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Mock prediction output
        return {
            "answer_tokens": np.random.randint(0, 32000, (1, 32)),
            "confidence_scores": np.random.rand(1, 32)
        }


@pytest.fixture
def mock_fastvlm_model():
    """Provide a mock FastVLM PyTorch model."""
    return MockFastVLMModel()


@pytest.fixture
def mock_coreml_model():
    """Provide a mock Core ML model."""
    return MockCoreMLModel()


@pytest.fixture
def sample_model_config():
    """Provide sample model configuration."""
    return {
        "model_type": "fast-vlm-base",
        "vision_encoder": {
            "type": "mobilevit",
            "input_size": 336,
            "patch_size": 16,
            "hidden_dim": 768
        },
        "text_encoder": {
            "type": "clip",
            "vocab_size": 32000,
            "hidden_dim": 512,
            "max_length": 77
        },
        "fusion_module": {
            "type": "cross_attention",
            "hidden_dim": 512,
            "num_heads": 8
        },
        "decoder": {
            "type": "autoregressive",
            "vocab_size": 32000,
            "hidden_dim": 512
        }
    }


@pytest.fixture
def sample_quantization_config():
    """Provide sample quantization configuration."""
    from fast_vlm_ondevice.quantization import QuantizationConfig
    
    return QuantizationConfig(
        vision_encoder="int4",
        text_encoder="int8", 
        fusion_layers="fp16",
        decoder="int4",
        calibration_samples=100
    )


@pytest.fixture
def sample_image_tensor():
    """Generate sample image tensor in expected format."""
    # RGB image tensor: (batch, channels, height, width)
    return torch.randn(1, 3, 336, 336)


@pytest.fixture
def sample_text_tokens():
    """Generate sample text token tensor."""
    # Random token IDs within vocab range
    return torch.randint(0, 32000, (1, 77))


@pytest.fixture
def sample_training_batch():
    """Generate a sample training batch."""
    batch_size = 4
    return {
        "images": torch.randn(batch_size, 3, 336, 336),
        "text_tokens": torch.randint(0, 32000, (batch_size, 77)),
        "answers": torch.randint(0, 32000, (batch_size, 32)),
        "attention_mask": torch.ones(batch_size, 77)
    }


@pytest.fixture
def sample_vqa_examples():
    """Provide sample VQA examples for testing."""
    return [
        {
            "image_path": "tests/data/sample_1.jpg",
            "question": "What color is the car in the image?", 
            "answer": "red",
            "question_id": 1001
        },
        {
            "image_path": "tests/data/sample_2.jpg",
            "question": "How many people are visible?",
            "answer": "three",
            "question_id": 1002
        },
        {
            "image_path": "tests/data/sample_3.jpg", 
            "question": "What is the weather like?",
            "answer": "sunny",
            "question_id": 1003
        }
    ]


@pytest.fixture
def mock_checkpoint_file(tmp_path):
    """Create a mock checkpoint file for testing."""
    checkpoint_path = tmp_path / "mock_model.pth"
    
    # Create mock state dict
    state_dict = {
        "vision_encoder.conv1.weight": torch.randn(64, 3, 7, 7),
        "vision_encoder.conv1.bias": torch.randn(64),
        "text_encoder.embedding.weight": torch.randn(32000, 512),
        "fusion.cross_attn.weight": torch.randn(512, 512),
        "decoder.lm_head.weight": torch.randn(32000, 512)
    }
    
    checkpoint = {
        "model_state_dict": state_dict,
        "config": {
            "model_type": "fast-vlm-base",
            "image_size": 336,
            "vocab_size": 32000
        },
        "training_args": {
            "learning_rate": 1e-4,
            "batch_size": 32
        },
        "epoch": 10,
        "global_step": 10000
    }
    
    torch.save(checkpoint, checkpoint_path)
    return str(checkpoint_path)


@pytest.fixture
def performance_test_data():
    """Generate data for performance testing."""
    return {
        "small_batch": {
            "images": torch.randn(1, 3, 336, 336),
            "text": torch.randint(0, 32000, (1, 77))
        },
        "medium_batch": {
            "images": torch.randn(4, 3, 336, 336), 
            "text": torch.randint(0, 32000, (4, 77))
        },
        "large_batch": {
            "images": torch.randn(8, 3, 336, 336),
            "text": torch.randint(0, 32000, (8, 77))
        }
    }


@pytest.fixture
def accuracy_test_dataset():
    """Generate dataset for accuracy testing."""
    # Small synthetic dataset for testing
    dataset = []
    for i in range(50):  # Small dataset for fast testing
        dataset.append({
            "image": torch.randn(3, 336, 336),
            "question": f"Test question {i}",
            "answer": f"test answer {i}",
            "question_id": i
        })
    return dataset


class ModelTestUtils:
    """Utility class for model testing."""
    
    @staticmethod
    def compare_model_outputs(output1: torch.Tensor, output2: torch.Tensor, 
                            tolerance: float = 1e-3) -> bool:
        """Compare model outputs within tolerance."""
        return torch.allclose(output1, output2, atol=tolerance, rtol=tolerance)
    
    @staticmethod
    def calculate_model_size(model) -> float:
        """Calculate model size in MB."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        return param_size / 1024 / 1024
    
    @staticmethod
    def count_parameters(model) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in model.parameters())
    
    @staticmethod
    def measure_inference_time(model, inputs: Dict[str, torch.Tensor], 
                             warmup_runs: int = 3, test_runs: int = 10) -> float:
        """Measure average inference time."""
        import time
        
        # Warmup runs
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = model(**inputs)
        
        # Measured runs
        start_time = time.perf_counter()
        for _ in range(test_runs):
            with torch.no_grad():
                _ = model(**inputs)
        end_time = time.perf_counter()
        
        return (end_time - start_time) / test_runs * 1000  # Return in milliseconds


@pytest.fixture
def model_test_utils():
    """Provide model testing utilities."""
    return ModelTestUtils()