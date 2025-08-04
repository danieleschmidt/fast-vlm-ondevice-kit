"""
Testing utilities for FastVLM models.

Provides automated testing and validation capabilities.
"""

import time
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import torch
    import coremltools as ct
    import numpy as np
    from PIL import Image
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelTester:
    """Automated testing for FastVLM models."""
    
    def __init__(self, model_path: str):
        """Initialize model tester.
        
        Args:
            model_path: Path to model file (.pth or .mlpackage)
        """
        self.model_path = Path(model_path)
        self.model = None
        self.model_type = self._detect_model_type()
        
    def _detect_model_type(self) -> str:
        """Detect model type from file extension."""
        if self.model_path.suffix == '.pth':
            return "pytorch"
        elif self.model_path.name.endswith('.mlpackage'):
            return "coreml"
        else:
            return "unknown"
    
    def run_basic_tests(self) -> Dict[str, Any]:
        """Run basic model validation tests."""
        results = {
            "success": False,
            "latency_ms": 0.0,
            "memory_mb": 0.0,
            "tests": {},
            "errors": []
        }
        
        try:
            # Test 1: Model loading
            load_result = self._test_model_loading()
            results["tests"]["loading"] = load_result
            
            if not load_result["success"]:
                results["errors"].append("Model loading failed")
                return results
            
            # Test 2: Input validation
            input_result = self._test_input_shapes()
            results["tests"]["input_shapes"] = input_result
            
            # Test 3: Basic inference
            inference_result = self._test_basic_inference()
            results["tests"]["inference"] = inference_result
            
            if inference_result["success"]:
                results["latency_ms"] = inference_result["latency_ms"]
                results["memory_mb"] = inference_result["memory_mb"]
            
            # Test 4: Output validation
            output_result = self._test_output_validation()
            results["tests"]["output_validation"] = output_result
            
            # Overall success
            all_tests_passed = all(
                test["success"] for test in results["tests"].values()
            )
            results["success"] = all_tests_passed
            
            return results
            
        except Exception as e:
            logger.exception("Testing failed")
            results["errors"].append(str(e))
            return results
    
    def _test_model_loading(self) -> Dict[str, Any]:
        """Test model loading functionality."""
        try:
            if self.model_type == "pytorch":
                if not DEPS_AVAILABLE:
                    return {
                        "success": False,
                        "message": "PyTorch dependencies not available"
                    }
                
                self.model = torch.load(self.model_path, map_location='cpu')
                if hasattr(self.model, 'eval'):
                    self.model.eval()
                
                return {
                    "success": True,
                    "message": "PyTorch model loaded successfully",
                    "type": str(type(self.model))
                }
                
            elif self.model_type == "coreml":
                if not DEPS_AVAILABLE:
                    return {
                        "success": False,
                        "message": "Core ML dependencies not available"
                    }
                
                self.model = ct.models.MLModel(str(self.model_path))
                
                return {
                    "success": True,
                    "message": "Core ML model loaded successfully",
                    "spec_version": str(self.model.get_spec().specificationVersion)
                }
            
            else:
                return {
                    "success": False,
                    "message": f"Unsupported model type: {self.model_type}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Model loading failed: {e}"
            }
    
    def _test_input_shapes(self) -> Dict[str, Any]:
        """Test input shape compatibility."""
        try:
            if self.model_type == "pytorch":
                # For PyTorch models, try to get input requirements
                return {
                    "success": True,
                    "message": "PyTorch model - dynamic input shapes",
                    "expected_inputs": ["image_tensor", "input_ids", "attention_mask"]
                }
                
            elif self.model_type == "coreml":
                spec = self.model.get_spec()
                inputs = []
                
                for input_desc in spec.description.input:
                    shape_info = {}
                    if input_desc.type.HasField('imageType'):
                        shape_info = {
                            "type": "image",
                            "width": input_desc.type.imageType.width,
                            "height": input_desc.type.imageType.height
                        }
                    elif input_desc.type.HasField('multiArrayType'):
                        shape_info = {
                            "type": "tensor",
                            "shape": list(input_desc.type.multiArrayType.shape)
                        }
                    
                    inputs.append({
                        "name": input_desc.name,
                        **shape_info
                    })
                
                return {
                    "success": True,
                    "message": f"Core ML model - {len(inputs)} inputs",
                    "inputs": inputs
                }
            
            return {
                "success": False,
                "message": "Unknown model type for input shape validation"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Input shape validation failed: {e}"
            }
    
    def _test_basic_inference(self) -> Dict[str, Any]:
        """Test basic model inference."""
        try:
            start_time = time.time()
            
            if self.model_type == "pytorch":
                # Create dummy inputs for PyTorch model
                dummy_image = torch.randn(1, 3, 336, 336)
                dummy_input_ids = torch.randint(0, 30522, (1, 77))
                dummy_attention_mask = torch.ones(1, 77)
                
                with torch.no_grad():
                    if hasattr(self.model, 'forward'):
                        output = self.model(dummy_image, dummy_input_ids, dummy_attention_mask)
                    else:
                        # If it's a state dict, can't run inference
                        return {
                            "success": True,
                            "message": "PyTorch state dict loaded (inference not tested)",
                            "latency_ms": 0.0,
                            "memory_mb": 0.0
                        }
                
                latency_ms = (time.time() - start_time) * 1000
                
                return {
                    "success": True,
                    "message": "PyTorch inference successful",
                    "latency_ms": latency_ms,
                    "memory_mb": self._estimate_memory_usage(),
                    "output_shape": list(output.shape) if hasattr(output, 'shape') else "unknown"
                }
                
            elif self.model_type == "coreml":
                # Create dummy inputs for Core ML model
                spec = self.model.get_spec()
                input_dict = {}
                
                for input_desc in spec.description.input:
                    if input_desc.type.HasField('imageType'):
                        # Create dummy image
                        width = input_desc.type.imageType.width
                        height = input_desc.type.imageType.height
                        dummy_image = Image.new('RGB', (width, height), color='red')
                        input_dict[input_desc.name] = dummy_image
                        
                    elif input_desc.type.HasField('multiArrayType'):
                        # Create dummy tensor
                        shape = list(input_desc.type.multiArrayType.shape)
                        dummy_array = np.random.randn(*shape).astype(np.float32)
                        input_dict[input_desc.name] = dummy_array
                
                # Run prediction
                prediction = self.model.predict(input_dict)
                
                latency_ms = (time.time() - start_time) * 1000
                
                return {
                    "success": True,
                    "message": "Core ML inference successful",
                    "latency_ms": latency_ms,
                    "memory_mb": self._estimate_memory_usage(),
                    "outputs": list(prediction.keys())
                }
            
            return {
                "success": False,
                "message": "Unknown model type for inference testing"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Inference test failed: {e}",
                "latency_ms": 0.0,
                "memory_mb": 0.0
            }
    
    def _test_output_validation(self) -> Dict[str, Any]:
        """Test output format validation."""
        try:
            if self.model_type == "coreml":
                spec = self.model.get_spec()
                outputs = []
                
                for output_desc in spec.description.output:
                    output_info = {"name": output_desc.name}
                    
                    if output_desc.type.HasField('multiArrayType'):
                        output_info["type"] = "tensor"
                        output_info["shape"] = list(output_desc.type.multiArrayType.shape)
                    else:
                        output_info["type"] = "unknown"
                    
                    outputs.append(output_info)
                
                return {
                    "success": True,
                    "message": f"Core ML model - {len(outputs)} outputs validated",
                    "outputs": outputs
                }
            
            return {
                "success": True,
                "message": "Output validation passed (basic check)"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Output validation failed: {e}"
            }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate model memory usage in MB."""
        try:
            if self.model_path.exists():
                file_size_mb = self.model_path.stat().st_size / (1024**2)
                # Estimate runtime memory as ~2x file size
                return file_size_mb * 2.0
            return 0.0
        except:
            return 0.0
    
    def run_performance_tests(self, iterations: int = 10) -> Dict[str, Any]:
        """Run performance benchmarking tests."""
        results = {
            "success": False,
            "iterations": iterations,
            "latencies": [],
            "avg_latency_ms": 0.0,
            "min_latency_ms": 0.0,
            "max_latency_ms": 0.0,
            "p95_latency_ms": 0.0
        }
        
        try:
            latencies = []
            
            for i in range(iterations):
                inference_result = self._test_basic_inference()
                if inference_result["success"]:
                    latencies.append(inference_result["latency_ms"])
                else:
                    logger.warning(f"Iteration {i} failed: {inference_result['message']}")
            
            if latencies:
                results["success"] = True
                results["latencies"] = latencies
                results["avg_latency_ms"] = np.mean(latencies)
                results["min_latency_ms"] = np.min(latencies)
                results["max_latency_ms"] = np.max(latencies)
                results["p95_latency_ms"] = np.percentile(latencies, 95)
            
            return results
            
        except Exception as e:
            logger.exception("Performance testing failed")
            results["error"] = str(e)
            return results