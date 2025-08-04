"""
Health checking utilities for FastVLM On-Device Kit.

Provides system status monitoring and validation.
"""

import os
import sys
import psutil
import platform
import logging
from typing import Dict, Any, Optional
from pathlib import Path

try:
    import torch
    import coremltools as ct
    from PIL import Image
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class HealthChecker:
    """System health checker for FastVLM deployment."""
    
    def __init__(self):
        """Initialize health checker."""
        self.minimum_ram_gb = 4.0
        self.minimum_disk_gb = 2.0
        
    def check_all(self) -> Dict[str, Dict[str, Any]]:
        """Run all health checks."""
        return {
            "system": self.check_system(),
            "dependencies": self.check_dependencies(),
            "hardware": self.check_hardware(),
            "storage": self.check_storage(),
            "model_compatibility": self.check_model_compatibility()
        }
    
    def check_system(self) -> Dict[str, Any]:
        """Check system compatibility."""
        try:
            system_info = {
                "platform": platform.system(),
                "platform_version": platform.version(),
                "python_version": sys.version,
                "architecture": platform.machine()
            }
            
            # Check for macOS (required for Core ML)
            is_macos = platform.system() == "Darwin"
            
            if is_macos:
                # Check macOS version (require 14.0+ for Core ML 7.0+)
                version_parts = platform.mac_ver()[0].split('.')
                major_version = int(version_parts[0])
                minor_version = int(version_parts[1]) if len(version_parts) > 1 else 0
                
                macos_compatible = major_version >= 14 or (major_version == 13 and minor_version >= 0)
                message = f"macOS {platform.mac_ver()[0]} detected"
                
                if not macos_compatible:
                    message += " (Warning: Core ML may have limited support)"
            else:
                macos_compatible = False
                message = f"{platform.system()} detected (Core ML not supported, PyTorch only)"
            
            return {
                "healthy": True,
                "compatible": is_macos and macos_compatible,
                "message": message,
                "details": system_info
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "compatible": False,
                "message": f"System check failed: {e}",
                "details": {}
            }
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check required dependencies."""
        try:
            missing_deps = []
            available_deps = {}
            
            # Check PyTorch
            if TORCH_AVAILABLE:
                available_deps["torch"] = torch.__version__
                available_deps["cuda_available"] = torch.cuda.is_available()
                available_deps["mps_available"] = torch.backends.mps.is_available()
            else:
                missing_deps.append("torch")
            
            # Check Core ML Tools
            try:
                import coremltools
                available_deps["coremltools"] = coremltools.__version__
            except ImportError:
                missing_deps.append("coremltools")
            
            # Check Transformers
            try:
                import transformers
                available_deps["transformers"] = transformers.__version__
            except ImportError:
                missing_deps.append("transformers")
            
            # Check PIL
            try:
                from PIL import Image
                available_deps["pillow"] = Image.__version__ if hasattr(Image, '__version__') else "available"
            except ImportError:
                missing_deps.append("pillow")
            
            # Check NumPy
            try:
                import numpy
                available_deps["numpy"] = numpy.__version__
            except ImportError:
                missing_deps.append("numpy")
            
            healthy = len(missing_deps) == 0
            message = "All dependencies available" if healthy else f"Missing: {', '.join(missing_deps)}"
            
            return {
                "healthy": healthy,
                "message": message,
                "available": available_deps,
                "missing": missing_deps
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Dependency check failed: {e}",
                "available": {},
                "missing": []
            }
    
    def check_hardware(self) -> Dict[str, Any]:
        """Check hardware requirements."""
        try:
            # Get memory info
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            # Get CPU info
            cpu_count = psutil.cpu_count(logical=True)
            cpu_freq = psutil.cpu_freq()
            
            # Check for Apple Silicon
            is_apple_silicon = platform.machine() in ["arm64", "Apple Silicon"]
            
            hardware_info = {
                "memory_gb": round(memory_gb, 2),
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "cpu_count": cpu_count,
                "cpu_freq_mhz": cpu_freq.current if cpu_freq else None,
                "apple_silicon": is_apple_silicon
            }
            
            # Check memory requirements
            memory_ok = memory_gb >= self.minimum_ram_gb
            
            message_parts = []
            if memory_ok:
                message_parts.append(f"Memory: {memory_gb:.1f}GB available")
            else:
                message_parts.append(f"Warning: Only {memory_gb:.1f}GB RAM (minimum {self.minimum_ram_gb}GB recommended)")
            
            if is_apple_silicon:
                message_parts.append("Apple Silicon detected (optimal performance)")
            
            return {
                "healthy": memory_ok,
                "message": ", ".join(message_parts),
                "details": hardware_info
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Hardware check failed: {e}",
                "details": {}
            }
    
    def check_storage(self) -> Dict[str, Any]:
        """Check storage requirements."""
        try:
            # Check current directory disk space
            disk_usage = psutil.disk_usage('.')
            free_gb = disk_usage.free / (1024**3)
            total_gb = disk_usage.total / (1024**3)
            
            storage_info = {
                "free_gb": round(free_gb, 2),
                "total_gb": round(total_gb, 2),
                "usage_percent": round((disk_usage.used / disk_usage.total) * 100, 1)
            }
            
            # Check disk space requirements
            storage_ok = free_gb >= self.minimum_disk_gb
            
            if storage_ok:
                message = f"Storage: {free_gb:.1f}GB free ({storage_info['usage_percent']}% used)"
            else:
                message = f"Warning: Only {free_gb:.1f}GB free (minimum {self.minimum_disk_gb}GB required)"
            
            return {
                "healthy": storage_ok,
                "message": message,
                "details": storage_info
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Storage check failed: {e}",
                "details": {}
            }
    
    def check_model_compatibility(self) -> Dict[str, Any]:
        """Check model format compatibility."""
        try:
            compatibility_info = {
                "pytorch_models": True,  # Always supported
                "coreml_models": False,
                "onnx_models": False
            }
            
            # Check Core ML support
            try:
                import coremltools
                compatibility_info["coreml_models"] = True
                compatibility_info["coreml_version"] = coremltools.__version__
            except ImportError:
                pass
            
            # Check ONNX support
            try:
                import onnx
                import onnxruntime
                compatibility_info["onnx_models"] = True
                compatibility_info["onnx_version"] = onnx.__version__
                compatibility_info["onnxruntime_version"] = onnxruntime.__version__
            except ImportError:
                pass
            
            supported_count = sum(1 for supported in compatibility_info.values() if isinstance(supported, bool) and supported)
            
            message = f"Model format support: {supported_count}/3 formats available"
            if compatibility_info["coreml_models"]:
                message += " (Core ML optimized)"
            
            return {
                "healthy": compatibility_info["pytorch_models"],  # Minimum requirement
                "message": message,
                "details": compatibility_info
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Model compatibility check failed: {e}",
                "details": {}
            }
    
    def check_model_file(self, model_path: str) -> Dict[str, Any]:
        """Check specific model file health."""
        try:
            path = Path(model_path)
            
            if not path.exists():
                return {
                    "healthy": False,
                    "message": f"Model file not found: {model_path}",
                    "details": {}
                }
            
            file_size_mb = path.stat().st_size / (1024**2)
            
            # Determine model type
            if path.suffix == '.pth':
                model_type = "PyTorch"
                # Try to load PyTorch model metadata
                try:
                    if TORCH_AVAILABLE:
                        checkpoint = torch.load(model_path, map_location='cpu')
                        if isinstance(checkpoint, dict):
                            keys = list(checkpoint.keys())[:5]  # First 5 keys
                        else:
                            keys = ["state_dict"]
                        details = {"keys": keys, "type": str(type(checkpoint))}
                    else:
                        details = {"error": "PyTorch not available"}
                except Exception as e:
                    details = {"error": str(e)}
                    
            elif path.suffix == '.mlpackage' or path.name.endswith('.mlpackage'):
                model_type = "Core ML"
                # Try to load Core ML model metadata
                try:
                    import coremltools as ct
                    model = ct.models.MLModel(model_path)
                    spec = model.get_spec()
                    details = {
                        "inputs": [input.name for input in spec.description.input],
                        "outputs": [output.name for output in spec.description.output]
                    }
                except Exception as e:
                    details = {"error": str(e)}
            else:
                model_type = "Unknown"
                details = {"extension": path.suffix}
            
            return {
                "healthy": True,
                "message": f"{model_type} model ({file_size_mb:.1f}MB)",
                "details": {
                    "path": str(path),
                    "size_mb": round(file_size_mb, 2),
                    "type": model_type,
                    **details
                }
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Model file check failed: {e}",
                "details": {}
            }


def quick_health_check() -> bool:
    """Quick health check returning only pass/fail status."""
    try:
        checker = HealthChecker()
        results = checker.check_all()
        
        # Check critical components
        critical_checks = ["dependencies", "hardware", "storage"]
        for check in critical_checks:
            if check in results and not results[check]["healthy"]:
                return False
        
        return True
        
    except Exception:
        return False