"""
Comprehensive Validation Framework for FastVLM On-Device Kit.

Provides input validation, model integrity checks, runtime monitoring,
and safety constraints for production deployment.
"""

import logging
import json
import time
import hashlib
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
import functools
import inspect

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Create fallback numpy-like functionality
    class np:
        @staticmethod
        def isnan(x):
            try:
                return x != x  # NaN != NaN is True
            except:
                return False
        
        @staticmethod
        def isinf(x):
            try:
                return abs(x) == float('inf')
            except:
                return False
        
        @staticmethod
        def any(arr):
            try:
                return any(arr)
            except:
                return False
        
        ndarray = object

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"          # Maximum validation, slower performance
    BALANCED = "balanced"      # Standard validation for production
    MINIMAL = "minimal"        # Basic validation for performance-critical paths
    DISABLED = "disabled"      # No validation (use with caution)


class ErrorSeverity(Enum):
    """Error severity classifications."""
    CRITICAL = "critical"      # System cannot continue safely
    ERROR = "error"           # Operation failed but system can recover
    WARNING = "warning"       # Issue detected but operation can continue
    INFO = "info"            # Informational message


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    valid: bool
    severity: ErrorSeverity = ErrorSeverity.INFO
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp
        }


@dataclass
class ValidationConfig:
    """Configuration for validation behavior."""
    level: ValidationLevel = ValidationLevel.BALANCED
    timeout_seconds: float = 30.0
    max_memory_mb: float = 2048.0
    enable_runtime_checks: bool = True
    strict_type_checking: bool = True
    validate_numerical_stability: bool = True
    check_input_bounds: bool = True
    monitor_performance: bool = True


class InputValidator:
    """Comprehensive input validation for all system components."""
    
    def __init__(self, config: ValidationConfig = None):
        """Initialize input validator."""
        self.config = config or ValidationConfig()
        self._validation_cache = {}
        self._cache_lock = threading.Lock()
        
    def validate_image_input(self, image_data: Any) -> ValidationResult:
        """Validate image input data."""
        try:
            if self.config.level == ValidationLevel.DISABLED:
                return ValidationResult(valid=True)
            
            # Check if image_data exists
            if image_data is None:
                return ValidationResult(
                    valid=False,
                    severity=ErrorSeverity.ERROR,
                    message="Image data is None",
                    details={"input_type": type(image_data)}
                )
            
            # Type checking
            if self.config.strict_type_checking:
                if not isinstance(image_data, (np.ndarray, list, tuple)):
                    return ValidationResult(
                        valid=False,
                        severity=ErrorSeverity.ERROR,
                        message="Invalid image data type",
                        details={"expected": "numpy array or list", "actual": type(image_data)}
                    )
            
            # Convert to numpy array if needed
            if isinstance(image_data, (list, tuple)):
                image_data = np.array(image_data)
            
            # Shape validation
            if hasattr(image_data, 'shape'):
                shape = image_data.shape
                
                # Check minimum dimensions
                if len(shape) < 2:
                    return ValidationResult(
                        valid=False,
                        severity=ErrorSeverity.ERROR,
                        message="Image must have at least 2 dimensions",
                        details={"shape": shape}
                    )
                
                # Check reasonable image sizes
                if self.config.check_input_bounds:
                    max_pixels = 4096 * 4096  # 16MP max
                    total_pixels = np.prod(shape[:2])
                    
                    if total_pixels > max_pixels:
                        return ValidationResult(
                            valid=False,
                            severity=ErrorSeverity.ERROR,
                            message="Image too large",
                            details={"shape": shape, "max_pixels": max_pixels}
                        )
                    
                    if total_pixels < 32 * 32:  # Minimum reasonable size
                        return ValidationResult(
                            valid=False,
                            severity=ErrorSeverity.WARNING,
                            message="Image very small, may affect accuracy",
                            details={"shape": shape}
                        )
                
                # Check channel dimension
                if len(shape) >= 3:
                    channels = shape[-1] if len(shape) == 3 else shape[1]
                    if channels not in [1, 3, 4]:
                        return ValidationResult(
                            valid=False,
                            severity=ErrorSeverity.WARNING,
                            message="Unusual number of channels",
                            details={"channels": channels, "shape": shape}
                        )
            
            # Numerical validation
            if self.config.validate_numerical_stability and hasattr(image_data, 'dtype'):
                if np.issubdtype(image_data.dtype, np.floating):
                    if np.any(np.isnan(image_data)):
                        return ValidationResult(
                            valid=False,
                            severity=ErrorSeverity.ERROR,
                            message="Image contains NaN values",
                            details={"nan_count": np.sum(np.isnan(image_data))}
                        )
                    
                    if np.any(np.isinf(image_data)):
                        return ValidationResult(
                            valid=False,
                            severity=ErrorSeverity.ERROR,
                            message="Image contains infinite values",
                            details={"inf_count": np.sum(np.isinf(image_data))}
                        )
                
                # Check value ranges
                if self.config.check_input_bounds:
                    min_val, max_val = np.min(image_data), np.max(image_data)
                    
                    # Common image ranges
                    if np.issubdtype(image_data.dtype, np.floating):
                        if min_val < -10.0 or max_val > 10.0:
                            return ValidationResult(
                                valid=False,
                                severity=ErrorSeverity.WARNING,
                                message="Unusual image value range for floating point",
                                details={"min": float(min_val), "max": float(max_val)}
                            )
                    elif np.issubdtype(image_data.dtype, np.integer):
                        if min_val < 0 or max_val > 255:
                            return ValidationResult(
                                valid=False,
                                severity=ErrorSeverity.WARNING,
                                message="Unusual image value range for integer",
                                details={"min": int(min_val), "max": int(max_val)}
                            )
            
            return ValidationResult(
                valid=True,
                message="Image validation passed",
                details={"shape": getattr(image_data, 'shape', None)}
            )
            
        except Exception as e:
            return ValidationResult(
                valid=False,
                severity=ErrorSeverity.CRITICAL,
                message=f"Image validation failed with exception: {e}",
                details={"exception": str(e), "exception_type": type(e).__name__}
            )
    
    def validate_text_input(self, text_data: Any) -> ValidationResult:
        """Validate text input data."""
        try:
            if self.config.level == ValidationLevel.DISABLED:
                return ValidationResult(valid=True)
            
            if text_data is None:
                return ValidationResult(
                    valid=False,
                    severity=ErrorSeverity.ERROR,
                    message="Text data is None"
                )
            
            # Type validation
            if self.config.strict_type_checking:
                if not isinstance(text_data, (str, list, np.ndarray)):
                    return ValidationResult(
                        valid=False,
                        severity=ErrorSeverity.ERROR,
                        message="Invalid text data type",
                        details={"expected": "string, list, or numpy array", "actual": type(text_data)}
                    )
            
            # String validation
            if isinstance(text_data, str):
                if len(text_data) == 0:
                    return ValidationResult(
                        valid=False,
                        severity=ErrorSeverity.WARNING,
                        message="Empty text string"
                    )
                
                if self.config.check_input_bounds:
                    if len(text_data) > 10000:  # Very long text
                        return ValidationResult(
                            valid=False,
                            severity=ErrorSeverity.WARNING,
                            message="Text is very long, may affect performance",
                            details={"length": len(text_data)}
                        )
                    
                    # Check for unusual characters
                    printable_ratio = sum(1 for c in text_data if c.isprintable()) / len(text_data)
                    if printable_ratio < 0.8:
                        return ValidationResult(
                            valid=False,
                            severity=ErrorSeverity.WARNING,
                            message="Text contains many non-printable characters",
                            details={"printable_ratio": printable_ratio}
                        )
            
            # Token/embedding validation
            elif isinstance(text_data, (list, np.ndarray)):
                if len(text_data) == 0:
                    return ValidationResult(
                        valid=False,
                        severity=ErrorSeverity.ERROR,
                        message="Empty text token list"
                    )
                
                if isinstance(text_data, np.ndarray):
                    if self.config.validate_numerical_stability:
                        if np.any(np.isnan(text_data)):
                            return ValidationResult(
                                valid=False,
                                severity=ErrorSeverity.ERROR,
                                message="Text embeddings contain NaN values"
                            )
                        
                        if np.any(np.isinf(text_data)):
                            return ValidationResult(
                                valid=False,
                                severity=ErrorSeverity.ERROR,
                                message="Text embeddings contain infinite values"
                            )
            
            return ValidationResult(
                valid=True,
                message="Text validation passed",
                details={"type": type(text_data).__name__, "length": len(text_data)}
            )
            
        except Exception as e:
            return ValidationResult(
                valid=False,
                severity=ErrorSeverity.CRITICAL,
                message=f"Text validation failed with exception: {e}",
                details={"exception": str(e)}
            )
    
    def validate_model_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate model configuration parameters."""
        try:
            if self.config.level == ValidationLevel.DISABLED:
                return ValidationResult(valid=True)
            
            if not isinstance(config, dict):
                return ValidationResult(
                    valid=False,
                    severity=ErrorSeverity.ERROR,
                    message="Model config must be a dictionary",
                    details={"actual_type": type(config)}
                )
            
            # Required fields validation
            required_fields = ["model_name", "version"]
            missing_fields = [field for field in required_fields if field not in config]
            
            if missing_fields:
                return ValidationResult(
                    valid=False,
                    severity=ErrorSeverity.ERROR,
                    message="Missing required configuration fields",
                    details={"missing_fields": missing_fields}
                )
            
            # Field type validation
            if self.config.strict_type_checking:
                type_checks = {
                    "model_name": str,
                    "version": str,
                    "quantization": str,
                    "batch_size": int,
                    "max_length": int,
                    "temperature": (int, float),
                    "top_k": int,
                    "top_p": (int, float)
                }
                
                type_errors = []
                for field, expected_type in type_checks.items():
                    if field in config and not isinstance(config[field], expected_type):
                        type_errors.append({
                            "field": field,
                            "expected": expected_type.__name__ if hasattr(expected_type, '__name__') else str(expected_type),
                            "actual": type(config[field]).__name__
                        })
                
                if type_errors:
                    return ValidationResult(
                        valid=False,
                        severity=ErrorSeverity.ERROR,
                        message="Configuration type validation failed",
                        details={"type_errors": type_errors}
                    )
            
            # Value range validation
            if self.config.check_input_bounds:
                range_checks = {
                    "batch_size": (1, 64),
                    "max_length": (1, 2048),
                    "temperature": (0.0, 2.0),
                    "top_k": (1, 100),
                    "top_p": (0.0, 1.0)
                }
                
                range_errors = []
                for field, (min_val, max_val) in range_checks.items():
                    if field in config:
                        value = config[field]
                        if not (min_val <= value <= max_val):
                            range_errors.append({
                                "field": field,
                                "value": value,
                                "min": min_val,
                                "max": max_val
                            })
                
                if range_errors:
                    return ValidationResult(
                        valid=False,
                        severity=ErrorSeverity.WARNING,
                        message="Configuration values outside recommended ranges",
                        details={"range_errors": range_errors}
                    )
            
            return ValidationResult(
                valid=True,
                message="Model configuration validation passed",
                details={"fields_validated": list(config.keys())}
            )
            
        except Exception as e:
            return ValidationResult(
                valid=False,
                severity=ErrorSeverity.CRITICAL,
                message=f"Configuration validation failed with exception: {e}",
                details={"exception": str(e)}
            )


class ModelIntegrityChecker:
    """Validates model integrity and consistency."""
    
    def __init__(self, config: ValidationConfig = None):
        """Initialize model integrity checker."""
        self.config = config or ValidationConfig()
        
    def validate_model_file(self, file_path: str) -> ValidationResult:
        """Validate model file integrity."""
        try:
            if self.config.level == ValidationLevel.DISABLED:
                return ValidationResult(valid=True)
            
            path = Path(file_path)
            
            # File existence check
            if not path.exists():
                return ValidationResult(
                    valid=False,
                    severity=ErrorSeverity.ERROR,
                    message="Model file does not exist",
                    details={"file_path": file_path}
                )
            
            # File size validation
            file_size = path.stat().st_size
            size_mb = file_size / (1024 * 1024)
            
            if file_size == 0:
                return ValidationResult(
                    valid=False,
                    severity=ErrorSeverity.ERROR,
                    message="Model file is empty",
                    details={"file_path": file_path}
                )
            
            if self.config.check_input_bounds:
                if size_mb > self.config.max_memory_mb:
                    return ValidationResult(
                        valid=False,
                        severity=ErrorSeverity.WARNING,
                        message="Model file very large, may exceed memory limits",
                        details={"size_mb": size_mb, "limit_mb": self.config.max_memory_mb}
                    )
            
            # File format validation based on extension
            suffix = path.suffix.lower()
            
            format_validators = {
                '.pth': self._validate_pytorch_file,
                '.pkl': self._validate_pickle_file,
                '.onnx': self._validate_onnx_file,
                '.mlpackage': self._validate_coreml_file,
                '.json': self._validate_json_file
            }
            
            validator = format_validators.get(suffix)
            if validator and self.config.level in [ValidationLevel.STRICT, ValidationLevel.BALANCED]:
                return validator(file_path)
            
            return ValidationResult(
                valid=True,
                message="Basic model file validation passed",
                details={"file_path": file_path, "size_mb": size_mb, "format": suffix}
            )
            
        except Exception as e:
            return ValidationResult(
                valid=False,
                severity=ErrorSeverity.CRITICAL,
                message=f"Model file validation failed: {e}",
                details={"file_path": file_path, "exception": str(e)}
            )
    
    def _validate_pytorch_file(self, file_path: str) -> ValidationResult:
        """Validate PyTorch model file."""
        try:
            import torch
            
            # Try to load the file
            checkpoint = torch.load(file_path, map_location='cpu')
            
            # Check for common PyTorch model structures
            if isinstance(checkpoint, dict):
                expected_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch', 'version']
                found_keys = [key for key in expected_keys if key in checkpoint]
                
                if not found_keys and 'state_dict' not in checkpoint:
                    # Might be a raw state dict
                    if not any(k.endswith('.weight') or k.endswith('.bias') for k in checkpoint.keys()):
                        return ValidationResult(
                            valid=False,
                            severity=ErrorSeverity.WARNING,
                            message="PyTorch file doesn't contain expected model structure",
                            details={"keys": list(checkpoint.keys())[:10]}  # First 10 keys
                        )
            
            return ValidationResult(
                valid=True,
                message="PyTorch model file validation passed",
                details={"structure_type": type(checkpoint).__name__}
            )
            
        except Exception as e:
            return ValidationResult(
                valid=False,
                severity=ErrorSeverity.ERROR,
                message=f"PyTorch model file validation failed: {e}",
                details={"exception": str(e)}
            )
    
    def _validate_json_file(self, file_path: str) -> ValidationResult:
        """Validate JSON configuration file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, dict):
                return ValidationResult(
                    valid=False,
                    severity=ErrorSeverity.ERROR,
                    message="JSON file must contain a dictionary",
                    details={"actual_type": type(data).__name__}
                )
            
            return ValidationResult(
                valid=True,
                message="JSON file validation passed",
                details={"keys": list(data.keys())}
            )
            
        except json.JSONDecodeError as e:
            return ValidationResult(
                valid=False,
                severity=ErrorSeverity.ERROR,
                message="Invalid JSON format",
                details={"json_error": str(e)}
            )
        except Exception as e:
            return ValidationResult(
                valid=False,
                severity=ErrorSeverity.ERROR,
                message=f"JSON validation failed: {e}",
                details={"exception": str(e)}
            )
    
    def _validate_onnx_file(self, file_path: str) -> ValidationResult:
        """Validate ONNX model file."""
        try:
            # Try basic file structure validation
            with open(file_path, 'rb') as f:
                header = f.read(8)
            
            # ONNX files should start with specific bytes
            if not header.startswith(b'\x08'):  # Simplified check
                return ValidationResult(
                    valid=False,
                    severity=ErrorSeverity.WARNING,
                    message="File may not be a valid ONNX model",
                    details={"header": header.hex()}
                )
            
            return ValidationResult(
                valid=True,
                message="ONNX file basic validation passed"
            )
            
        except Exception as e:
            return ValidationResult(
                valid=False,
                severity=ErrorSeverity.ERROR,
                message=f"ONNX validation failed: {e}",
                details={"exception": str(e)}
            )
    
    def _validate_coreml_file(self, file_path: str) -> ValidationResult:
        """Validate Core ML model file."""
        try:
            path = Path(file_path)
            
            # Core ML models are typically directories with .mlpackage extension
            if not path.is_dir():
                return ValidationResult(
                    valid=False,
                    severity=ErrorSeverity.ERROR,
                    message="Core ML model should be a directory",
                    details={"path_type": "file" if path.is_file() else "unknown"}
                )
            
            # Check for required Core ML structure
            required_files = ['Manifest.json', 'Data']
            missing_files = [f for f in required_files if not (path / f).exists()]
            
            if missing_files:
                return ValidationResult(
                    valid=False,
                    severity=ErrorSeverity.WARNING,
                    message="Core ML package missing expected files",
                    details={"missing_files": missing_files}
                )
            
            return ValidationResult(
                valid=True,
                message="Core ML model validation passed"
            )
            
        except Exception as e:
            return ValidationResult(
                valid=False,
                severity=ErrorSeverity.ERROR,
                message=f"Core ML validation failed: {e}",
                details={"exception": str(e)}
            )
    
    def _validate_pickle_file(self, file_path: str) -> ValidationResult:
        """Validate pickle file (with security considerations)."""
        try:
            import pickle
            
            # Security warning for pickle files
            with open(file_path, 'rb') as f:
                # Just check if it can be opened and has pickle header
                header = f.read(2)
                
            if not header.startswith(b'\x80'):  # Pickle protocol header
                return ValidationResult(
                    valid=False,
                    severity=ErrorSeverity.WARNING,
                    message="File may not be a valid pickle file",
                    details={"header": header.hex()}
                )
            
            return ValidationResult(
                valid=True,
                severity=ErrorSeverity.WARNING,  # Always warn about pickle
                message="Pickle file validation passed (SECURITY WARNING: Only load from trusted sources)",
                details={"security_warning": "Pickle files can execute arbitrary code"}
            )
            
        except Exception as e:
            return ValidationResult(
                valid=False,
                severity=ErrorSeverity.ERROR,
                message=f"Pickle validation failed: {e}",
                details={"exception": str(e)}
            )


class RuntimeValidator:
    """Validates runtime behavior and performance."""
    
    def __init__(self, config: ValidationConfig = None):
        """Initialize runtime validator."""
        self.config = config or ValidationConfig()
        self.performance_history = []
        self._lock = threading.Lock()
        
    def validate_inference_output(self, output: Any, expected_shape: Optional[Tuple] = None) -> ValidationResult:
        """Validate model inference output."""
        try:
            if self.config.level == ValidationLevel.DISABLED:
                return ValidationResult(valid=True)
            
            if output is None:
                return ValidationResult(
                    valid=False,
                    severity=ErrorSeverity.ERROR,
                    message="Inference output is None"
                )
            
            # Shape validation
            if expected_shape and hasattr(output, 'shape'):
                actual_shape = output.shape
                if actual_shape != expected_shape:
                    return ValidationResult(
                        valid=False,
                        severity=ErrorSeverity.ERROR,
                        message="Output shape mismatch",
                        details={"expected": expected_shape, "actual": actual_shape}
                    )
            
            # Numerical stability validation
            if self.config.validate_numerical_stability and hasattr(output, 'dtype'):
                if np.issubdtype(output.dtype, np.floating):
                    nan_count = np.sum(np.isnan(output))
                    inf_count = np.sum(np.isinf(output))
                    
                    if nan_count > 0:
                        return ValidationResult(
                            valid=False,
                            severity=ErrorSeverity.ERROR,
                            message="Output contains NaN values",
                            details={"nan_count": int(nan_count)}
                        )
                    
                    if inf_count > 0:
                        return ValidationResult(
                            valid=False,
                            severity=ErrorSeverity.ERROR,
                            message="Output contains infinite values",
                            details={"inf_count": int(inf_count)}
                        )
            
            return ValidationResult(
                valid=True,
                message="Inference output validation passed",
                details={"shape": getattr(output, 'shape', None)}
            )
            
        except Exception as e:
            return ValidationResult(
                valid=False,
                severity=ErrorSeverity.CRITICAL,
                message=f"Output validation failed: {e}",
                details={"exception": str(e)}
            )
    
    def monitor_performance(self, func: Callable) -> Callable:
        """Decorator to monitor function performance."""
        if not self.config.monitor_performance:
            return func
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                end_memory = self._get_memory_usage()
                
                # Record performance metrics
                with self._lock:
                    self.performance_history.append({
                        "function": func.__name__,
                        "execution_time": execution_time,
                        "memory_delta": end_memory - start_memory,
                        "timestamp": time.time(),
                        "success": True
                    })
                
                # Validate performance
                if execution_time > self.config.timeout_seconds:
                    logger.warning(f"Function {func.__name__} took {execution_time:.2f}s (>timeout)")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                with self._lock:
                    self.performance_history.append({
                        "function": func.__name__,
                        "execution_time": execution_time,
                        "timestamp": time.time(),
                        "success": False,
                        "error": str(e)
                    })
                
                raise
        
        return wrapper
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance monitoring summary."""
        with self._lock:
            if not self.performance_history:
                return {"no_data": True}
            
            successful_calls = [h for h in self.performance_history if h.get("success", False)]
            
            if not successful_calls:
                return {"no_successful_calls": True}
            
            execution_times = [h["execution_time"] for h in successful_calls]
            
            return {
                "total_calls": len(self.performance_history),
                "successful_calls": len(successful_calls),
                "success_rate": len(successful_calls) / len(self.performance_history),
                "avg_execution_time": np.mean(execution_times),
                "max_execution_time": np.max(execution_times),
                "min_execution_time": np.min(execution_times),
                "recent_calls": self.performance_history[-5:]  # Last 5 calls
            }


def create_validation_suite(level: ValidationLevel = ValidationLevel.BALANCED) -> Dict[str, Any]:
    """Create a comprehensive validation suite."""
    config = ValidationConfig(level=level)
    
    return {
        "input_validator": InputValidator(config),
        "model_integrity_checker": ModelIntegrityChecker(config),
        "runtime_validator": RuntimeValidator(config),
        "config": config
    }


def validate_system_health() -> ValidationResult:
    """Perform comprehensive system health validation."""
    try:
        health_checks = []
        
        # Memory check
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                health_checks.append("High memory usage detected")
        except ImportError:
            pass
        
        # Disk space check
        try:
            import shutil
            free_space = shutil.disk_usage(".").free / (1024**3)  # GB
            if free_space < 1.0:  # Less than 1GB free
                health_checks.append("Low disk space")
        except:
            pass
        
        # Python version check
        import sys
        if sys.version_info < (3, 8):
            health_checks.append("Python version may be too old")
        
        if health_checks:
            return ValidationResult(
                valid=False,
                severity=ErrorSeverity.WARNING,
                message="System health issues detected",
                details={"issues": health_checks}
            )
        
        return ValidationResult(
            valid=True,
            message="System health validation passed"
        )
        
    except Exception as e:
        return ValidationResult(
            valid=False,
            severity=ErrorSeverity.ERROR,
            message=f"System health check failed: {e}",
            details={"exception": str(e)}
        )


def safe_execute(func: Callable, *args, validation_config: ValidationConfig = None, **kwargs) -> Tuple[Any, List[ValidationResult]]:
    """Safely execute a function with comprehensive validation."""
    config = validation_config or ValidationConfig()
    validator_suite = create_validation_suite(config.level)
    validation_results = []
    
    try:
        # Pre-execution validation
        if config.enable_runtime_checks:
            system_health = validate_system_health()
            validation_results.append(system_health)
            
            if not system_health.valid and system_health.severity == ErrorSeverity.CRITICAL:
                return None, validation_results
        
        # Execute with monitoring
        runtime_validator = validator_suite["runtime_validator"]
        monitored_func = runtime_validator.monitor_performance(func)
        
        result = monitored_func(*args, **kwargs)
        
        # Post-execution validation
        if hasattr(result, 'shape') or hasattr(result, '__len__'):
            output_validation = runtime_validator.validate_inference_output(result)
            validation_results.append(output_validation)
        
        return result, validation_results
        
    except Exception as e:
        error_result = ValidationResult(
            valid=False,
            severity=ErrorSeverity.CRITICAL,
            message=f"Function execution failed: {e}",
            details={"function": func.__name__, "exception": str(e)}
        )
        validation_results.append(error_result)
        return None, validation_results