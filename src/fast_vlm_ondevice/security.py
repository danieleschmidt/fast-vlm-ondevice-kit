"""
Security utilities for FastVLM On-Device Kit.

Provides input validation, secure file handling, and security scanning.
"""

import os
import hashlib
import hmac
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import re
import json

try:
    import numpy as np
    from PIL import Image
    import torch
    SECURITY_DEPS = True
except ImportError:
    SECURITY_DEPS = False
    # Create fallback classes
    class Image:
        class Image:
            def convert(self, mode): return self
            def getdata(self): return []
            def save(self, *args, **kwargs): pass
        @staticmethod
        def new(mode, size, color=None): return Image.Image()
        @staticmethod
        def open(path): return Image.Image()
    
    class np:
        ndarray = object
        uint8 = object
        float32 = object
        float64 = object
        @staticmethod
        def clip(*args, **kwargs): return object

logger = logging.getLogger(__name__)


class InputValidator:
    """Validates and sanitizes model inputs."""
    
    def __init__(self):
        """Initialize input validator."""
        self.max_image_size = (4096, 4096)  # Max image dimensions
        self.max_image_file_size = 50 * 1024 * 1024  # 50MB max file size
        self.max_text_length = 10000  # Max text length
        self.allowed_image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        self.allowed_model_formats = {'.pth', '.mlpackage', '.onnx'}
        
    def validate_image_input(self, image_input: Union[str, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Validate image input for security and format compliance.
        
        Args:
            image_input: Image as file path, PIL Image, or numpy array
            
        Returns:
            Validation result dictionary
        """
        result = {
            "valid": False,
            "sanitized": None,
            "warnings": [],
            "errors": []
        }
        
        try:
            if isinstance(image_input, str):
                # File path input
                result.update(self._validate_image_file(image_input))
                
            elif isinstance(image_input, Image.Image):
                # PIL Image input
                result.update(self._validate_pil_image(image_input))
                
            elif isinstance(image_input, np.ndarray):
                # NumPy array input
                result.update(self._validate_numpy_image(image_input))
                
            else:
                result["errors"].append(f"Unsupported image input type: {type(image_input)}")
                
        except Exception as e:
            result["errors"].append(f"Image validation failed: {e}")
            logger.error(f"Image validation error: {e}")
            
        return result
    
    def _validate_image_file(self, filepath: str) -> Dict[str, Any]:
        """Validate image file."""
        result = {"valid": False, "sanitized": None, "warnings": [], "errors": []}
        
        try:
            path = Path(filepath)
            
            # Check file exists
            if not path.exists():
                result["errors"].append(f"Image file not found: {filepath}")
                return result
                
            # Check file extension
            if path.suffix.lower() not in self.allowed_image_formats:
                result["errors"].append(f"Unsupported image format: {path.suffix}")
                return result
                
            # Check file size
            file_size = path.stat().st_size
            if file_size > self.max_image_file_size:
                result["errors"].append(f"Image file too large: {file_size} bytes (max {self.max_image_file_size})")
                return result
                
            # Load and validate image content
            try:
                with Image.open(filepath) as img:
                    return self._validate_pil_image(img)
            except Exception as e:
                result["errors"].append(f"Invalid image file: {e}")
                return result
                
        except Exception as e:
            result["errors"].append(f"File validation failed: {e}")
            return result
    
    def _validate_pil_image(self, image: Image.Image) -> Dict[str, Any]:
        """Validate PIL Image."""
        result = {"valid": False, "sanitized": None, "warnings": [], "errors": []}
        
        try:
            # Check image dimensions
            width, height = image.size
            if width > self.max_image_size[0] or height > self.max_image_size[1]:
                result["errors"].append(f"Image too large: {width}x{height} (max {self.max_image_size[0]}x{self.max_image_size[1]})")
                return result
                
            # Check for suspicious metadata
            if hasattr(image, '_getexif') and image._getexif():
                result["warnings"].append("Image contains EXIF metadata (will be stripped)")
                
            # Sanitize image - remove metadata and convert to RGB
            sanitized = image.convert('RGB')
            
            # Remove any metadata
            sanitized_data = sanitized.getdata()
            clean_image = Image.new('RGB', sanitized.size)
            clean_image.putdata(list(sanitized_data))
            
            result["valid"] = True
            result["sanitized"] = clean_image
            
        except Exception as e:
            result["errors"].append(f"PIL image validation failed: {e}")
            
        return result
    
    def _validate_numpy_image(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Validate NumPy image array."""
        result = {"valid": False, "sanitized": None, "warnings": [], "errors": []}
        
        try:
            # Check array dimensions
            if image_array.ndim not in [2, 3]:
                result["errors"].append(f"Invalid image array dimensions: {image_array.ndim} (expected 2 or 3)")
                return result
                
            # Check array shape
            height, width = image_array.shape[:2]
            if width > self.max_image_size[0] or height > self.max_image_size[1]:
                result["errors"].append(f"Image too large: {width}x{height} (max {self.max_image_size[0]}x{self.max_image_size[1]})")
                return result
                
            # Check data type and range
            if image_array.dtype not in [np.uint8, np.float32, np.float64]:
                result["warnings"].append(f"Unusual image data type: {image_array.dtype}")
                
            # Sanitize array - ensure proper range and type
            if image_array.dtype == np.uint8:
                sanitized = image_array.copy()
            else:
                # Convert to uint8 range [0, 255]
                if image_array.max() <= 1.0:
                    sanitized = (image_array * 255).astype(np.uint8)
                else:
                    sanitized = np.clip(image_array, 0, 255).astype(np.uint8)
                    
            result["valid"] = True
            result["sanitized"] = sanitized
            
        except Exception as e:
            result["errors"].append(f"NumPy array validation failed: {e}")
            
        return result
    
    def validate_text_input(self, text: str) -> Dict[str, Any]:
        """Validate text input for security and format compliance.
        
        Args:
            text: Input text string
            
        Returns:
            Validation result dictionary
        """
        result = {
            "valid": False,
            "sanitized": None,
            "warnings": [],
            "errors": []
        }
        
        try:
            # Check text length
            if len(text) > self.max_text_length:
                result["errors"].append(f"Text too long: {len(text)} characters (max {self.max_text_length})")
                return result
                
            # Check for suspicious patterns
            suspicious_patterns = [
                r'<script[^>]*>',  # Script tags
                r'javascript:',    # JavaScript protocols
                r'data:text/html', # Data URLs
                r'<!--.*?-->',     # HTML comments
                r'eval\s*\(',      # eval() calls
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    result["warnings"].append(f"Suspicious pattern detected: {pattern}")
            
            # Sanitize text - remove control characters and normalize
            sanitized = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
            sanitized = sanitized.strip()
            
            # Check for empty result
            if not sanitized:
                result["errors"].append("Text input is empty after sanitization")
                return result
                
            result["valid"] = True
            result["sanitized"] = sanitized
            
        except Exception as e:
            result["errors"].append(f"Text validation failed: {e}")
            
        return result
    
    def validate_model_file(self, filepath: str) -> Dict[str, Any]:
        """Validate model file for security.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Validation result dictionary
        """
        result = {
            "valid": False,
            "file_info": {},
            "warnings": [],
            "errors": []
        }
        
        try:
            path = Path(filepath)
            
            # Check file exists
            if not path.exists():
                result["errors"].append(f"Model file not found: {filepath}")
                return result
                
            # Check file extension
            if path.suffix.lower() not in self.allowed_model_formats and not path.name.endswith('.mlpackage'):
                result["errors"].append(f"Unsupported model format: {path.suffix}")
                return result
                
            # Get file info
            stat = path.stat()
            result["file_info"] = {
                "size_bytes": stat.st_size,
                "size_mb": stat.st_size / (1024**2),
                "modified_time": stat.st_mtime,
                "permissions": oct(stat.st_mode)[-3:]
            }
            
            # Check file size (reasonable limits)
            max_model_size = 10 * 1024 * 1024 * 1024  # 10GB
            if stat.st_size > max_model_size:
                result["errors"].append(f"Model file too large: {stat.st_size} bytes")
                return result
                
            # Additional checks based on file type
            if path.suffix == '.pth':
                result.update(self._validate_pytorch_model(filepath))
            elif path.name.endswith('.mlpackage'):
                result.update(self._validate_coreml_model(filepath))
                
            if not result["errors"]:
                result["valid"] = True
                
        except Exception as e:
            result["errors"].append(f"Model file validation failed: {e}")
            
        return result
    
    def _validate_pytorch_model(self, filepath: str) -> Dict[str, Any]:
        """Validate PyTorch model file."""
        result = {"warnings": [], "errors": []}
        
        try:
            if not SECURITY_DEPS:
                result["warnings"].append("PyTorch not available for validation")
                return result
                
            # Try to load model metadata without executing code
            with open(filepath, 'rb') as f:
                # Check file header
                header = f.read(8)
                if not header.startswith(b'PK'):  # ZIP header for PyTorch files
                    result["warnings"].append("Unusual PyTorch file format")
                    
            # Try to load with restricted unpickler (if available)
            try:
                checkpoint = torch.load(filepath, map_location='cpu', weights_only=True)
                result["warnings"].append("Model loaded successfully with weights_only=True")
            except:
                # Fallback to regular loading (less secure)
                checkpoint = torch.load(filepath, map_location='cpu')
                result["warnings"].append("Model required full unpickling (potential security risk)")
                
        except Exception as e:
            result["errors"].append(f"PyTorch model validation failed: {e}")
            
        return result
    
    def _validate_coreml_model(self, filepath: str) -> Dict[str, Any]:
        """Validate Core ML model file."""
        result = {"warnings": [], "errors": []}
        
        try:
            path = Path(filepath)
            
            # Check if it's a proper mlpackage directory
            if not path.is_dir():
                result["errors"].append("Core ML model should be a .mlpackage directory")
                return result
                
            # Check for required files
            required_files = ['Manifest.json', 'Data', 'Metadata']
            for req_file in required_files:
                if not (path / req_file).exists():
                    result["errors"].append(f"Missing required Core ML file: {req_file}")
                    
            # Validate manifest if present
            manifest_path = path / 'Manifest.json'
            if manifest_path.exists():
                try:
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                    result["warnings"].append("Core ML manifest validated")
                except json.JSONDecodeError:
                    result["errors"].append("Invalid Core ML manifest JSON")
                    
        except Exception as e:
            result["errors"].append(f"Core ML model validation failed: {e}")
            
        return result


class SecureFileHandler:
    """Handles file operations securely."""
    
    def __init__(self, temp_dir: Optional[str] = None):
        """Initialize secure file handler.
        
        Args:
            temp_dir: Directory for temporary files (default: system temp)
        """
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pth', '.mlpackage', '.onnx'}
        
    def create_secure_temp_file(self, suffix: str = '', prefix: str = 'fastvlm_') -> str:
        """Create secure temporary file.
        
        Args:
            suffix: File suffix
            prefix: File prefix
            
        Returns:
            Path to temporary file
        """
        try:
            # Create temporary file with secure permissions
            fd, filepath = tempfile.mkstemp(
                suffix=suffix,
                prefix=prefix,
                dir=self.temp_dir
            )
            os.close(fd)
            
            # Set restrictive permissions (owner only)
            os.chmod(filepath, 0o600)
            
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to create secure temp file: {e}")
            raise
    
    def validate_file_path(self, filepath: str) -> bool:
        """Validate file path for security.
        
        Args:
            filepath: File path to validate
            
        Returns:
            True if path is safe
        """
        try:
            path = Path(filepath).resolve()
            
            # Check for path traversal attempts
            if '..' in str(path):
                logger.warning(f"Path traversal attempt detected: {filepath}")
                return False
                
            # Check file extension
            if path.suffix.lower() not in self.allowed_extensions and not path.name.endswith('.mlpackage'):
                logger.warning(f"Disallowed file extension: {path.suffix}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"File path validation failed: {e}")
            return False
    
    def compute_file_hash(self, filepath: str, algorithm: str = 'sha256') -> str:
        """Compute secure hash of file.
        
        Args:
            filepath: Path to file
            algorithm: Hash algorithm (sha256, sha512, md5)
            
        Returns:
            Hex digest of file hash
        """
        try:
            hash_obj = hashlib.new(algorithm)
            
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    hash_obj.update(chunk)
                    
            return hash_obj.hexdigest()
            
        except Exception as e:
            logger.error(f"File hash computation failed: {e}")
            raise
    
    def verify_file_integrity(self, filepath: str, expected_hash: str, algorithm: str = 'sha256') -> bool:
        """Verify file integrity using hash.
        
        Args:
            filepath: Path to file
            expected_hash: Expected hash value
            algorithm: Hash algorithm
            
        Returns:
            True if file integrity is verified
        """
        try:
            actual_hash = self.compute_file_hash(filepath, algorithm)
            return hmac.compare_digest(expected_hash.lower(), actual_hash.lower())
            
        except Exception as e:
            logger.error(f"File integrity verification failed: {e}")
            return False


class SecurityScanner:
    """Scans for security vulnerabilities and suspicious patterns."""
    
    def __init__(self):
        """Initialize security scanner."""
        self.vulnerability_patterns = {
            'code_injection': [
                r'eval\s*\(',
                r'exec\s*\(',
                r'__import__\s*\(',
                r'compile\s*\(',
                r'globals\s*\(\)',
                r'locals\s*\(\)',
            ],
            'file_access': [
                r'open\s*\(',
                r'file\s*\(',
                r'os\.system',
                r'subprocess\.',
                r'commands\.',
            ],
            'network_access': [
                r'urllib\.',
                r'requests\.',
                r'socket\.',
                r'http\.',
                r'ftp\.',
            ],
            'pickle_security': [
                r'pickle\.loads',
                r'cPickle\.loads',
                r'_pickle\.loads',
                r'marshal\.loads',
            ]
        }
    
    def scan_model_security(self, model_path: str) -> Dict[str, Any]:
        """Scan model for security vulnerabilities.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Security scan results
        """
        results = {
            "secure": True,
            "vulnerabilities": [],
            "warnings": [],
            "scan_timestamp": logger.time()
        }
        
        try:
            path = Path(model_path)
            
            if path.suffix == '.pth':
                results.update(self._scan_pytorch_model(model_path))
            elif path.name.endswith('.mlpackage'):
                results.update(self._scan_coreml_model(model_path))
            else:
                results["warnings"].append(f"Unknown model format: {path.suffix}")
                
        except Exception as e:
            results["vulnerabilities"].append(f"Security scan failed: {e}")
            results["secure"] = False
            
        return results
    
    def _scan_pytorch_model(self, model_path: str) -> Dict[str, Any]:
        """Scan PyTorch model for security issues."""
        results = {"vulnerabilities": [], "warnings": []}
        
        try:
            # Try to examine model without executing arbitrary code
            if SECURITY_DEPS:
                try:
                    # Use weights_only=True for safer loading
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
                    results["warnings"].append("Model loaded safely with weights_only=True")
                except:
                    results["vulnerabilities"].append("Model requires full unpickling (security risk)")
                    results["secure"] = False
                    
        except Exception as e:
            results["vulnerabilities"].append(f"PyTorch model scan failed: {e}")
            
        return results
    
    def _scan_coreml_model(self, model_path: str) -> Dict[str, Any]:
        """Scan Core ML model for security issues."""
        results = {"vulnerabilities": [], "warnings": []}
        
        try:
            path = Path(model_path)
            
            # Core ML models are generally safer than PyTorch
            # Check for unusual files in mlpackage
            if path.is_dir():
                for file in path.rglob('*'):
                    if file.is_file():
                        if file.suffix not in {'.json', '.dat', '.bin', '.plist'}:
                            results["warnings"].append(f"Unusual file in mlpackage: {file.name}")
                            
        except Exception as e:
            results["warnings"].append(f"Core ML model scan failed: {e}")
            
        return results


def setup_security_validation() -> Tuple[InputValidator, SecureFileHandler, SecurityScanner]:
    """Setup complete security validation stack.
    
    Returns:
        Tuple of (input_validator, file_handler, security_scanner)
    """
    input_validator = InputValidator()
    file_handler = SecureFileHandler()
    security_scanner = SecurityScanner()
    
    return input_validator, file_handler, security_scanner