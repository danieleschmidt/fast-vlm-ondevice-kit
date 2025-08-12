"""
Comprehensive input validation and sanitization for FastVLM.

Provides security validation, content filtering, and input sanitization
to protect against malicious inputs and ensure data integrity.
"""

import re
import logging
import hashlib
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import mimetypes

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    error_message: Optional[str] = None
    sanitized_input: Optional[Any] = None
    confidence: float = 1.0
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class TextValidator:
    """Validates and sanitizes text inputs."""
    
    def __init__(self):
        """Initialize text validator."""
        # Common malicious patterns
        self.malicious_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',                # JavaScript URLs
            r'vbscript:',                 # VBScript URLs
            r'data:.*base64',             # Base64 data URLs
            r'file:\/\/',                 # File URLs
            r'\\x[0-9a-fA-F]{2}',        # Hex encoding
            r'%[0-9a-fA-F]{2}',          # URL encoding
            r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]',  # Control characters
        ]
        
        # SQL injection patterns
        self.sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(\b(OR|AND)\s+['\"]?\d['\"]?\s*=\s*['\"]?\d)",
            r"['\"];?\s*(--|\#|\/\*)",
            r"(\b(WAITFOR|DELAY)\b)",
        ]
        
        # Command injection patterns
        self.command_patterns = [
            r"[;&|`$\(\){}[\]<>]",
            r"(\b(rm|cat|ls|pwd|whoami|id|ps|kill|chmod|chown)\b)",
            r"(\\x[0-9a-fA-F]{2})+",
        ]
        
        # Compile patterns for efficiency
        self.compiled_malicious = [re.compile(pattern, re.IGNORECASE | re.DOTALL) 
                                  for pattern in self.malicious_patterns]
        self.compiled_sql = [re.compile(pattern, re.IGNORECASE) 
                            for pattern in self.sql_patterns]
        self.compiled_command = [re.compile(pattern, re.IGNORECASE) 
                                for pattern in self.command_patterns]
    
    def validate_question(self, question: str) -> ValidationResult:
        """Validate a question input."""
        if not question or not isinstance(question, str):
            return ValidationResult(
                is_valid=False,
                error_message="Question must be a non-empty string"
            )
        
        # Length validation
        if len(question) > 1000:
            return ValidationResult(
                is_valid=False,
                error_message="Question exceeds maximum length of 1000 characters"
            )
        
        if len(question.strip()) < 3:
            return ValidationResult(
                is_valid=False,
                error_message="Question must be at least 3 characters long"
            )
        
        # Check for malicious patterns
        warnings = []
        sanitized = question.strip()
        
        # Check for malicious content
        for pattern in self.compiled_malicious:
            if pattern.search(question):
                return ValidationResult(
                    is_valid=False,
                    error_message="Question contains potentially malicious content"
                )
        
        # Check for SQL injection
        for pattern in self.compiled_sql:
            if pattern.search(question):
                return ValidationResult(
                    is_valid=False,
                    error_message="Question contains SQL injection patterns"
                )
        
        # Check for command injection
        for pattern in self.compiled_command:
            if pattern.search(question):
                warnings.append("Question contains shell command patterns")
        
        # Sanitize HTML entities
        sanitized = self._sanitize_html_entities(sanitized)
        
        # Check for excessive repetition
        if self._has_excessive_repetition(sanitized):
            warnings.append("Question contains excessive character repetition")
        
        return ValidationResult(
            is_valid=True,
            sanitized_input=sanitized,
            warnings=warnings,
            confidence=0.8 if warnings else 1.0
        )
    
    def validate_model_name(self, model_name: str) -> ValidationResult:
        """Validate model name input."""
        if not model_name or not isinstance(model_name, str):
            return ValidationResult(
                is_valid=False,
                error_message="Model name must be a non-empty string"
            )
        
        # Allow only alphanumeric, hyphens, underscores, and dots
        if not re.match(r'^[a-zA-Z0-9._-]+$', model_name):
            return ValidationResult(
                is_valid=False,
                error_message="Model name contains invalid characters"
            )
        
        if len(model_name) > 100:
            return ValidationResult(
                is_valid=False,
                error_message="Model name exceeds maximum length"
            )
        
        return ValidationResult(
            is_valid=True,
            sanitized_input=model_name.lower().strip()
        )
    
    def _sanitize_html_entities(self, text: str) -> str:
        """Sanitize HTML entities in text."""
        # Basic HTML entity replacements
        replacements = {
            '&lt;': '<',
            '&gt;': '>',
            '&amp;': '&',
            '&quot;': '"',
            '&#39;': "'",
            '&nbsp;': ' '
        }
        
        for entity, replacement in replacements.items():
            text = text.replace(entity, replacement)
        
        return text
    
    def _has_excessive_repetition(self, text: str, threshold: int = 5) -> bool:
        """Check for excessive character repetition."""
        consecutive_count = 1
        prev_char = None
        
        for char in text:
            if char == prev_char:
                consecutive_count += 1
                if consecutive_count > threshold:
                    return True
            else:
                consecutive_count = 1
                prev_char = char
        
        return False


class ImageValidator:
    """Validates image inputs."""
    
    def __init__(self):
        """Initialize image validator."""
        self.allowed_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.min_file_size = 100  # 100 bytes
        
        # Magic bytes for common image formats
        self.magic_bytes = {
            b'\xff\xd8\xff': 'jpeg',
            b'\x89\x50\x4e\x47\x0d\x0a\x1a\x0a': 'png',
            b'\x47\x49\x46\x38': 'gif',
            b'\x42\x4d': 'bmp',
            b'RIFF': 'webp'  # Simplified - WEBP has more complex header
        }
    
    def validate_image_data(self, image_data: bytes, filename: Optional[str] = None) -> ValidationResult:
        """Validate image data."""
        if not isinstance(image_data, bytes):
            return ValidationResult(
                is_valid=False,
                error_message="Image data must be bytes"
            )
        
        # Size validation
        if len(image_data) < self.min_file_size:
            return ValidationResult(
                is_valid=False,
                error_message=f"Image too small (minimum {self.min_file_size} bytes)"
            )
        
        if len(image_data) > self.max_file_size:
            return ValidationResult(
                is_valid=False,
                error_message=f"Image too large (maximum {self.max_file_size // (1024*1024)}MB)"
            )
        
        # Check magic bytes
        image_format = self._detect_format(image_data)
        if not image_format:
            return ValidationResult(
                is_valid=False,
                error_message="Invalid or unsupported image format"
            )
        
        warnings = []
        
        # Validate filename if provided
        if filename:
            filename_validation = self._validate_filename(filename)
            if not filename_validation.is_valid:
                return filename_validation
            warnings.extend(filename_validation.warnings)
        
        # Check for suspicious patterns
        if self._has_suspicious_patterns(image_data):
            warnings.append("Image contains suspicious binary patterns")
        
        return ValidationResult(
            is_valid=True,
            sanitized_input=image_data,
            warnings=warnings,
            confidence=0.9 if warnings else 1.0
        )
    
    def validate_image_path(self, image_path: Union[str, Path]) -> ValidationResult:
        """Validate image file path."""
        if isinstance(image_path, str):
            image_path = Path(image_path)
        
        if not isinstance(image_path, Path):
            return ValidationResult(
                is_valid=False,
                error_message="Image path must be string or Path object"
            )
        
        # Check if file exists
        if not image_path.exists():
            return ValidationResult(
                is_valid=False,
                error_message="Image file does not exist"
            )
        
        # Check if it's a file (not directory)
        if not image_path.is_file():
            return ValidationResult(
                is_valid=False,
                error_message="Path is not a file"
            )
        
        # Validate filename
        filename_validation = self._validate_filename(image_path.name)
        if not filename_validation.is_valid:
            return filename_validation
        
        # Read and validate file content
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            return self.validate_image_data(image_data, image_path.name)
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Error reading image file: {e}"
            )
    
    def _detect_format(self, image_data: bytes) -> Optional[str]:
        """Detect image format from magic bytes."""
        for magic_bytes, format_name in self.magic_bytes.items():
            if image_data.startswith(magic_bytes):
                return format_name
        return None
    
    def _validate_filename(self, filename: str) -> ValidationResult:
        """Validate image filename."""
        warnings = []
        
        # Check extension
        suffix = Path(filename).suffix.lower()
        if suffix not in self.allowed_formats:
            return ValidationResult(
                is_valid=False,
                error_message=f"Unsupported file format: {suffix}"
            )
        
        # Check for suspicious filename patterns
        if '..' in filename or '/' in filename or '\\' in filename:
            return ValidationResult(
                is_valid=False,
                error_message="Filename contains path traversal characters"
            )
        
        # Check for hidden files
        if filename.startswith('.'):
            warnings.append("Hidden file detected")
        
        # Check for very long filenames
        if len(filename) > 255:
            return ValidationResult(
                is_valid=False,
                error_message="Filename too long"
            )
        
        return ValidationResult(
            is_valid=True,
            sanitized_input=filename,
            warnings=warnings
        )
    
    def _has_suspicious_patterns(self, image_data: bytes) -> bool:
        """Check for suspicious patterns in image data."""
        # Check for embedded executables
        executable_patterns = [
            b'MZ',      # DOS executable
            b'\x7fELF', # ELF executable
            b'\xfe\xed\xfa', # Mach-O executable
        ]
        
        for pattern in executable_patterns:
            if pattern in image_data:
                return True
        
        # Check for script tags in image data (polyglot attacks)
        script_patterns = [
            b'<script',
            b'javascript:',
            b'eval(',
        ]
        
        for pattern in script_patterns:
            if pattern.lower() in image_data.lower():
                return True
        
        return False


class ConfigValidator:
    """Validates configuration inputs."""
    
    def validate_inference_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate inference configuration."""
        required_fields = ['model_name']
        warnings = []
        
        # Check required fields
        for field in required_fields:
            if field not in config:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Missing required field: {field}"
                )
        
        # Validate model name
        model_validation = TextValidator().validate_model_name(config['model_name'])
        if not model_validation.is_valid:
            return model_validation
        
        # Validate optional numeric fields
        numeric_fields = {
            'max_sequence_length': (1, 10000),
            'quantization_bits': (1, 32),
            'timeout_seconds': (0.1, 300.0),
            'batch_size': (1, 100)
        }
        
        for field, (min_val, max_val) in numeric_fields.items():
            if field in config:
                value = config[field]
                if not isinstance(value, (int, float)):
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Field {field} must be numeric"
                    )
                
                if value < min_val or value > max_val:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Field {field} must be between {min_val} and {max_val}"
                    )
        
        # Validate image size
        if 'image_size' in config:
            image_size = config['image_size']
            if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
                return ValidationResult(
                    is_valid=False,
                    error_message="image_size must be a tuple/list of 2 integers"
                )
            
            width, height = image_size
            if not all(isinstance(x, int) and 1 <= x <= 4096 for x in [width, height]):
                return ValidationResult(
                    is_valid=False,
                    error_message="image_size dimensions must be integers between 1 and 4096"
                )
        
        # Validate boolean fields
        boolean_fields = ['enable_caching']
        for field in boolean_fields:
            if field in config and not isinstance(config[field], bool):
                warnings.append(f"Field {field} should be boolean")
        
        return ValidationResult(
            is_valid=True,
            sanitized_input=config,
            warnings=warnings,
            confidence=0.9 if warnings else 1.0
        )


class CompositeValidator:
    """Composite validator that combines all validation types."""
    
    def __init__(self):
        """Initialize composite validator."""
        self.text_validator = TextValidator()
        self.image_validator = ImageValidator()
        self.config_validator = ConfigValidator()
        
        logger.info("Composite validator initialized")
    
    def validate_inference_request(self, 
                                 image_data: Optional[bytes], 
                                 question: str,
                                 config: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate a complete inference request."""
        warnings = []
        
        # Validate question
        question_result = self.text_validator.validate_question(question)
        if not question_result.is_valid:
            return question_result
        warnings.extend(question_result.warnings)
        
        # Validate image if provided
        if image_data is not None:
            image_result = self.image_validator.validate_image_data(image_data)
            if not image_result.is_valid:
                return image_result
            warnings.extend(image_result.warnings)
        
        # Validate config if provided
        if config is not None:
            config_result = self.config_validator.validate_inference_config(config)
            if not config_result.is_valid:
                return config_result
            warnings.extend(config_result.warnings)
        
        return ValidationResult(
            is_valid=True,
            sanitized_input={
                'question': question_result.sanitized_input,
                'image_data': image_data,
                'config': config
            },
            warnings=warnings,
            confidence=min(0.9 if warnings else 1.0, 
                          question_result.confidence,
                          *([] if image_data is None else [image_result.confidence]),
                          *([] if config is None else [config_result.confidence]))
        )
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            "validators": {
                "text_validator": True,
                "image_validator": True,
                "config_validator": True
            },
            "supported_formats": list(self.image_validator.allowed_formats),
            "max_image_size_mb": self.image_validator.max_file_size // (1024 * 1024),
            "max_question_length": 1000
        }


# Global validator instance
input_validator = CompositeValidator()


def validate_input(func):
    """Decorator to validate function inputs."""
    def wrapper(*args, **kwargs):
        # Extract validation parameters
        question = kwargs.get('question') or (args[1] if len(args) > 1 else None)
        image_data = kwargs.get('image_data') or (args[0] if len(args) > 0 else None)
        
        if question:
            validation_result = input_validator.validate_inference_request(
                image_data, question
            )
            
            if not validation_result.is_valid:
                raise ValueError(f"Input validation failed: {validation_result.error_message}")
            
            if validation_result.warnings:
                logger.warning(f"Input validation warnings: {validation_result.warnings}")
        
        return func(*args, **kwargs)
    
    return wrapper


if __name__ == "__main__":
    # Demo input validation
    print("FastVLM Input Validation Demo")
    print("=" * 40)
    
    validator = CompositeValidator()
    
    # Test question validation
    print("\n📝 Testing question validation:")
    
    test_questions = [
        "What objects are in this image?",  # Valid
        "SELECT * FROM users",             # SQL injection
        "<script>alert('xss')</script>",   # XSS
        "What do you see?" * 100,          # Too long
        "a",                               # Too short
    ]
    
    for question in test_questions:
        result = validator.text_validator.validate_question(question)
        status = "✓" if result.is_valid else "✗"
        truncated = question[:30] + "..." if len(question) > 30 else question
        print(f"  {status} '{truncated}' - {result.error_message or 'Valid'}")
    
    # Test image validation
    print("\n🖼️  Testing image validation:")
    
    # Create dummy image data
    fake_jpeg = b'\xff\xd8\xff\xe0' + b'dummy jpeg data' * 10
    fake_png = b'\x89\x50\x4e\x47\x0d\x0a\x1a\x0a' + b'dummy png data' * 10
    malicious_data = b'MZ' + b'executable data'
    
    test_images = [
        (fake_jpeg, "test.jpg", "Valid JPEG"),
        (fake_png, "test.png", "Valid PNG"),
        (malicious_data, "test.jpg", "Malicious data"),
        (b'short', "test.jpg", "Too short"),
    ]
    
    for image_data, filename, description in test_images:
        result = validator.image_validator.validate_image_data(image_data, filename)
        status = "✓" if result.is_valid else "✗"
        print(f"  {status} {description} - {result.error_message or 'Valid'}")
    
    # Test complete inference request
    print("\n🔍 Testing complete inference request:")
    result = validator.validate_inference_request(
        fake_jpeg,
        "What objects are in this image?",
        {"model_name": "fast-vlm-base", "max_sequence_length": 77}
    )
    
    print(f"  Complete validation: {'✓ Valid' if result.is_valid else '✗ Invalid'}")
    if result.warnings:
        print(f"  Warnings: {result.warnings}")
    
    print(f"\n📊 Validation stats:")
    stats = validator.get_validation_stats()
    print(f"  Supported formats: {stats['supported_formats']}")
    print(f"  Max image size: {stats['max_image_size_mb']}MB")
    print(f"  Max question length: {stats['max_question_length']} chars")