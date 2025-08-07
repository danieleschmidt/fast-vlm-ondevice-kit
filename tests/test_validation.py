"""
Test suite for validation framework components.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from src.fast_vlm_ondevice.validation import (
    ValidationLevel,
    ValidationConfig,
    ValidationResult,
    InputValidator,
    ModelIntegrityChecker,
    RuntimeValidator,
    create_validation_suite,
    validate_system_health,
    safe_execute
)


class TestValidationResult:
    """Test suite for ValidationResult."""
    
    def test_validation_result_creation(self):
        """Test validation result creation."""
        result = ValidationResult(
            valid=True,
            message="Test passed",
            details={"test": "value"}
        )
        
        assert result.valid == True
        assert result.message == "Test passed"
        assert result.details["test"] == "value"
        assert result.timestamp > 0
    
    def test_validation_result_to_dict(self):
        """Test conversion to dictionary."""
        result = ValidationResult(
            valid=False,
            message="Test failed",
            details={"error": "test error"}
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["valid"] == False
        assert result_dict["message"] == "Test failed"
        assert result_dict["details"]["error"] == "test error"
        assert "timestamp" in result_dict


class TestValidationConfig:
    """Test suite for ValidationConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ValidationConfig()
        
        assert config.level == ValidationLevel.BALANCED
        assert config.timeout_seconds == 30.0
        assert config.enable_runtime_checks == True
        assert config.strict_type_checking == True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ValidationConfig(
            level=ValidationLevel.STRICT,
            timeout_seconds=60.0,
            strict_type_checking=False
        )
        
        assert config.level == ValidationLevel.STRICT
        assert config.timeout_seconds == 60.0
        assert config.strict_type_checking == False


class TestInputValidator:
    """Test suite for InputValidator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ValidationConfig()
        self.validator = InputValidator(self.config)
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = InputValidator()
        assert validator.config is not None
    
    def test_valid_image_input(self):
        """Test validation of valid image input."""
        self.setUp()
        
        # Test valid numpy array image
        valid_image = np.random.rand(224, 224, 3).astype(np.float32)
        
        result = self.validator.validate_image_input(valid_image)
        
        assert result.valid == True
        assert "shape" in result.details
    
    def test_invalid_image_input(self):
        """Test validation of invalid image input."""
        self.setUp()
        
        # Test None input
        result = self.validator.validate_image_input(None)
        assert result.valid == False
        assert "None" in result.message
        
        # Test wrong type
        result = self.validator.validate_image_input("not_an_image")
        assert result.valid == False
        
        # Test wrong dimensions
        result = self.validator.validate_image_input(np.array([1, 2, 3]))
        assert result.valid == False
        assert "dimensions" in result.message
    
    def test_image_with_nan_values(self):
        """Test image containing NaN values."""
        self.setUp()
        
        image_with_nan = np.random.rand(100, 100, 3)
        image_with_nan[50, 50, 0] = np.nan
        
        result = self.validator.validate_image_input(image_with_nan)
        
        assert result.valid == False
        assert "NaN" in result.message
    
    def test_image_with_inf_values(self):
        """Test image containing infinite values."""
        self.setUp()
        
        image_with_inf = np.random.rand(100, 100, 3)
        image_with_inf[25, 25, 1] = np.inf
        
        result = self.validator.validate_image_input(image_with_inf)
        
        assert result.valid == False
        assert "infinite" in result.message
    
    def test_image_size_limits(self):
        """Test image size validation."""
        self.setUp()
        
        # Test very large image
        large_image = np.random.rand(5000, 5000, 3)
        result = self.validator.validate_image_input(large_image)
        assert result.valid == False
        assert "too large" in result.message
        
        # Test very small image
        small_image = np.random.rand(10, 10, 3)
        result = self.validator.validate_image_input(small_image)
        # Should be warning, not error
        assert result.valid == False
        assert "very small" in result.message
    
    def test_valid_text_input(self):
        """Test validation of valid text input."""
        self.setUp()
        
        # Test string input
        result = self.validator.validate_text_input("This is valid text")
        assert result.valid == True
        
        # Test list input
        result = self.validator.validate_text_input([1, 2, 3, 4, 5])
        assert result.valid == True
        
        # Test numpy array input
        text_array = np.random.rand(50, 768)
        result = self.validator.validate_text_input(text_array)
        assert result.valid == True
    
    def test_invalid_text_input(self):
        """Test validation of invalid text input."""
        self.setUp()
        
        # Test None input
        result = self.validator.validate_text_input(None)
        assert result.valid == False
        
        # Test empty string
        result = self.validator.validate_text_input("")
        assert result.valid == False
        assert "Empty" in result.message
        
        # Test empty list
        result = self.validator.validate_text_input([])
        assert result.valid == False
    
    def test_text_with_nan_embeddings(self):
        """Test text embeddings containing NaN values."""
        self.setUp()
        
        text_embeddings = np.random.rand(10, 512)
        text_embeddings[5, 100] = np.nan
        
        result = self.validator.validate_text_input(text_embeddings)
        
        assert result.valid == False
        assert "NaN" in result.message
    
    def test_very_long_text(self):
        """Test very long text input."""
        self.setUp()
        
        very_long_text = "word " * 3000  # Very long text
        
        result = self.validator.validate_text_input(very_long_text)
        
        # Should be warning about performance impact
        assert result.valid == False
        assert "very long" in result.message
    
    def test_valid_model_config(self):
        """Test validation of valid model configuration."""
        self.setUp()
        
        valid_config = {
            "model_name": "FastVLM-Test",
            "version": "1.0.0",
            "quantization": "int4",
            "batch_size": 2,
            "max_length": 512,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9
        }
        
        result = self.validator.validate_model_config(valid_config)
        
        assert result.valid == True
        assert "fields_validated" in result.details
    
    def test_invalid_model_config(self):
        """Test validation of invalid model configuration."""
        self.setUp()
        
        # Test missing required fields
        invalid_config = {"quantization": "int4"}
        
        result = self.validator.validate_model_config(invalid_config)
        
        assert result.valid == False
        assert "Missing required" in result.message
        
        # Test wrong types
        invalid_types_config = {
            "model_name": "FastVLM-Test",
            "version": "1.0.0",
            "batch_size": "not_an_integer",
            "temperature": "not_a_float"
        }
        
        result = self.validator.validate_model_config(invalid_types_config)
        
        assert result.valid == False
        assert "type validation failed" in result.message
    
    def test_config_value_ranges(self):
        """Test model config value range validation."""
        self.setUp()
        
        out_of_range_config = {
            "model_name": "FastVLM-Test",
            "version": "1.0.0",
            "batch_size": 100,  # Too large
            "temperature": 5.0,  # Too high
            "top_p": 1.5  # Invalid probability
        }
        
        result = self.validator.validate_model_config(out_of_range_config)
        
        assert result.valid == False
        assert "outside recommended ranges" in result.message
    
    def test_validation_level_disabled(self):
        """Test behavior when validation is disabled."""
        config = ValidationConfig(level=ValidationLevel.DISABLED)
        validator = InputValidator(config)
        
        # All validations should pass when disabled
        assert validator.validate_image_input(None).valid == True
        assert validator.validate_text_input(None).valid == True
        assert validator.validate_model_config({}).valid == True


class TestModelIntegrityChecker:
    """Test suite for ModelIntegrityChecker."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ValidationConfig()
        self.checker = ModelIntegrityChecker(self.config)
    
    def test_checker_initialization(self):
        """Test checker initialization."""
        checker = ModelIntegrityChecker()
        assert checker.config is not None
    
    def test_nonexistent_file(self):
        """Test validation of non-existent file."""
        self.setUp()
        
        result = self.checker.validate_model_file("/path/that/does/not/exist")
        
        assert result.valid == False
        assert "does not exist" in result.message
    
    def test_empty_file(self):
        """Test validation of empty file."""
        self.setUp()
        
        # Create temporary empty file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        result = self.checker.validate_model_file(tmp_path)
        
        assert result.valid == False
        assert "empty" in result.message
        
        # Clean up
        Path(tmp_path).unlink()
    
    def test_valid_json_file(self):
        """Test validation of valid JSON file."""
        self.setUp()
        
        # Create temporary JSON file
        test_data = {"model": "test", "version": "1.0"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(test_data, tmp)
            tmp_path = tmp.name
        
        result = self.checker.validate_model_file(tmp_path)
        
        assert result.valid == True
        
        # Clean up
        Path(tmp_path).unlink()
    
    def test_invalid_json_file(self):
        """Test validation of invalid JSON file."""
        self.setUp()
        
        # Create temporary invalid JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp.write('{"invalid": json}')  # Invalid JSON syntax
            tmp_path = tmp.name
        
        result = self.checker.validate_model_file(tmp_path)
        
        assert result.valid == False
        assert "JSON" in result.message
        
        # Clean up
        Path(tmp_path).unlink()
    
    def test_large_file_warning(self):
        """Test warning for very large files."""
        config = ValidationConfig(max_memory_mb=1.0)  # Very small limit
        checker = ModelIntegrityChecker(config)
        
        # Create temporary large file (larger than limit)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b'x' * (2 * 1024 * 1024))  # 2MB file
            tmp_path = tmp.name
        
        result = checker.validate_model_file(tmp_path)
        
        # Should warn about large size
        assert result.valid == False
        assert "very large" in result.message
        
        # Clean up
        Path(tmp_path).unlink()
    
    def test_validation_disabled(self):
        """Test behavior when validation is disabled."""
        config = ValidationConfig(level=ValidationLevel.DISABLED)
        checker = ModelIntegrityChecker(config)
        
        # Should pass even for non-existent file when disabled
        result = checker.validate_model_file("/nonexistent/file")
        assert result.valid == True


class TestRuntimeValidator:
    """Test suite for RuntimeValidator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ValidationConfig()
        self.validator = RuntimeValidator(self.config)
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = RuntimeValidator()
        assert validator.config is not None
        assert len(validator.performance_history) == 0
    
    def test_valid_inference_output(self):
        """Test validation of valid inference output."""
        self.setUp()
        
        valid_output = np.random.rand(10, 1000).astype(np.float32)
        
        result = self.validator.validate_inference_output(valid_output)
        
        assert result.valid == True
        assert "shape" in result.details
    
    def test_none_output(self):
        """Test validation of None output."""
        self.setUp()
        
        result = self.validator.validate_inference_output(None)
        
        assert result.valid == False
        assert "None" in result.message
    
    def test_output_shape_mismatch(self):
        """Test validation of output with wrong shape."""
        self.setUp()
        
        output = np.random.rand(5, 100)
        expected_shape = (10, 200)
        
        result = self.validator.validate_inference_output(output, expected_shape)
        
        assert result.valid == False
        assert "shape mismatch" in result.message
        assert result.details["expected"] == expected_shape
        assert result.details["actual"] == output.shape
    
    def test_output_with_nan(self):
        """Test validation of output containing NaN."""
        self.setUp()
        
        output = np.random.rand(10, 100)
        output[5, 50] = np.nan
        
        result = self.validator.validate_inference_output(output)
        
        assert result.valid == False
        assert "NaN" in result.message
    
    def test_output_with_inf(self):
        """Test validation of output containing infinity."""
        self.setUp()
        
        output = np.random.rand(10, 100)
        output[3, 25] = np.inf
        
        result = self.validator.validate_inference_output(output)
        
        assert result.valid == False
        assert "infinite" in result.message
    
    def test_performance_monitoring_decorator(self):
        """Test performance monitoring decorator."""
        self.setUp()
        
        @self.validator.monitor_performance
        def test_function(x, y):
            return x + y
        
        result = test_function(2, 3)
        
        assert result == 5
        assert len(self.validator.performance_history) == 1
        
        perf_record = self.validator.performance_history[0]
        assert perf_record["function"] == "test_function"
        assert perf_record["success"] == True
        assert perf_record["execution_time"] > 0
    
    def test_performance_monitoring_with_exception(self):
        """Test performance monitoring when function raises exception."""
        self.setUp()
        
        @self.validator.monitor_performance
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_function()
        
        assert len(self.validator.performance_history) == 1
        
        perf_record = self.validator.performance_history[0]
        assert perf_record["success"] == False
        assert perf_record["error"] == "Test error"
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        self.setUp()
        
        # Add some mock performance data
        @self.validator.monitor_performance
        def mock_function():
            return "result"
        
        # Run function multiple times
        for _ in range(5):
            mock_function()
        
        summary = self.validator.get_performance_summary()
        
        assert "total_calls" in summary
        assert "successful_calls" in summary
        assert "success_rate" in summary
        assert "avg_execution_time" in summary
        
        assert summary["total_calls"] == 5
        assert summary["successful_calls"] == 5
        assert summary["success_rate"] == 1.0
    
    def test_monitoring_disabled(self):
        """Test behavior when monitoring is disabled."""
        config = ValidationConfig(monitor_performance=False)
        validator = RuntimeValidator(config)
        
        @validator.monitor_performance
        def test_function():
            return "result"
        
        result = test_function()
        
        assert result == "result"
        # No performance data should be recorded
        assert len(validator.performance_history) == 0


class TestUtilityFunctions:
    """Test suite for utility functions."""
    
    def test_create_validation_suite(self):
        """Test validation suite creation."""
        suite = create_validation_suite(ValidationLevel.STRICT)
        
        assert "input_validator" in suite
        assert "model_integrity_checker" in suite
        assert "runtime_validator" in suite
        assert "config" in suite
        
        assert isinstance(suite["input_validator"], InputValidator)
        assert isinstance(suite["model_integrity_checker"], ModelIntegrityChecker)
        assert isinstance(suite["runtime_validator"], RuntimeValidator)
        
        assert suite["config"].level == ValidationLevel.STRICT
    
    def test_validate_system_health(self):
        """Test system health validation."""
        result = validate_system_health()
        
        assert isinstance(result, ValidationResult)
        # System health should generally be valid in test environment
        # (unless we're specifically testing on a constrained system)
    
    def test_safe_execute_success(self):
        """Test safe execution with successful function."""
        def successful_function(x, y):
            return x * y
        
        result, validation_results = safe_execute(successful_function, 3, 4)
        
        assert result == 12
        assert isinstance(validation_results, list)
        assert len(validation_results) > 0  # Should have system health check
    
    def test_safe_execute_failure(self):
        """Test safe execution with failing function."""
        def failing_function():
            raise RuntimeError("Test failure")
        
        result, validation_results = safe_execute(failing_function)
        
        assert result is None
        assert len(validation_results) > 0
        
        # Should have error validation result
        error_results = [vr for vr in validation_results if not vr.valid]
        assert len(error_results) > 0
    
    def test_safe_execute_with_validation_config(self):
        """Test safe execution with custom validation config."""
        config = ValidationConfig(
            level=ValidationLevel.MINIMAL,
            enable_runtime_checks=False
        )
        
        def test_function():
            return "success"
        
        result, validation_results = safe_execute(
            test_function, 
            validation_config=config
        )
        
        assert result == "success"
        # Minimal validation should produce fewer validation results
        assert isinstance(validation_results, list)


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""
    
    def test_validation_with_extreme_values(self):
        """Test validation with extreme input values."""
        validator = InputValidator()
        
        # Test with very large numbers
        extreme_image = np.full((100, 100, 3), 1e10)
        result = validator.validate_image_input(extreme_image)
        
        # Should handle gracefully (may warn about unusual values)
        assert isinstance(result, ValidationResult)
    
    def test_validation_with_mixed_types(self):
        """Test validation with mixed data types."""
        validator = InputValidator()
        
        # Test with mixed type configuration
        mixed_config = {
            "model_name": "test",
            "version": "1.0",
            "batch_size": 1,
            "temperature": 0.5,
            "mixed_field": [1, "two", 3.0]  # Mixed types
        }
        
        result = validator.validate_model_config(mixed_config)
        assert isinstance(result, ValidationResult)
    
    def test_validation_with_unicode_text(self):
        """Test validation with unicode text input."""
        validator = InputValidator()
        
        # Test with various unicode characters
        unicode_text = "Hello 世界 🌍 مرحبا עולם"
        
        result = validator.validate_text_input(unicode_text)
        assert result.valid == True
    
    def test_concurrent_validation(self):
        """Test validation under concurrent access."""
        import threading
        
        validator = InputValidator()
        results = []
        errors = []
        
        def validate_concurrently():
            try:
                test_data = np.random.rand(50, 50, 3)
                result = validator.validate_image_input(test_data)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=validate_concurrently)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0  # No errors should occur
        assert len(results) == 10  # All validations should complete
        assert all(result.valid for result in results)  # All should be valid


if __name__ == "__main__":
    pytest.main([__file__])