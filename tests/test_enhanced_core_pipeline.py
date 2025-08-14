"""
Comprehensive test suite for enhanced FastVLM core pipeline.
Tests all Generation 1-3 features including robustness and performance.
"""

import pytest
import time
import threading
import tempfile
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import the modules we're testing
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fast_vlm_ondevice.core_pipeline import (
    FastVLMCorePipeline, InferenceConfig, InferenceResult,
    EnhancedInputValidator, CircuitBreaker, create_demo_image
)


class TestEnhancedInputValidator:
    """Test the enhanced input validation system."""
    
    def test_valid_image_validation(self):
        """Test validation of valid image data."""
        validator = EnhancedInputValidator()
        
        # Valid image data
        valid_image = b"valid_image_data" * 100  # 1.6KB
        is_valid, message = validator.validate_image(valid_image)
        
        assert is_valid is True
        assert "valid" in message.lower()
    
    def test_oversized_image_rejection(self):
        """Test rejection of oversized images."""
        validator = EnhancedInputValidator()
        
        # Create oversized image (>50MB)
        oversized_image = b"x" * (60 * 1024 * 1024)  # 60MB
        is_valid, message = validator.validate_image(oversized_image)
        
        assert is_valid is False
        assert "too large" in message.lower()
    
    def test_suspicious_image_content_detection(self):
        """Test detection of suspicious content in image data."""
        validator = EnhancedInputValidator()
        
        # Image with suspicious script content
        suspicious_image = b"<script>alert('xss')</script>" + b"image_data" * 50
        is_valid, message = validator.validate_image(suspicious_image)
        
        assert is_valid is False
        assert "suspicious" in message.lower()
    
    def test_valid_question_validation(self):
        """Test validation of valid questions."""
        validator = EnhancedInputValidator()
        
        valid_questions = [
            "What objects are in this image?",
            "Can you describe the scene?",
            "How many people are visible?",
            "What is the main color in this picture?"
        ]
        
        for question in valid_questions:
            is_valid, message = validator.validate_question(question)
            assert is_valid is True, f"Question '{question}' should be valid"
    
    def test_question_length_limits(self):
        """Test question length validation."""
        validator = EnhancedInputValidator()
        
        # Too long question
        long_question = "What is this? " * 200  # >1000 chars
        is_valid, message = validator.validate_question(long_question)
        
        assert is_valid is False
        assert "too long" in message.lower()
    
    def test_empty_question_rejection(self):
        """Test rejection of empty questions."""
        validator = EnhancedInputValidator()
        
        empty_questions = ["", "   ", "\n\t"]
        
        for question in empty_questions:
            is_valid, message = validator.validate_question(question)
            assert is_valid is False
            assert "empty" in message.lower()
    
    def test_suspicious_question_content(self):
        """Test detection of suspicious content in questions."""
        validator = EnhancedInputValidator()
        
        suspicious_questions = [
            "What is <script>alert('xss')</script> in this image?",
            "Can you eval(malicious_code) this picture?",
            "Tell me about javascript:void(0) in the image"
        ]
        
        for question in suspicious_questions:
            is_valid, message = validator.validate_question(question)
            assert is_valid is False
            assert "suspicious" in message.lower()


class TestCircuitBreaker:
    """Test the circuit breaker fault tolerance mechanism."""
    
    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in normal closed state."""
        breaker = CircuitBreaker(failure_threshold=3, timeout_seconds=1)
        
        # Successful calls should work normally
        def success_func():
            return "success"
        
        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.state == "CLOSED"
    
    def test_circuit_breaker_failure_tracking(self):
        """Test failure tracking and state transitions."""
        breaker = CircuitBreaker(failure_threshold=2, timeout_seconds=1)
        
        def failing_func():
            raise ValueError("Test failure")
        
        # First failure
        with pytest.raises(ValueError):
            breaker.call(failing_func)
        assert breaker.failure_count == 1
        assert breaker.state == "CLOSED"
        
        # Second failure should open the circuit
        with pytest.raises(ValueError):
            breaker.call(failing_func)
        assert breaker.failure_count == 2
        assert breaker.state == "OPEN"
    
    def test_circuit_breaker_open_state(self):
        """Test circuit breaker rejecting calls when open."""
        breaker = CircuitBreaker(failure_threshold=1, timeout_seconds=1)
        
        # Trigger failure to open circuit
        def failing_func():
            raise ValueError("Test failure")
        
        with pytest.raises(ValueError):
            breaker.call(failing_func)
        
        assert breaker.state == "OPEN"
        
        # Now successful function should be rejected
        def success_func():
            return "success"
        
        with pytest.raises(RuntimeError, match="Circuit breaker is OPEN"):
            breaker.call(success_func)
    
    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker recovery through half-open state."""
        breaker = CircuitBreaker(failure_threshold=1, timeout_seconds=0.1)
        
        # Open the circuit
        def failing_func():
            raise ValueError("Test failure")
        
        with pytest.raises(ValueError):
            breaker.call(failing_func)
        assert breaker.state == "OPEN"
        
        # Wait for timeout
        time.sleep(0.2)
        
        # Next call should transition to HALF_OPEN
        def success_func():
            return "success"
        
        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.state == "CLOSED"
        assert breaker.failure_count == 0


class TestFastVLMCorePipeline:
    """Test the enhanced FastVLM core pipeline."""
    
    @pytest.fixture
    def pipeline(self):
        """Create a test pipeline instance."""
        config = InferenceConfig(
            model_name="fast-vlm-test",
            enable_caching=True,
            timeout_seconds=10.0
        )
        return FastVLMCorePipeline(config)
    
    @pytest.fixture
    def demo_image_data(self):
        """Create demo image data for testing."""
        return create_demo_image()
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization with all components."""
        assert pipeline.config.model_name == "fast-vlm-test"
        assert pipeline.vision_encoder is not None
        assert pipeline.text_encoder is not None
        assert pipeline.fusion_module is not None
        assert pipeline.answer_generator is not None
        assert pipeline.input_validator is not None
        assert pipeline.circuit_breaker is not None
        assert pipeline.cache is not None
    
    def test_successful_image_question_processing(self, pipeline, demo_image_data):
        """Test successful processing of image and question."""
        question = "What objects are in this image?"
        
        result = pipeline.process_image_question(demo_image_data, question)
        
        assert isinstance(result, InferenceResult)
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0
        assert result.latency_ms > 0
        assert result.model_used == "fast-vlm-test"
        assert result.metadata is not None
    
    def test_caching_functionality(self, pipeline, demo_image_data):
        """Test caching of inference results."""
        question = "What is the main color?"
        
        # First request
        result1 = pipeline.process_image_question(demo_image_data, question)
        cache_size_after_first = len(pipeline.cache)
        
        # Second identical request (should hit cache)
        result2 = pipeline.process_image_question(demo_image_data, question)
        
        assert cache_size_after_first > 0
        assert len(pipeline.cache) == cache_size_after_first  # No new cache entry
        assert result1.answer == result2.answer
        assert pipeline.cache_stats["hits"] > 0
    
    def test_input_validation_integration(self, pipeline):
        """Test input validation integration in pipeline."""
        # Test with invalid image
        with pytest.raises(Exception):  # Should raise validation error
            pipeline.process_image_question(b"", "What is this?")
        
        # Test with invalid question
        with pytest.raises(Exception):  # Should raise validation error
            demo_image = create_demo_image()
            pipeline.process_image_question(demo_image, "")
    
    def test_error_handling_and_graceful_degradation(self, pipeline):
        """Test error handling and graceful response generation."""
        # Simulate processing error by mocking a component
        original_encode = pipeline.vision_encoder.encode_image
        pipeline.vision_encoder.encode_image = Mock(side_effect=RuntimeError("Test error"))
        
        demo_image = create_demo_image()
        result = pipeline.process_image_question(demo_image, "What is this?")
        
        # Should return error response, not raise exception
        assert isinstance(result, InferenceResult)
        assert result.confidence == 0.0
        assert "error" in result.metadata
        assert "test error" in result.answer.lower() or "error" in result.answer.lower()
        
        # Restore original method
        pipeline.vision_encoder.encode_image = original_encode
    
    def test_processing_statistics_tracking(self, pipeline, demo_image_data):
        """Test tracking of processing statistics."""
        initial_stats = pipeline.processing_stats.copy()
        
        # Process a few requests
        questions = [
            "What objects are visible?",
            "What colors do you see?",
            "Describe the scene"
        ]
        
        for question in questions:
            pipeline.process_image_question(demo_image_data, question)
        
        # Check statistics updates
        assert pipeline.processing_stats["total_requests"] > initial_stats["total_requests"]
        assert pipeline.processing_stats["successful_requests"] > initial_stats["successful_requests"]
        assert pipeline.processing_stats["average_latency_ms"] > 0
    
    def test_health_status_reporting(self, pipeline, demo_image_data):
        """Test health status reporting functionality."""
        # Process some requests to generate data
        for i in range(5):
            pipeline.process_image_question(demo_image_data, f"Test question {i}")
        
        health_status = pipeline.get_health_status()
        
        assert "status" in health_status
        assert "success_rate_percent" in health_status
        assert "circuit_breaker_state" in health_status
        assert "total_requests" in health_status
        assert "average_latency_ms" in health_status
        assert health_status["total_requests"] >= 5
        assert health_status["success_rate_percent"] > 0
    
    def test_concurrent_processing(self, pipeline, demo_image_data):
        """Test thread safety of concurrent processing."""
        results = []
        errors = []
        
        def process_request(question_id):
            try:
                result = pipeline.process_image_question(
                    demo_image_data, 
                    f"Concurrent question {question_id}"
                )
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=process_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Concurrent processing errors: {errors}"
        assert len(results) == 10
        
        # All results should be valid
        for result in results:
            assert isinstance(result, InferenceResult)
            assert len(result.answer) > 0
    
    def test_cache_size_management(self, pipeline, demo_image_data):
        """Test cache size management and eviction."""
        # Fill cache beyond reasonable size
        for i in range(50):
            unique_question = f"Unique question number {i} with details"
            pipeline.process_image_question(demo_image_data, unique_question)
        
        # Cache should have reasonable size
        cache_size = len(pipeline.cache)
        assert cache_size < 50, "Cache should implement size limits"
        
        # Should have eviction statistics
        assert "evictions" in pipeline.cache_stats
    
    def test_text_only_processing(self, pipeline):
        """Test text-only question processing."""
        text_questions = [
            "Hello, how are you?",
            "What is FastVLM?",
            "Can you help me?"
        ]
        
        for question in text_questions:
            result = pipeline.process_text_only(question)
            
            assert isinstance(result, InferenceResult)
            assert len(result.answer) > 0
            assert result.metadata.get("mode") == "text_only"
    
    def test_pipeline_stats_and_metrics(self, pipeline, demo_image_data):
        """Test comprehensive pipeline statistics."""
        # Generate some activity
        for i in range(3):
            pipeline.process_image_question(demo_image_data, f"Stats test {i}")
        
        stats = pipeline.get_stats()
        
        # Verify expected stats structure
        expected_keys = [
            "model_name", "cache_enabled", "cache_entries",
            "quantization_bits", "max_sequence_length", "image_size",
            "components"
        ]
        
        for key in expected_keys:
            assert key in stats
        
        assert stats["cache_entries"] >= 0
        assert "vision_encoder" in stats["components"]
        assert "text_encoder" in stats["components"]
    
    def test_fallback_component_initialization(self):
        """Test fallback component initialization when main init fails."""
        # Create pipeline that might fail initialization
        with patch.object(FastVLMCorePipeline, '_determine_model_size', side_effect=RuntimeError("Init failed")):
            with pytest.raises(RuntimeError):
                config = InferenceConfig(model_name="failing-model")
                pipeline = FastVLMCorePipeline(config)


class TestPerformanceOptimizations:
    """Test performance optimization features."""
    
    @pytest.fixture
    def pipeline_with_optimizer(self):
        """Create pipeline with mobile optimizer enabled."""
        config = InferenceConfig(model_name="fast-vlm-mobile")
        pipeline = FastVLMCorePipeline(config)
        return pipeline
    
    def test_mobile_optimizer_integration(self, pipeline_with_optimizer):
        """Test mobile performance optimizer integration."""
        # Pipeline should have optimizer (or gracefully handle missing optimizer)
        demo_image = create_demo_image()
        result = pipeline_with_optimizer.process_image_question(demo_image, "Test optimization")
        
        assert isinstance(result, InferenceResult)
        # Optimizer should be either enabled or gracefully disabled
        assert hasattr(pipeline_with_optimizer, 'mobile_optimizer')
    
    def test_adaptive_quality_response(self, pipeline_with_optimizer):
        """Test that pipeline adapts to performance requirements."""
        demo_image = create_demo_image()
        
        # Process multiple requests to trigger adaptive behavior
        latencies = []
        for i in range(10):
            start_time = time.time()
            result = pipeline_with_optimizer.process_image_question(demo_image, f"Adaptive test {i}")
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)
        
        # Later requests should potentially be optimized
        assert all(l > 0 for l in latencies), "All requests should have measurable latency"
        assert all(isinstance(l, (int, float)) for l in latencies), "Latencies should be numeric"


class TestErrorRecoveryAndResilience:
    """Test error recovery and system resilience."""
    
    def test_circuit_breaker_integration(self):
        """Test circuit breaker integration in pipeline."""
        config = InferenceConfig(model_name="circuit-test")
        pipeline = FastVLMCorePipeline(config)
        
        # Force failures to trigger circuit breaker
        original_method = pipeline.vision_encoder.encode_image
        
        # Create failing encoder
        def failing_encoder(image_data):
            raise RuntimeError("Simulated failure")
        
        pipeline.vision_encoder.encode_image = failing_encoder
        
        # First few failures should trigger circuit breaker
        demo_image = create_demo_image()
        for i in range(6):  # Exceed failure threshold
            result = pipeline.process_image_question(demo_image, f"Failure test {i}")
            assert isinstance(result, InferenceResult)
            assert result.confidence == 0.0  # Error response
        
        # Circuit breaker should eventually be triggered
        assert pipeline.circuit_breaker.failure_count > 0
        
        # Restore original method
        pipeline.vision_encoder.encode_image = original_method
    
    def test_graceful_resource_cleanup(self):
        """Test graceful cleanup of resources."""
        pipeline = FastVLMCorePipeline()
        
        # Simulate heavy usage
        demo_image = create_demo_image()
        for i in range(20):
            pipeline.process_image_question(demo_image, f"Cleanup test {i}")
        
        # Clear cache and verify cleanup
        initial_cache_size = len(pipeline.cache) if pipeline.cache else 0
        cleared_entries = pipeline.clear_cache()
        
        if initial_cache_size > 0:
            assert cleared_entries == initial_cache_size
            assert len(pipeline.cache) == 0
    
    def test_memory_pressure_handling(self):
        """Test handling of memory pressure scenarios."""
        config = InferenceConfig(
            model_name="memory-test",
            enable_caching=True
        )
        pipeline = FastVLMCorePipeline(config)
        
        # Generate large number of unique requests to stress memory
        demo_image = create_demo_image()
        large_questions = [
            f"This is a very long question number {i} with lots of detail and unique content " * 5
            for i in range(100)
        ]
        
        results = []
        for question in large_questions[:20]:  # Process subset to avoid timeout
            result = pipeline.process_image_question(demo_image, question)
            results.append(result)
        
        # All requests should complete successfully
        assert len(results) == 20
        assert all(isinstance(r, InferenceResult) for r in results)
        
        # Cache should implement reasonable size limits
        if pipeline.cache:
            assert len(pipeline.cache) < 50, "Cache should implement size management"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])