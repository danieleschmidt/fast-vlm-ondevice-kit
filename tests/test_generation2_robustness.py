"""
Comprehensive robustness testing for FastVLM Generation 2.
Tests error handling, validation, security, and reliability features.
"""

import time
import threading
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.fast_vlm_ondevice import (
    FastVLMCorePipeline, 
    InferenceConfig, 
    create_demo_image,
    quick_inference
)


class TestGeneration2Robustness:
    """Test suite for Generation 2 robustness features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = InferenceConfig(
            model_name="fast-vlm-base",
            enable_caching=True,
            timeout_seconds=10.0
        )
        self.pipeline = FastVLMCorePipeline(self.config)
        self.demo_image = create_demo_image()
    
    def test_input_validation_security(self):
        """Test security-focused input validation."""
        # Test malicious image data
        malicious_image_patterns = [
            b"<script>alert('xss')</script>" + b"x" * 1000,
            b"javascript:void(0)" + b"x" * 1000,
            b"data:text/html,<script>" + b"x" * 1000,
        ]
        
        for malicious_data in malicious_image_patterns:
            result = self.pipeline.process_image_question(malicious_data, "What is this?")
            # Should handle gracefully with error response
            assert "validation failed" in result.answer.lower() or "error" in result.answer.lower()
            assert result.confidence == 0.0
        
        # Test malicious question patterns
        malicious_questions = [
            "<script>alert('xss')</script>",
            "javascript:void(0)",
            "eval(malicious_code)",
            "document.cookie",
            "window.location"
        ]
        
        for malicious_question in malicious_questions:
            result = self.pipeline.process_image_question(self.demo_image, malicious_question)
            assert "suspicious content" in result.answer.lower() or "validation failed" in result.answer.lower()
            assert result.confidence == 0.0
    
    def test_input_size_limits(self):
        """Test input size validation."""
        # Test oversized image
        oversized_image = b"x" * (60 * 1024 * 1024)  # 60MB
        result = self.pipeline.process_image_question(oversized_image, "What is this?")
        assert "too large" in result.answer.lower()
        assert result.confidence == 0.0
        
        # Test oversized question
        oversized_question = "What is this? " * 1000  # Very long question
        result = self.pipeline.process_image_question(self.demo_image, oversized_question)
        assert "too long" in result.answer.lower()
        assert result.confidence == 0.0
        
        # Test empty inputs
        result = self.pipeline.process_image_question(b"", "What is this?")
        assert "too small" in result.answer.lower() or "error" in result.answer.lower()
        
        result = self.pipeline.process_image_question(self.demo_image, "")
        assert "empty" in result.answer.lower() or "error" in result.answer.lower()
    
    def test_circuit_breaker_functionality(self):
        """Test circuit breaker fault tolerance."""
        # Create a pipeline that will fail
        def create_failing_pipeline():
            config = InferenceConfig(model_name="fast-vlm-base")
            pipeline = FastVLMCorePipeline(config)
            
            # Override a component to always fail
            original_encode = pipeline.vision_encoder.encode_image
            def failing_encode(image_data):
                raise RuntimeError("Simulated component failure")
            pipeline.vision_encoder.encode_image = failing_encode
            
            return pipeline
        
        failing_pipeline = create_failing_pipeline()
        
        # Trigger multiple failures to open circuit breaker
        failure_count = 0
        for i in range(8):  # Exceed failure threshold
            result = failing_pipeline.process_image_question(self.demo_image, "Test question")
            if "error" in result.answer.lower():
                failure_count += 1
        
        assert failure_count >= 5  # Should have recorded failures
        
        # Check circuit breaker state
        health = failing_pipeline.get_health_status()
        assert health["circuit_breaker_state"] in ["OPEN", "HALF_OPEN"]
        assert health["status"] in ["degraded", "unhealthy"]
    
    def test_concurrent_processing_safety(self):
        """Test thread safety under concurrent load."""
        results = []
        errors = []
        
        def process_request(thread_id):
            try:
                question = f"Thread {thread_id} question"
                result = self.pipeline.process_image_question(self.demo_image, question)
                results.append((thread_id, result))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=process_request, args=(i,))
            threads.append(thread)
        
        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        processing_time = time.time() - start_time
        
        # Verify results
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"
        assert len(errors) == 0, f"Unexpected errors: {errors}"
        assert processing_time < 30, f"Processing took too long: {processing_time}s"
        
        # Check that all threads got valid responses
        for thread_id, result in results:
            assert result.answer is not None
            assert result.latency_ms > 0
            assert 0 <= result.confidence <= 1
    
    def test_cache_security_and_integrity(self):
        """Test cache security and data integrity."""
        # Test cache key security
        sensitive_questions = [
            "password123",
            "admin:secret",
            "private_key_data"
        ]
        
        for question in sensitive_questions:
            result1 = self.pipeline.process_image_question(self.demo_image, question)
            result2 = self.pipeline.process_image_question(self.demo_image, question)
            
            # Should get consistent cached results
            assert result2.metadata.get("cache_used", False) == True
            assert result1.answer == result2.answer
        
        # Test cache size management
        cache_stress_questions = [f"Question {i}" for i in range(100)]
        for question in cache_stress_questions:
            self.pipeline.process_image_question(self.demo_image, question)
        
        # Cache should not grow indefinitely
        stats = self.pipeline.get_stats()
        assert stats["cache_entries"] < 100, "Cache size not properly managed"
    
    def test_memory_management(self):
        """Test memory usage and leak prevention."""
        initial_stats = self.pipeline.get_stats()
        
        # Process many requests
        for i in range(50):
            question = f"Memory test question {i}"
            result = self.pipeline.process_image_question(self.demo_image, question)
            assert result.answer is not None
        
        # Clear cache to test cleanup
        cleared_entries = self.pipeline.clear_cache()
        assert cleared_entries > 0
        
        final_stats = self.pipeline.get_stats()
        assert final_stats["cache_entries"] == 0
    
    def test_error_recovery_and_fallback(self):
        """Test error recovery and fallback mechanisms."""
        # Test graceful degradation
        config = InferenceConfig(model_name="nonexistent-model")
        pipeline = FastVLMCorePipeline(config)
        
        result = pipeline.process_image_question(self.demo_image, "Test question")
        assert result.answer is not None  # Should provide fallback response
        assert "error" not in result.answer.lower() or "apologize" in result.answer.lower()
    
    def test_health_monitoring_accuracy(self):
        """Test health monitoring and metrics accuracy."""
        # Perform several successful operations
        for i in range(5):
            result = self.pipeline.process_image_question(self.demo_image, f"Test {i}")
            assert result.confidence > 0
        
        # Check health metrics
        health = self.pipeline.get_health_status()
        assert health["status"] == "healthy"
        assert health["success_rate_percent"] == 100.0
        assert health["total_requests"] >= 5
        assert health["successful_requests"] >= 5
        assert health["average_latency_ms"] > 0
        
        # Test with some failures
        for i in range(3):
            # Use invalid input to trigger failures
            result = self.pipeline.process_image_question(b"", "Empty image test")
        
        health = self.pipeline.get_health_status()
        assert health["error_count"] >= 3
        assert health["success_rate_percent"] < 100.0
    
    def test_quick_inference_robustness(self):
        """Test robustness of quick inference API."""
        # Test with various input combinations
        test_cases = [
            (self.demo_image, "What is this?", "fast-vlm-base"),
            (self.demo_image, "Describe the image", "fast-vlm-tiny"),
            (create_demo_image(), "Count objects", "fast-vlm-base"),
        ]
        
        for image_data, question, model_name in test_cases:
            result_dict = quick_inference(image_data, question, model_name)
            
            # Verify structure and content
            required_keys = ["answer", "confidence", "latency_ms", "model_used", "timestamp", "metadata"]
            for key in required_keys:
                assert key in result_dict, f"Missing key: {key}"
            
            assert isinstance(result_dict["answer"], str)
            assert 0 <= result_dict["confidence"] <= 1
            assert result_dict["latency_ms"] > 0
            assert result_dict["model_used"] == model_name
    
    def test_text_only_processing_robustness(self):
        """Test text-only processing mode."""
        test_questions = [
            "Hello, what can you do?",
            "What is FastVLM?",
            "How does vision-language modeling work?",
            "",  # Edge case: empty question
            "x" * 2000,  # Edge case: very long question
        ]
        
        for question in test_questions:
            result = self.pipeline.process_text_only(question)
            assert result.answer is not None
            assert isinstance(result.answer, str)
            assert result.metadata.get("mode") == "text_only"
            
            if len(question.strip()) == 0:
                assert "error" in result.answer.lower()
            elif len(question) > 1000:
                assert "error" in result.answer.lower()
            else:
                assert result.confidence > 0


def test_generation2_comprehensive_robustness():
    """Comprehensive test that exercises all robustness features."""
    print("\n🛡️ GENERATION 2 ROBUSTNESS VALIDATION")
    print("=" * 50)
    
    # Run the test suite
    test_instance = TestGeneration2Robustness()
    test_instance.setup_method()
    
    # Test categories
    test_methods = [
        ("Security Validation", test_instance.test_input_validation_security),
        ("Size Limits", test_instance.test_input_size_limits),
        ("Circuit Breaker", test_instance.test_circuit_breaker_functionality),
        ("Thread Safety", test_instance.test_concurrent_processing_safety),
        ("Cache Security", test_instance.test_cache_security_and_integrity),
        ("Memory Management", test_instance.test_memory_management),
        ("Error Recovery", test_instance.test_error_recovery_and_fallback),
        ("Health Monitoring", test_instance.test_health_monitoring_accuracy),
        ("Quick Inference", test_instance.test_quick_inference_robustness),
        ("Text Processing", test_instance.test_text_only_processing_robustness),
    ]
    
    passed_tests = 0
    total_tests = len(test_methods)
    
    for test_name, test_method in test_methods:
        try:
            print(f"\n🧪 Testing {test_name}...")
            start_time = time.time()
            test_method()
            test_time = (time.time() - start_time) * 1000
            print(f"   ✅ {test_name}: PASSED ({test_time:.1f}ms)")
            passed_tests += 1
        except Exception as e:
            print(f"   ❌ {test_name}: FAILED - {str(e)}")
    
    # Final assessment
    success_rate = (passed_tests / total_tests) * 100
    print(f"\n📊 GENERATION 2 ROBUSTNESS RESULTS:")
    print(f"   Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 90:
        print("   🟢 ROBUST - Ready for Generation 3")
        return True
    elif success_rate >= 70:
        print("   🟡 PARTIALLY ROBUST - Needs improvements")
        return False
    else:
        print("   🔴 NOT ROBUST - Major issues detected")
        return False


if __name__ == "__main__":
    test_generation2_comprehensive_robustness()