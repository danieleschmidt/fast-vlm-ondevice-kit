#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite for FastVLM On-Device Kit
Tests all three generations: Basic, Robust, and Scaling
"""

import sys
import time
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fast_vlm_ondevice.core_pipeline import FastVLMCorePipeline, InferenceConfig
from fast_vlm_ondevice.enhanced_security_framework import create_enhanced_validator
from fast_vlm_ondevice.hyper_scaling_engine import create_hyper_scaling_engine, ScalingStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class FastVLMIntegrationTests:
    """Comprehensive integration test suite."""
    
    def __init__(self):
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        
    def run_test(self, test_name: str, test_func):
        """Run a single test and record results."""
        self.total_tests += 1
        print(f"\n🧪 Running Test: {test_name}")
        print("-" * 50)
        
        try:
            start_time = time.time()
            result = test_func()
            duration = time.time() - start_time
            
            if result:
                self.passed_tests += 1
                status = "✅ PASSED"
                self.test_results.append({"name": test_name, "status": "PASSED", "duration": duration})
            else:
                status = "❌ FAILED"
                self.test_results.append({"name": test_name, "status": "FAILED", "duration": duration})
                
            print(f"{status} ({duration:.3f}s)")
            return result
            
        except Exception as e:
            status = "💥 ERROR"
            self.test_results.append({"name": test_name, "status": "ERROR", "duration": 0, "error": str(e)})
            print(f"{status}: {str(e)}")
            return False
    
    def test_generation1_basic_functionality(self) -> bool:
        """Test Generation 1: Basic functionality."""
        print("Testing basic FastVLM inference pipeline...")
        
        # Test with proper demo image data
        config = InferenceConfig(model_name="fast-vlm-base", enable_caching=True)
        pipeline = FastVLMCorePipeline(config)
        
        # Create valid demo image data (larger than validation threshold)
        demo_image = b"DEMO_IMAGE_DATA" + b"x" * 1000  # 1KB+ demo image
        test_question = "What objects are in this image?"
        
        # Process image-question pair
        result = pipeline.process_image_question(demo_image, test_question)
        
        # Validate result
        if (hasattr(result, 'answer') and 
            hasattr(result, 'confidence') and 
            hasattr(result, 'latency_ms') and
            len(result.answer) > 0):
            
            print(f"   📝 Answer: {result.answer}")
            print(f"   🎯 Confidence: {result.confidence:.2f}")
            print(f"   ⚡ Latency: {result.latency_ms:.1f}ms")
            
            # Test caching
            cached_result = pipeline.process_image_question(demo_image, test_question)
            cache_hit = cached_result.metadata.get("cache_used", False)
            print(f"   💾 Cache Hit: {'Yes' if cache_hit else 'No'}")
            
            # Test health status
            health = pipeline.get_health_status()
            print(f"   💊 Health Status: {health['status']}")
            
            return True
        
        return False
    
    def test_generation2_security_robustness(self) -> bool:
        """Test Generation 2: Security and robustness."""
        print("Testing enhanced security framework...")
        
        validator = create_enhanced_validator()
        
        # Test valid request
        valid_image = b"VALID_IMAGE_DATA" + b"x" * 1000
        valid_question = "What is in this image?"
        
        is_safe, message, incidents = validator.validate_request(valid_image, valid_question)
        if not is_safe:
            print(f"   ❌ Valid request marked as unsafe: {message}")
            return False
        
        print(f"   ✅ Valid request: {message}")
        
        # Test malicious content detection
        malicious_image = b"<script>alert('xss')</script>" + b"x" * 1000
        is_safe, message, incidents = validator.validate_request(malicious_image, valid_question)
        
        if is_safe:
            print("   ❌ Failed to detect malicious image content")
            return False
        
        print(f"   🛡️ Blocked malicious content: {message}")
        print(f"   📊 Security incidents detected: {len(incidents)}")
        
        # Test security status
        security_status = validator.get_security_status()
        print(f"   🔍 Total incidents: {security_status['total_incidents']}")
        print(f"   🚫 Blocked incidents: {security_status['blocked_incidents']}")
        
        return True
    
    def test_generation3_scaling_performance(self) -> bool:
        """Test Generation 3: Scaling and performance optimization."""
        print("Testing hyper scaling engine...")
        
        # Create scaling engine
        engine = create_hyper_scaling_engine(
            min_workers=1,
            max_workers=4,
            cache_l1_size=10,
            cache_l2_size=50,
            strategy=ScalingStrategy.ADAPTIVE
        )
        
        # Mock processing function
        def mock_pipeline_func(image_data: bytes, question: str):
            """Mock processing to simulate pipeline behavior."""
            time.sleep(0.01)  # Simulate processing time
            return {
                "answer": f"Processed question: {question[:30]}...",
                "confidence": 0.85,
                "latency_ms": 10.0,
                "model_used": "mock_model",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "metadata": {"mock": True}
            }
        
        # Test scaled processing
        test_image = b"TEST_IMAGE_FOR_SCALING" + b"x" * 1000
        test_question = "Test scaling performance"
        
        # First request (should be processed)
        result1 = engine.process_request_scaled(test_image, test_question, mock_pipeline_func)
        if not result1.get("answer"):
            print("   ❌ Failed to get answer from scaled processing")
            return False
        
        print(f"   🚀 Scaled Answer: {result1['answer']}")
        print(f"   👥 Workers: {result1['metadata'].get('workers_used', 'unknown')}")
        
        # Second request (should hit cache)
        result2 = engine.process_request_scaled(test_image, test_question, mock_pipeline_func)
        cache_used = result2.get("metadata", {}).get("cache_used", False)
        print(f"   💾 Cache Hit: {'Yes' if cache_used else 'No'}")
        
        # Test performance report
        report = engine.get_performance_report()
        print(f"   📊 Total Processed: {report['engine_stats']['total_processed']}")
        print(f"   📈 Cache Hit Rate: {report['cache_performance']['overall_hit_rate_percent']:.1f}%")
        print(f"   🔄 Auto Scaling: {'Enabled' if report['auto_scaling_enabled'] else 'Disabled'}")
        
        # Cleanup
        engine.shutdown()
        
        return True
    
    def test_cross_generation_integration(self) -> bool:
        """Test integration across all three generations."""
        print("Testing cross-generation integration...")
        
        # Initialize all components
        config = InferenceConfig(model_name="fast-vlm-base", enable_caching=True)
        pipeline = FastVLMCorePipeline(config)
        validator = create_enhanced_validator()
        engine = create_hyper_scaling_engine(min_workers=1, max_workers=2, strategy=ScalingStrategy.ADAPTIVE)
        
        # Test image and question
        test_image = b"INTEGRATION_TEST_IMAGE" + b"x" * 1000
        test_question = "Integration test: what do you see?"
        
        # Step 1: Security validation
        is_safe, message, incidents = validator.validate_request(test_image, test_question)
        if not is_safe:
            print(f"   ❌ Security validation failed: {message}")
            return False
        
        print(f"   🛡️ Security: {message}")
        
        # Step 2: Scaled processing with security
        def secure_pipeline_func(image_data: bytes, question: str):
            # Validate again within pipeline
            safe, _, _ = validator.validate_request(image_data, question)
            if not safe:
                raise ValueError("Security validation failed in pipeline")
            
            # Process with core pipeline
            return pipeline.process_image_question(image_data, question)
        
        try:
            scaled_result = engine.process_request_scaled(test_image, test_question, secure_pipeline_func)
            
            print(f"   🚀 Integrated Answer: {scaled_result.get('answer', 'No answer')}")
            print(f"   ⚡ Total Latency: {scaled_result.get('latency_ms', 0):.1f}ms")
            
            # Step 3: Health check across all systems
            pipeline_health = pipeline.get_health_status()
            security_status = validator.get_security_status()
            performance_report = engine.get_performance_report()
            
            print(f"   💊 Pipeline Health: {pipeline_health['status']}")
            print(f"   🔒 Security Status: {security_status['status']}")
            print(f"   📊 Scaling Workers: {performance_report['worker_pool']['current_workers']}")
            
            # Cleanup
            engine.shutdown()
            
            return True
            
        except Exception as e:
            print(f"   ❌ Integration error: {str(e)}")
            engine.shutdown()
            return False
    
    def test_performance_benchmarks(self) -> bool:
        """Test performance benchmarks and SLA compliance."""
        print("Testing performance benchmarks...")
        
        config = InferenceConfig(model_name="fast-vlm-base", enable_caching=True)
        pipeline = FastVLMCorePipeline(config)
        
        # Performance targets
        target_latency_ms = 250  # FastVLM target: <250ms
        target_success_rate = 95  # 95% success rate
        
        test_image = b"PERFORMANCE_TEST_IMAGE" + b"x" * 1000
        test_questions = [
            "What objects are visible?",
            "Describe the scene",
            "What colors do you see?",
            "Count the items",
            "Is this indoors or outdoors?"
        ]
        
        latencies = []
        successes = 0
        
        for i, question in enumerate(test_questions):
            start_time = time.time()
            result = pipeline.process_image_question(test_image, question)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            if hasattr(result, 'answer') and len(result.answer) > 0:
                successes += 1
        
        # Calculate metrics
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        success_rate = (successes / len(test_questions)) * 100
        
        print(f"   ⚡ Average Latency: {avg_latency:.1f}ms")
        print(f"   📊 P95 Latency: {p95_latency:.1f}ms")
        print(f"   ✅ Success Rate: {success_rate:.1f}%")
        
        # Check SLA compliance
        latency_ok = p95_latency <= target_latency_ms
        success_ok = success_rate >= target_success_rate
        
        print(f"   🎯 Latency SLA: {'✅ Met' if latency_ok else '❌ Missed'} (target: <{target_latency_ms}ms)")
        print(f"   🎯 Success SLA: {'✅ Met' if success_ok else '❌ Missed'} (target: >{target_success_rate}%)")
        
        return latency_ok and success_ok
    
    def run_all_tests(self):
        """Run all integration tests."""
        print("🚀 FastVLM On-Device Kit - Comprehensive Integration Tests")
        print("=" * 60)
        print("Testing all three generations: Basic → Robust → Scaling")
        
        # Generation 1: Basic Functionality
        self.run_test("Generation 1: Basic Functionality", self.test_generation1_basic_functionality)
        
        # Generation 2: Security & Robustness  
        self.run_test("Generation 2: Security & Robustness", self.test_generation2_security_robustness)
        
        # Generation 3: Scaling & Performance
        self.run_test("Generation 3: Scaling & Performance", self.test_generation3_scaling_performance)
        
        # Cross-Generation Integration
        self.run_test("Cross-Generation Integration", self.test_cross_generation_integration)
        
        # Performance Benchmarks
        self.run_test("Performance Benchmarks", self.test_performance_benchmarks)
        
        # Print final results
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print comprehensive test summary."""
        print("\n" + "=" * 60)
        print("🧪 TEST SUMMARY")
        print("=" * 60)
        
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.total_tests - self.passed_tests}")
        print(f"Success Rate: {(self.passed_tests / self.total_tests * 100):.1f}%")
        
        print("\nDetailed Results:")
        for result in self.test_results:
            status_emoji = {"PASSED": "✅", "FAILED": "❌", "ERROR": "💥"}
            emoji = status_emoji.get(result["status"], "❓")
            print(f"  {emoji} {result['name']} ({result.get('duration', 0):.3f}s)")
            if "error" in result:
                print(f"      Error: {result['error']}")
        
        if self.passed_tests == self.total_tests:
            print("\n🎉 ALL TESTS PASSED! FastVLM On-Device Kit is production ready!")
        else:
            print(f"\n⚠️ {self.total_tests - self.passed_tests} test(s) failed. Review and fix issues.")
        
        print("=" * 60)


if __name__ == "__main__":
    # Run comprehensive integration tests
    test_suite = FastVLMIntegrationTests()
    test_suite.run_all_tests()