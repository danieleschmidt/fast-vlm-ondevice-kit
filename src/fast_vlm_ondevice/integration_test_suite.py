"""
Comprehensive integration test suite for FastVLM components.

Tests all core functionality with real-world scenarios and edge cases.
Validates quality gates and production readiness.
"""

import unittest
import asyncio
import time
import logging
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock, patch
import sys
import os

# Add the src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import our modules
from fast_vlm_ondevice.converter import FastVLMConverter
from fast_vlm_ondevice.real_time_mobile_optimizer import RealTimeMobileOptimizer, MobileOptimizationConfig
from fast_vlm_ondevice.core_pipeline import FastVLMPipeline, InferenceConfig
from fast_vlm_ondevice.health import HealthChecker
from fast_vlm_ondevice.monitoring import MetricsCollector, PerformanceProfiler
from fast_vlm_ondevice.security import SecurityScanner
from fast_vlm_ondevice.validation import ValidationLevel

logger = logging.getLogger(__name__)


class FastVLMIntegrationTestSuite(unittest.TestCase):
    """Comprehensive integration tests for FastVLM system."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_models_dir = Path(cls.temp_dir) / "models"
        cls.test_models_dir.mkdir(exist_ok=True)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        
        print(f"Integration test suite initialized with temp dir: {cls.temp_dir}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        if hasattr(cls, 'temp_dir'):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def setUp(self):
        """Set up individual test."""
        self.start_time = time.time()
    
    def tearDown(self):
        """Clean up individual test."""
        test_time = time.time() - self.start_time
        print(f"Test completed in {test_time:.2f}s")
    
    def test_converter_initialization(self):
        """Test FastVLM converter initialization."""
        try:
            converter = FastVLMConverter(validation_level=ValidationLevel.BALANCED)
            self.assertIsNotNone(converter)
            self.assertIsNotNone(converter.session_id)
            self.assertTrue(converter.adaptive_quantization)
            print("✅ Converter initialization test passed")
        except Exception as e:
            self.fail(f"Converter initialization failed: {e}")
    
    def test_mobile_optimizer_basic_functionality(self):
        """Test mobile optimizer basic operations."""
        config = MobileOptimizationConfig(
            target_latency_ms=200,
            max_memory_mb=400,
            quantization_strategy="adaptive"
        )
        optimizer = RealTimeMobileOptimizer(config)
        
        # Test optimization
        model_data = {
            "size_mb": 350,
            "complexity_score": 0.6,
            "target_device": "iphone_15_pro"
        }
        
        result = optimizer.optimize_for_mobile(model_data)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result.optimized, bool)
        self.assertGreater(result.latency_ms, 0)
        self.assertGreater(result.memory_usage_mb, 0)
        print(f"✅ Mobile optimization test passed - Latency: {result.latency_ms}ms")
    
    def test_core_pipeline_inference(self):
        """Test core pipeline inference functionality."""
        config = InferenceConfig(
            batch_size=1,
            max_sequence_length=77,
            use_cache=True
        )
        
        pipeline = FastVLMPipeline(config)
        
        # Mock image and question data
        mock_image_data = b"fake_image_data" * 1000  # 13KB of mock data
        mock_question = "What is in this image?"
        
        result = pipeline.process_vlm_query(mock_image_data, mock_question)
        
        self.assertIsNotNone(result)
        self.assertIn("result", result)
        self.assertIn("confidence", result)
        self.assertIn("processing_time_ms", result)
        print(f"✅ Core pipeline test passed - Time: {result['processing_time_ms']:.1f}ms")
    
    def test_health_monitoring_system(self):
        """Test health monitoring and quality gates."""
        health_checker = HealthChecker()
        
        # Run comprehensive health check
        health_result = health_checker.run_comprehensive_health_check()
        
        self.assertIsNotNone(health_result)
        self.assertIn("overall_status", health_result)
        self.assertIn("checks", health_result)
        self.assertIn("timestamp", health_result)
        
        # Verify critical checks pass
        critical_checks = [check for check in health_result["checks"] 
                          if check.get("level") == "critical"]
        passed_critical = [check for check in critical_checks 
                          if check.get("status") == "pass"]
        
        self.assertEqual(len(passed_critical), len(critical_checks), 
                        "All critical health checks must pass")
        print(f"✅ Health monitoring test passed - {len(passed_critical)} critical checks passed")
    
    def test_performance_monitoring(self):
        """Test performance monitoring and metrics collection."""
        metrics_collector = MetricsCollector()
        profiler = PerformanceProfiler(metrics_collector, "test-session")
        
        # Simulate some operations
        with profiler.profile_operation("test_inference"):
            time.sleep(0.1)  # Simulate inference time
            
        with profiler.profile_operation("test_preprocessing"):
            time.sleep(0.05)  # Simulate preprocessing time
        
        # Get performance metrics
        metrics = profiler.get_performance_summary()
        
        self.assertIsNotNone(metrics)
        self.assertIn("operations", metrics)
        self.assertIn("test_inference", metrics["operations"])
        self.assertIn("test_preprocessing", metrics["operations"])
        
        # Verify timing accuracy
        inference_time = metrics["operations"]["test_inference"]["avg_duration_ms"]
        self.assertGreater(inference_time, 90)  # Should be around 100ms
        self.assertLess(inference_time, 150)    # With some tolerance
        
        print(f"✅ Performance monitoring test passed - Inference: {inference_time:.1f}ms")
    
    def test_security_validation(self):
        """Test security scanning and validation."""
        scanner = SecurityScanner()
        
        # Test safe input
        safe_image = b"safe_image_data" * 100
        safe_question = "What objects are visible in this image?"
        
        safe_result = scanner.scan_input(safe_image, safe_question)
        self.assertTrue(safe_result["safe"])
        self.assertEqual(safe_result["risk_level"], "low")
        
        # Test potentially unsafe input
        unsafe_question = "<script>alert('xss')</script>What is this?"
        unsafe_result = scanner.scan_input(safe_image, unsafe_question)
        self.assertFalse(unsafe_result["safe"])
        self.assertGreater(unsafe_result["risk_level"], "low")
        
        print("✅ Security validation test passed")
    
    def test_caching_and_optimization(self):
        """Test caching system and optimization features."""
        config = InferenceConfig(use_cache=True, cache_size=50)
        pipeline = FastVLMPipeline(config)
        
        # Same input should be faster on second call
        mock_image = b"cached_test_image" * 500
        mock_question = "Test question for caching"
        
        # First call
        start_time = time.time()
        result1 = pipeline.process_vlm_query(mock_image, mock_question)
        first_call_time = time.time() - start_time
        
        # Second call (should use cache)
        start_time = time.time()
        result2 = pipeline.process_vlm_query(mock_image, mock_question)
        second_call_time = time.time() - start_time
        
        # Verify results are consistent
        self.assertEqual(result1["result"], result2["result"])
        
        # Second call should be significantly faster
        self.assertLess(second_call_time, first_call_time * 0.5)
        
        print(f"✅ Caching test passed - First: {first_call_time*1000:.1f}ms, Cached: {second_call_time*1000:.1f}ms")
    
    def test_error_recovery_mechanisms(self):
        """Test error recovery and resilience features."""
        converter = FastVLMConverter(validation_level=ValidationLevel.STRICT)
        
        # Test recovery from invalid model path
        invalid_model_path = "/nonexistent/model/path.pth"
        
        # This should use fallback mechanisms instead of crashing
        try:
            result = converter.load_pytorch_model(invalid_model_path)
            # Should either succeed with fallback or handle gracefully
            self.assertIsNotNone(result)
        except Exception as e:
            # If it does throw, should be a handled exception
            self.assertIn("fallback", str(e).lower())
        
        print("✅ Error recovery test passed")
    
    def test_scalability_and_concurrency(self):
        """Test system behavior under concurrent load."""
        import concurrent.futures
        
        def run_inference():
            config = InferenceConfig(batch_size=1)
            pipeline = FastVLMPipeline(config)
            mock_image = b"concurrent_test" * 200
            mock_question = f"Question {time.time()}"
            return pipeline.process_vlm_query(mock_image, mock_question)
        
        # Run multiple concurrent inferences
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_inference) for _ in range(8)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Verify all completed successfully
        self.assertEqual(len(results), 8)
        for result in results:
            self.assertIsNotNone(result)
            self.assertIn("result", result)
        
        print(f"✅ Concurrency test passed - {len(results)} parallel inferences completed")
    
    def test_memory_management(self):
        """Test memory usage and cleanup."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and use multiple components
        components = []
        for i in range(5):
            config = InferenceConfig(batch_size=1)
            pipeline = FastVLMPipeline(config)
            components.append(pipeline)
            
            # Run some operations
            mock_image = b"memory_test" * 1000
            mock_question = f"Memory test question {i}"
            pipeline.process_vlm_query(mock_image, mock_question)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Clean up
        del components
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = peak_memory - initial_memory
        memory_cleanup = peak_memory - final_memory
        
        # Memory should not grow excessively
        self.assertLess(memory_growth, 200, "Memory growth should be reasonable")
        
        print(f"✅ Memory management test passed - Growth: {memory_growth:.1f}MB, Cleanup: {memory_cleanup:.1f}MB")
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Initialize all components
        converter = FastVLMConverter(validation_level=ValidationLevel.BALANCED)
        optimizer = RealTimeMobileOptimizer()
        health_checker = HealthChecker()
        
        # Step 1: Health check
        health_result = health_checker.quick_health_check()
        self.assertEqual(health_result["status"], "healthy")
        
        # Step 2: Model optimization
        model_data = {
            "size_mb": 400,
            "complexity_score": 0.5,
            "target_device": "iphone_14"
        }
        
        optimization_result = optimizer.optimize_for_mobile(model_data)
        self.assertTrue(optimization_result.optimized)
        
        # Step 3: Core inference
        config = InferenceConfig(
            batch_size=1,
            use_cache=True,
            enable_monitoring=True
        )
        pipeline = FastVLMPipeline(config)
        
        mock_image = b"end_to_end_test" * 800
        mock_question = "Describe what you see in this image."
        
        inference_result = pipeline.process_vlm_query(mock_image, mock_question)
        
        # Verify complete workflow
        self.assertIsNotNone(inference_result)
        self.assertIn("result", inference_result)
        self.assertLess(inference_result["processing_time_ms"], 500)  # Should be fast
        
        print("✅ End-to-end workflow test passed")
    
    def test_quality_gates_validation(self):
        """Test all quality gates pass requirements."""
        # Performance requirements
        config = InferenceConfig(batch_size=1)
        pipeline = FastVLMPipeline(config)
        
        mock_image = b"quality_gate_test" * 600
        mock_question = "Quality gate validation question"
        
        # Test latency requirement
        start_time = time.time()
        result = pipeline.process_vlm_query(mock_image, mock_question)
        latency_ms = (time.time() - start_time) * 1000
        
        self.assertLess(latency_ms, 300, "Latency must be under 300ms")
        
        # Test accuracy/confidence requirement
        confidence = result.get("confidence", 0)
        self.assertGreater(confidence, 0.7, "Confidence must be above 70%")
        
        # Test security requirement
        scanner = SecurityScanner()
        security_result = scanner.scan_input(mock_image, mock_question)
        self.assertTrue(security_result["safe"], "Input must pass security validation")
        
        # Test memory requirement
        import psutil
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        self.assertLess(memory_mb, 1000, "Memory usage must be under 1GB")
        
        print("✅ Quality gates validation passed")


class TestRunner:
    """Test runner with comprehensive reporting."""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def run_all_tests(self):
        """Run all integration tests with detailed reporting."""
        print("🚀 Starting FastVLM Integration Test Suite")
        print("=" * 60)
        
        self.start_time = time.time()
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(FastVLMIntegrationTestSuite)
        
        # Run tests with custom result handling
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(suite)
        
        self.end_time = time.time()
        
        # Generate comprehensive report
        self._generate_report(result)
        
        return result.wasSuccessful()
    
    def _generate_report(self, result):
        """Generate comprehensive test report."""
        total_time = self.end_time - self.start_time
        
        print("\n" + "=" * 60)
        print("📊 INTEGRATION TEST SUMMARY")
        print("=" * 60)
        
        print(f"Total Tests: {result.testsRun}")
        print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
        print(f"Failed: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
        
        if result.failures:
            print("\n❌ FAILURES:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")
        
        if result.errors:
            print("\n💥 ERRORS:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")
        
        if result.wasSuccessful():
            print("\n🎉 ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION!")
        else:
            print("\n⚠️  SOME TESTS FAILED - REVIEW BEFORE DEPLOYMENT")
        
        print("=" * 60)


def main():
    """Main test execution function."""
    runner = TestRunner()
    success = runner.run_all_tests()
    
    # Return appropriate exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()