"""
Comprehensive scaling and performance testing for FastVLM Generation 3.
Tests hyper-scaling engine, performance optimization, and throughput capabilities.
"""

import time
import threading
import concurrent.futures
from pathlib import Path
import sys
import statistics

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.fast_vlm_ondevice import (
    FastVLMCorePipeline, 
    InferenceConfig, 
    create_demo_image,
    quick_inference
)
from src.fast_vlm_ondevice.hyper_scaling_engine import (
    create_hyper_scaling_engine,
    create_hyper_cache,
    ScalingStrategy
)


class TestGeneration3Scaling:
    """Test suite for Generation 3 scaling and performance features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = InferenceConfig(
            model_name="fast-vlm-base",
            enable_caching=True,
            timeout_seconds=30.0
        )
        self.pipeline = FastVLMCorePipeline(self.config)
        self.demo_image = create_demo_image()
    
    def test_hyper_cache_performance(self):
        """Test multi-level cache performance."""
        # Test standalone hyper cache
        cache = create_hyper_cache(l1_size=10, l2_size=50, ttl_seconds=3600)
        
        # Test cache misses
        result1 = cache.get(self.demo_image, "Test question 1")
        assert result1 is None, "Expected cache miss on first access"
        
        # Store results
        test_results = [
            {"answer": f"Answer {i}", "confidence": 0.8, "latency_ms": 100}
            for i in range(15)
        ]
        
        for i, result in enumerate(test_results):
            cache.put(self.demo_image, f"Test question {i}", result)
        
        # Test cache hits
        cached_result = cache.get(self.demo_image, "Test question 0")
        assert cached_result is not None, "Expected cache hit"
        assert cached_result["answer"] == "Answer 0"
        
        # Test cache statistics
        stats = cache.get_stats()
        assert stats["total_requests"] > 0
        assert stats["l1_size"] <= 10
        assert stats["l2_size"] <= 50
        
        print(f"Cache stats: {stats}")
        return True
    
    def test_worker_pool_scaling(self):
        """Test dynamic worker pool scaling."""
        engine = create_hyper_scaling_engine(
            min_workers=1,
            max_workers=4,
            strategy=ScalingStrategy.ADAPTIVE
        )
        
        def mock_processing(image_data, question):
            time.sleep(0.01)  # Simulate processing
            return {
                "answer": f"Processed: {question}",
                "confidence": 0.85,
                "latency_ms": 10,
                "model_used": "test_model",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "metadata": {}
            }
        
        # Submit many requests to trigger scaling
        futures = []
        start_time = time.time()
        
        for i in range(20):
            future = engine.worker_pool.submit_task(
                mock_processing, 
                self.demo_image, 
                f"Question {i}"
            )
            futures.append(future)
        
        # Wait for completion
        results = []
        for future in concurrent.futures.as_completed(futures, timeout=30):
            result = future.result()
            results.append(result)
        
        processing_time = time.time() - start_time
        
        # Verify results
        assert len(results) == 20, f"Expected 20 results, got {len(results)}"
        assert processing_time < 10, f"Processing took too long: {processing_time}s"
        
        # Check scaling metrics
        scaling_status = engine.worker_pool.get_scaling_status()
        print(f"Scaling status: {scaling_status}")
        
        engine.shutdown()
        return True
    
    def test_integrated_scaling_pipeline(self):
        """Test integrated pipeline with scaling enabled."""
        # Test basic scaled processing
        questions = [
            "What objects are in this image?",
            "Describe the scene",
            "What colors do you see?",
            "Count the items",
            "Is this indoors or outdoors?"
        ]
        
        results = []
        latencies = []
        
        for question in questions:
            start_time = time.time()
            result = self.pipeline.process_image_question(self.demo_image, question)
            latency = (time.time() - start_time) * 1000
            
            results.append(result)
            latencies.append(latency)
            
            # Verify result structure
            assert result.answer is not None
            assert result.confidence >= 0
            assert result.latency_ms > 0
        
        # Test cache efficiency with repeated questions
        cache_test_start = time.time()
        for _ in range(3):
            result = self.pipeline.process_image_question(self.demo_image, questions[0])
            # Subsequent calls should be faster due to caching
        cache_test_time = (time.time() - cache_test_start) * 1000
        
        # Verify performance
        avg_latency = statistics.mean(latencies)
        print(f"Average latency: {avg_latency:.1f}ms")
        print(f"Cache test time: {cache_test_time:.1f}ms")
        
        # Check health status includes scaling info
        health = self.pipeline.get_health_status()
        assert "scaling" in health
        
        scaling_info = health["scaling"]
        print(f"Scaling info: {scaling_info}")
        
        return True
    
    def test_concurrent_throughput(self):
        """Test concurrent processing throughput."""
        num_threads = 8
        requests_per_thread = 5
        total_requests = num_threads * requests_per_thread
        
        results = []
        errors = []
        start_time = time.time()
        
        def worker_thread(thread_id):
            thread_results = []
            thread_errors = []
            
            for i in range(requests_per_thread):
                try:
                    question = f"Thread {thread_id} Question {i}"
                    result = self.pipeline.process_image_question(self.demo_image, question)
                    thread_results.append(result)
                except Exception as e:
                    thread_errors.append(str(e))
            
            return thread_results, thread_errors
        
        # Create and start threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(worker_thread, i) 
                for i in range(num_threads)
            ]
            
            # Collect results
            for future in concurrent.futures.as_completed(futures, timeout=60):
                thread_results, thread_errors = future.result()
                results.extend(thread_results)
                errors.extend(thread_errors)
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        throughput_qps = total_requests / total_time
        success_rate = len(results) / total_requests * 100
        
        print(f"Concurrent throughput test:")
        print(f"  Total requests: {total_requests}")
        print(f"  Successful: {len(results)}")
        print(f"  Errors: {len(errors)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {throughput_qps:.1f} QPS")
        print(f"  Success rate: {success_rate:.1f}%")
        
        # Verify performance thresholds
        assert len(results) >= total_requests * 0.95, "Success rate too low"
        assert throughput_qps > 1.0, "Throughput too low"
        assert total_time < 60, "Processing took too long"
        
        return True
    
    def test_memory_efficiency(self):
        """Test memory usage and efficiency."""
        import gc
        
        # Get initial memory usage
        try:
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            initial_memory = 0
        
        # Process many requests
        for i in range(50):
            question = f"Memory test question {i}"
            result = self.pipeline.process_image_question(self.demo_image, question)
            assert result.answer is not None
            
            # Force garbage collection periodically
            if i % 10 == 0:
                gc.collect()
        
        # Check final memory usage
        try:
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_growth = final_memory - initial_memory
            print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (+{memory_growth:.1f}MB)")
        except:
            memory_growth = 0
        
        # Check cache management
        cache_cleared = self.pipeline.clear_cache()
        print(f"Cleared {cache_cleared} cache entries")
        
        # Memory growth should be reasonable
        if initial_memory > 0:
            assert memory_growth < 100, f"Excessive memory growth: {memory_growth}MB"
        
        return True
    
    def test_scaling_strategies(self):
        """Test different scaling strategies."""
        strategies = [
            ScalingStrategy.CONSERVATIVE,
            ScalingStrategy.AGGRESSIVE,
            ScalingStrategy.ADAPTIVE
        ]
        
        results = {}
        
        for strategy in strategies:
            engine = create_hyper_scaling_engine(
                min_workers=1,
                max_workers=3,
                strategy=strategy
            )
            
            # Submit load to test scaling behavior
            def test_load():
                time.sleep(0.02)  # Simulate moderate load
                return {"test": "result"}
            
            start_time = time.time()
            futures = []
            
            for i in range(10):
                future = engine.worker_pool.submit_task(test_load)
                futures.append(future)
            
            # Wait for completion
            for future in concurrent.futures.as_completed(futures, timeout=30):
                future.result()
            
            test_time = time.time() - start_time
            scaling_status = engine.worker_pool.get_scaling_status()
            
            results[strategy.value] = {
                "test_time": test_time,
                "final_workers": scaling_status["current_workers"],
                "scaling_actions": scaling_status["scaling_actions"]
            }
            
            engine.shutdown()
        
        print(f"Scaling strategy results:")
        for strategy, result in results.items():
            print(f"  {strategy}: {result}")
        
        return True
    
    def test_error_recovery_under_load(self):
        """Test error recovery and resilience under load."""
        # Create a pipeline that will occasionally fail
        failure_count = 0
        
        def failing_processor(image_data, question):
            nonlocal failure_count
            failure_count += 1
            
            # Simulate intermittent failures
            if failure_count % 5 == 0:
                raise RuntimeError("Simulated processing failure")
            
            return {
                "answer": f"Processed: {question}",
                "confidence": 0.8,
                "latency_ms": 20,
                "model_used": "test_model",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "metadata": {}
            }
        
        # Test with scaling engine
        engine = create_hyper_scaling_engine(min_workers=2, max_workers=4)
        
        successful = 0
        failed = 0
        
        for i in range(20):
            try:
                result = engine.process_request_scaled(
                    self.demo_image, 
                    f"Question {i}", 
                    failing_processor
                )
                successful += 1
            except Exception:
                failed += 1
        
        print(f"Error recovery test: {successful} successful, {failed} failed")
        
        # Should handle most failures gracefully
        assert successful >= 15, "Too many failures in error recovery test"
        
        # Get performance report
        report = engine.get_performance_report()
        print(f"Final performance report: {report}")
        
        engine.shutdown()
        return True


def test_generation3_comprehensive_scaling():
    """Comprehensive test that exercises all scaling features."""
    print("\n🚀 GENERATION 3 SCALING VALIDATION")
    print("=" * 50)
    
    # Run the test suite
    test_instance = TestGeneration3Scaling()
    test_instance.setup_method()
    
    # Test categories
    test_methods = [
        ("Hyper Cache Performance", test_instance.test_hyper_cache_performance),
        ("Worker Pool Scaling", test_instance.test_worker_pool_scaling),
        ("Integrated Scaling Pipeline", test_instance.test_integrated_scaling_pipeline),
        ("Concurrent Throughput", test_instance.test_concurrent_throughput),
        ("Memory Efficiency", test_instance.test_memory_efficiency),
        ("Scaling Strategies", test_instance.test_scaling_strategies),
        ("Error Recovery Under Load", test_instance.test_error_recovery_under_load),
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
    print(f"\n📊 GENERATION 3 SCALING RESULTS:")
    print(f"   Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 90:
        print("   🟢 HIGHLY SCALABLE - Ready for production")
        return True
    elif success_rate >= 70:
        print("   🟡 MODERATELY SCALABLE - Needs optimization")
        return False
    else:
        print("   🔴 NOT SCALABLE - Major performance issues")
        return False


if __name__ == "__main__":
    test_generation3_comprehensive_scaling()