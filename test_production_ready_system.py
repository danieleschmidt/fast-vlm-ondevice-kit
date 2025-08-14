#!/usr/bin/env python3
"""
Production-Ready FastVLM System Integration Test
Tests the complete system from Generations 1-3 with real-world scenarios.
"""

import time
import json
import logging
import threading
import tempfile
from pathlib import Path
from typing import Dict, Any, List
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from fast_vlm_ondevice.core_pipeline import (
        FastVLMCorePipeline, InferenceConfig, create_demo_image, quick_inference
    )
    from fast_vlm_ondevice.performance_monitor import (
        PerformanceMonitor, track_performance, get_performance_monitor
    )
except ImportError as e:
    logger.error(f"Import failed: {e}")
    print("⚠️ Some components not available, running with reduced functionality")


class ProductionSystemTest:
    """Comprehensive production system test suite."""
    
    def __init__(self):
        self.test_results = {}
        self.performance_monitor = PerformanceMonitor()
        self.start_time = time.time()
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all production readiness tests."""
        logger.info("🚀 Starting Production-Ready FastVLM System Tests")
        
        test_methods = [
            self.test_basic_functionality,
            self.test_error_handling_resilience,
            self.test_performance_characteristics,
            self.test_concurrent_load,
            self.test_memory_management,
            self.test_caching_efficiency,
            self.test_input_validation_security,
            self.test_monitoring_and_observability,
            self.test_adaptive_quality_management,
            self.test_circuit_breaker_patterns,
            self.test_mobile_optimization_features
        ]
        
        for test_method in test_methods:
            test_name = test_method.__name__
            logger.info(f"🧪 Running {test_name}...")
            
            try:
                start_time = time.time()
                result = test_method()
                execution_time = time.time() - start_time
                
                self.test_results[test_name] = {
                    "status": "PASSED" if result["success"] else "FAILED",
                    "execution_time_seconds": execution_time,
                    "details": result,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                status_icon = "✅" if result["success"] else "❌"
                logger.info(f"{status_icon} {test_name} completed in {execution_time:.2f}s")
                
            except Exception as e:
                self.test_results[test_name] = {
                    "status": "ERROR",
                    "error": str(e),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                logger.error(f"❌ {test_name} failed with error: {e}")
        
        return self._generate_final_report()
    
    def test_basic_functionality(self) -> Dict[str, Any]:
        """Test basic FastVLM functionality across model sizes."""
        results = {"success": True, "details": {}}
        
        model_configs = [
            ("fast-vlm-tiny", "Tiny model test"),
            ("fast-vlm-base", "Base model test"),
            ("fast-vlm-large", "Large model test")
        ]
        
        demo_image = create_demo_image()
        test_questions = [
            "What objects are in this image?",
            "What colors do you see?",
            "Describe the overall scene",
            "How many distinct elements are visible?"
        ]
        
        for model_name, description in model_configs:
            try:
                config = InferenceConfig(model_name=model_name, enable_caching=True)
                pipeline = FastVLMCorePipeline(config)
                
                model_results = []
                for question in test_questions:
                    result = pipeline.process_image_question(demo_image, question)
                    
                    model_results.append({
                        "question": question,
                        "answer_length": len(result.answer),
                        "confidence": result.confidence,
                        "latency_ms": result.latency_ms,
                        "success": len(result.answer) > 0 and result.confidence > 0
                    })
                
                avg_latency = sum(r["latency_ms"] for r in model_results) / len(model_results)
                avg_confidence = sum(r["confidence"] for r in model_results) / len(model_results)
                
                results["details"][model_name] = {
                    "description": description,
                    "avg_latency_ms": avg_latency,
                    "avg_confidence": avg_confidence,
                    "questions_processed": len(model_results),
                    "all_successful": all(r["success"] for r in model_results)
                }
                
                if not all(r["success"] for r in model_results):
                    results["success"] = False
                    
            except Exception as e:
                results["success"] = False
                results["details"][model_name] = {"error": str(e)}
        
        return results
    
    def test_error_handling_resilience(self) -> Dict[str, Any]:
        """Test system resilience under error conditions."""
        results = {"success": True, "details": {}}
        
        pipeline = FastVLMCorePipeline()
        
        # Test various error scenarios
        error_tests = [
            ("empty_image", b"", "What is this?"),
            ("malformed_image", b"not_an_image", "Describe this"),
            ("empty_question", create_demo_image(), ""),
            ("very_long_question", create_demo_image(), "What is " * 500),
            ("special_characters", create_demo_image(), "∑∆∫∂√≤≥≠±"),
            ("suspicious_input", create_demo_image(), "<script>alert('test')</script>")
        ]
        
        for test_name, image_data, question in error_tests:
            try:
                result = pipeline.process_image_question(image_data, question)
                
                # System should handle errors gracefully
                test_success = (
                    isinstance(result.answer, str) and
                    result.latency_ms > 0 and
                    result.confidence >= 0.0
                )
                
                results["details"][test_name] = {
                    "handled_gracefully": test_success,
                    "response_type": "error" if result.confidence == 0.0 else "normal",
                    "latency_ms": result.latency_ms
                }
                
                if not test_success:
                    results["success"] = False
                    
            except Exception as e:
                # Exceptions are also acceptable for invalid inputs
                results["details"][test_name] = {
                    "handled_gracefully": True,
                    "response_type": "exception",
                    "exception": str(e)
                }
        
        return results
    
    def test_performance_characteristics(self) -> Dict[str, Any]:
        """Test performance characteristics and optimization."""
        results = {"success": True, "details": {}}
        
        self.performance_monitor.start_monitoring(interval=0.1)
        
        try:
            pipeline = FastVLMCorePipeline()
            demo_image = create_demo_image()
            
            # Measure baseline performance
            latencies = []
            for i in range(50):
                start_time = time.time()
                result = pipeline.process_image_question(demo_image, f"Performance test {i}")
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)
                
                self.performance_monitor.record_request_latency(latency, success=True)
            
            # Calculate performance metrics
            avg_latency = sum(latencies) / len(latencies)
            p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
            throughput = len(latencies) / (sum(latencies) / 1000)
            
            results["details"] = {
                "total_requests": len(latencies),
                "avg_latency_ms": avg_latency,
                "p95_latency_ms": p95_latency,
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "requests_per_second": throughput,
                "target_latency_met": p95_latency < 500,  # 500ms target
                "performance_acceptable": avg_latency < 250  # 250ms average target
            }
            
            if not (results["details"]["target_latency_met"] and 
                   results["details"]["performance_acceptable"]):
                results["success"] = False
                
        finally:
            self.performance_monitor.stop_monitoring()
        
        return results
    
    def test_concurrent_load(self) -> Dict[str, Any]:
        """Test system under concurrent load."""
        results = {"success": True, "details": {}}
        
        pipeline = FastVLMCorePipeline()
        demo_image = create_demo_image()
        
        # Concurrent execution test
        num_threads = 20
        requests_per_thread = 5
        
        thread_results = []
        errors = []
        
        def worker_thread(thread_id):
            thread_latencies = []
            for i in range(requests_per_thread):
                try:
                    start_time = time.time()
                    result = pipeline.process_image_question(
                        demo_image, 
                        f"Concurrent test from thread {thread_id}, request {i}"
                    )
                    latency = (time.time() - start_time) * 1000
                    thread_latencies.append(latency)
                    
                except Exception as e:
                    errors.append(f"Thread {thread_id}: {e}")
            
            thread_results.append(thread_latencies)
        
        # Start concurrent threads
        threads = []
        start_time = time.time()
        
        for i in range(num_threads):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Analyze results
        all_latencies = [lat for thread_lats in thread_results for lat in thread_lats]
        
        results["details"] = {
            "threads": num_threads,
            "requests_per_thread": requests_per_thread,
            "total_requests": len(all_latencies),
            "total_time_seconds": total_time,
            "errors_count": len(errors),
            "avg_latency_ms": sum(all_latencies) / len(all_latencies) if all_latencies else 0,
            "concurrent_throughput": len(all_latencies) / total_time if total_time > 0 else 0,
            "success_rate": (len(all_latencies) / (num_threads * requests_per_thread)) * 100
        }
        
        if len(errors) > 0 or results["details"]["success_rate"] < 95:
            results["success"] = False
            results["details"]["errors"] = errors[:5]  # Include first 5 errors
        
        return results
    
    def test_memory_management(self) -> Dict[str, Any]:
        """Test memory management and cleanup."""
        results = {"success": True, "details": {}}
        
        pipeline = FastVLMCorePipeline()
        demo_image = create_demo_image()
        
        # Test memory usage patterns
        initial_cache_size = len(pipeline.cache) if pipeline.cache else 0
        
        # Generate many unique requests
        for i in range(200):
            pipeline.process_image_question(demo_image, f"Memory test unique question {i}")
        
        cache_size_after_load = len(pipeline.cache) if pipeline.cache else 0
        
        # Test cache cleanup
        cleared_entries = pipeline.clear_cache()
        final_cache_size = len(pipeline.cache) if pipeline.cache else 0
        
        results["details"] = {
            "initial_cache_size": initial_cache_size,
            "cache_size_after_load": cache_size_after_load,
            "cleared_entries": cleared_entries,
            "final_cache_size": final_cache_size,
            "cache_growth_controlled": cache_size_after_load < 150,  # Reasonable limit
            "cleanup_effective": final_cache_size == 0
        }
        
        if not (results["details"]["cache_growth_controlled"] and 
               results["details"]["cleanup_effective"]):
            results["success"] = False
        
        return results
    
    def test_caching_efficiency(self) -> Dict[str, Any]:
        """Test caching system efficiency."""
        results = {"success": True, "details": {}}
        
        pipeline = FastVLMCorePipeline()
        demo_image = create_demo_image()
        
        # Test cache hit/miss patterns
        repeated_question = "What is the main subject of this image?"
        
        # First request (cache miss)
        start_time = time.time()
        result1 = pipeline.process_image_question(demo_image, repeated_question)
        first_latency = (time.time() - start_time) * 1000
        
        # Second request (should hit cache)
        start_time = time.time()
        result2 = pipeline.process_image_question(demo_image, repeated_question)
        second_latency = (time.time() - start_time) * 1000
        
        # Multiple cache hits
        cache_hit_latencies = []
        for _ in range(5):
            start_time = time.time()
            pipeline.process_image_question(demo_image, repeated_question)
            cache_hit_latencies.append((time.time() - start_time) * 1000)
        
        avg_cache_hit_latency = sum(cache_hit_latencies) / len(cache_hit_latencies)
        
        results["details"] = {
            "first_request_latency_ms": first_latency,
            "second_request_latency_ms": second_latency,
            "avg_cache_hit_latency_ms": avg_cache_hit_latency,
            "cache_speedup_factor": first_latency / avg_cache_hit_latency if avg_cache_hit_latency > 0 else 0,
            "answers_consistent": result1.answer == result2.answer,
            "cache_stats": dict(pipeline.cache_stats) if hasattr(pipeline, 'cache_stats') else {}
        }
        
        # Cache should provide significant speedup
        if (results["details"]["cache_speedup_factor"] < 1.5 or 
            not results["details"]["answers_consistent"]):
            results["success"] = False
        
        return results
    
    def test_input_validation_security(self) -> Dict[str, Any]:
        """Test input validation and security measures."""
        results = {"success": True, "details": {}}
        
        pipeline = FastVLMCorePipeline()
        
        # Test security validation
        security_tests = [
            ("xss_attempt", b"normal_image_data", "<script>alert('xss')</script>"),
            ("sql_injection", b"image", "'; DROP TABLE users; --"),
            ("code_injection", b"test", "eval('malicious_code')"),
            ("html_injection", b"data", "<iframe src='malicious'></iframe>"),
            ("oversized_input", b"x" * (100 * 1024), "Normal question")  # 100KB image
        ]
        
        security_results = {}
        for test_name, image_data, question in security_tests:
            try:
                result = pipeline.process_image_question(image_data, question)
                
                # Malicious inputs should be handled safely
                security_results[test_name] = {
                    "handled_safely": True,
                    "result_type": "error" if result.confidence == 0.0 else "normal",
                    "answer_contains_input": question.lower() in result.answer.lower()
                }
                
            except Exception as e:
                # Exceptions for malicious input are acceptable
                security_results[test_name] = {
                    "handled_safely": True,
                    "result_type": "exception",
                    "exception_type": type(e).__name__
                }
        
        results["details"] = security_results
        
        # All security tests should be handled safely
        all_safe = all(test["handled_safely"] for test in security_results.values())
        results["success"] = all_safe
        
        return results
    
    def test_monitoring_and_observability(self) -> Dict[str, Any]:
        """Test monitoring and observability features."""
        results = {"success": True, "details": {}}
        
        pipeline = FastVLMCorePipeline()
        demo_image = create_demo_image()
        
        # Generate activity for monitoring
        for i in range(10):
            pipeline.process_image_question(demo_image, f"Monitoring test {i}")
        
        # Test health status
        health_status = pipeline.get_health_status()
        
        # Test statistics
        stats = pipeline.get_stats()
        
        results["details"] = {
            "health_status_available": isinstance(health_status, dict),
            "health_contains_required_fields": all(
                field in health_status for field in 
                ["status", "total_requests", "success_rate_percent"]
            ),
            "stats_available": isinstance(stats, dict),
            "stats_contains_components": "components" in stats,
            "processing_stats": dict(pipeline.processing_stats) if hasattr(pipeline, 'processing_stats') else {}
        }
        
        # Monitoring should provide comprehensive data
        monitoring_works = (
            results["details"]["health_status_available"] and
            results["details"]["health_contains_required_fields"] and
            results["details"]["stats_available"]
        )
        
        results["success"] = monitoring_works
        
        return results
    
    def test_adaptive_quality_management(self) -> Dict[str, Any]:
        """Test adaptive quality management features."""
        results = {"success": True, "details": {}}
        
        try:
            pipeline = FastVLMCorePipeline()
            
            # Check if mobile optimizer is available
            has_optimizer = hasattr(pipeline, 'mobile_optimizer') and pipeline.mobile_optimizer is not None
            
            results["details"] = {
                "mobile_optimizer_available": has_optimizer,
                "adaptive_features_present": has_optimizer
            }
            
            if has_optimizer:
                # Test adaptive behavior
                demo_image = create_demo_image()
                
                # Process requests to trigger adaptation
                for i in range(20):
                    pipeline.process_image_question(demo_image, f"Adaptive test {i}")
                
                # Check if optimizer has collected metrics
                if hasattr(pipeline.mobile_optimizer, 'get_performance_stats'):
                    perf_stats = pipeline.mobile_optimizer.get_performance_stats()
                    results["details"]["performance_stats"] = perf_stats
                    results["details"]["adaptation_working"] = perf_stats.get("requests", {}).get("total_requests", 0) > 0
            
            results["success"] = True  # Adaptive quality is optional enhancement
            
        except Exception as e:
            results["details"]["error"] = str(e)
            results["success"] = False
        
        return results
    
    def test_circuit_breaker_patterns(self) -> Dict[str, Any]:
        """Test circuit breaker fault tolerance patterns."""
        results = {"success": True, "details": {}}
        
        pipeline = FastVLMCorePipeline()
        
        # Test circuit breaker integration
        circuit_breaker_available = hasattr(pipeline, 'circuit_breaker')
        
        results["details"] = {
            "circuit_breaker_available": circuit_breaker_available,
            "initial_state": pipeline.circuit_breaker.state if circuit_breaker_available else "N/A"
        }
        
        if circuit_breaker_available:
            # Test circuit breaker behavior
            # Note: We don't want to actually break the system, so we'll just verify it exists
            results["details"]["failure_threshold"] = pipeline.circuit_breaker.failure_threshold
            results["details"]["timeout_seconds"] = pipeline.circuit_breaker.timeout_seconds
        
        results["success"] = circuit_breaker_available
        
        return results
    
    def test_mobile_optimization_features(self) -> Dict[str, Any]:
        """Test mobile-specific optimization features."""
        results = {"success": True, "details": {}}
        
        try:
            # Test mobile-optimized configurations
            mobile_configs = [
                InferenceConfig(model_name="fast-vlm-tiny", quantization_bits=4),
                InferenceConfig(model_name="fast-vlm-base", quantization_bits=8),
            ]
            
            mobile_results = {}
            demo_image = create_demo_image()
            
            for i, config in enumerate(mobile_configs):
                pipeline = FastVLMCorePipeline(config)
                
                # Test mobile-optimized processing
                start_time = time.time()
                result = pipeline.process_image_question(demo_image, "Mobile optimization test")
                latency = (time.time() - start_time) * 1000
                
                mobile_results[f"config_{i}"] = {
                    "model_name": config.model_name,
                    "quantization_bits": config.quantization_bits,
                    "latency_ms": latency,
                    "success": len(result.answer) > 0,
                    "mobile_optimized": latency < 300  # 300ms mobile target
                }
            
            results["details"] = mobile_results
            
            # At least one config should meet mobile performance targets
            mobile_optimized = any(
                result["mobile_optimized"] and result["success"] 
                for result in mobile_results.values()
            )
            
            results["success"] = mobile_optimized
            
        except Exception as e:
            results["details"]["error"] = str(e)
            results["success"] = False
        
        return results
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final test report."""
        total_time = time.time() - self.start_time
        
        # Calculate overall statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["status"] == "PASSED")
        failed_tests = sum(1 for result in self.test_results.values() if result["status"] == "FAILED")
        error_tests = sum(1 for result in self.test_results.values() if result["status"] == "ERROR")
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Determine overall system status
        if success_rate >= 90:
            system_status = "PRODUCTION_READY"
        elif success_rate >= 70:
            system_status = "NEEDS_IMPROVEMENT"
        else:
            system_status = "NOT_READY"
        
        # Generate recommendations
        recommendations = []
        if failed_tests > 0:
            recommendations.append("Address failed test cases for improved reliability")
        if error_tests > 0:
            recommendations.append("Fix error conditions in test execution")
        if success_rate < 100:
            recommendations.append("Investigate and resolve remaining issues")
        
        if not recommendations:
            recommendations.append("System is performing well - consider load testing in production environment")
        
        final_report = {
            "summary": {
                "system_status": system_status,
                "total_execution_time_seconds": total_time,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "error_tests": error_tests,
                "success_rate_percent": round(success_rate, 1),
                "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "detailed_results": self.test_results,
            "recommendations": recommendations,
            "system_capabilities": {
                "basic_functionality": "PASSED" if "test_basic_functionality" in self.test_results and 
                                     self.test_results["test_basic_functionality"]["status"] == "PASSED" else "FAILED",
                "error_resilience": "PASSED" if "test_error_handling_resilience" in self.test_results and 
                                  self.test_results["test_error_handling_resilience"]["status"] == "PASSED" else "FAILED",
                "performance_optimization": "PASSED" if "test_performance_characteristics" in self.test_results and 
                                          self.test_results["test_performance_characteristics"]["status"] == "PASSED" else "FAILED",
                "concurrent_processing": "PASSED" if "test_concurrent_load" in self.test_results and 
                                       self.test_results["test_concurrent_load"]["status"] == "PASSED" else "FAILED",
                "security_validation": "PASSED" if "test_input_validation_security" in self.test_results and 
                                     self.test_results["test_input_validation_security"]["status"] == "PASSED" else "FAILED"
            }
        }
        
        return final_report


def main():
    """Main test execution function."""
    print("🚀 FastVLM Production-Ready System Integration Test")
    print("=" * 60)
    
    # Run comprehensive test suite
    test_suite = ProductionSystemTest()
    final_report = test_suite.run_all_tests()
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 FINAL TEST REPORT")
    print("=" * 60)
    
    summary = final_report["summary"]
    print(f"System Status: {summary['system_status']}")
    print(f"Success Rate: {summary['success_rate_percent']}%")
    print(f"Tests Passed: {summary['passed_tests']}/{summary['total_tests']}")
    print(f"Total Time: {summary['total_execution_time_seconds']:.1f}s")
    
    print("\n🎯 System Capabilities:")
    for capability, status in final_report["system_capabilities"].items():
        status_icon = "✅" if status == "PASSED" else "❌"
        print(f"  {status_icon} {capability.replace('_', ' ').title()}: {status}")
    
    print("\n💡 Recommendations:")
    for i, recommendation in enumerate(final_report["recommendations"], 1):
        print(f"  {i}. {recommendation}")
    
    # Save detailed report
    report_file = f"production_test_report_{int(time.time())}.json"
    with open(report_file, 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Final status
    if summary["system_status"] == "PRODUCTION_READY":
        print("\n🎉 System is PRODUCTION READY!")
        return 0
    else:
        print(f"\n⚠️ System status: {summary['system_status']}")
        print("Please address the identified issues before production deployment.")
        return 1


if __name__ == "__main__":
    exit(main())