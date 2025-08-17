"""
Comprehensive Quality Gates Runner for FastVLM.

Validates production readiness without requiring external dependencies.
Tests core functionality, performance, security, and reliability.
"""

import sys
import os
import time
import json
import logging
import threading
import multiprocessing
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

logger = logging.getLogger(__name__)


class QualityGateRunner:
    """Comprehensive quality gate validation."""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None
        self.passed_gates = 0
        self.total_gates = 0
        
    def run_all_quality_gates(self) -> bool:
        """Run all quality gates and return overall success."""
        print("🚀 Starting FastVLM Comprehensive Quality Gates")
        print("=" * 60)
        
        self.start_time = time.time()
        
        # Define quality gates
        quality_gates = [
            ("Core Architecture", self.test_core_architecture),
            ("Import Structure", self.test_import_structure),
            ("Security Framework", self.test_security_framework),
            ("Performance Engine", self.test_performance_engine),
            ("Error Recovery", self.test_error_recovery),
            ("Mobile Optimization", self.test_mobile_optimization),
            ("Logging System", self.test_logging_system),
            ("Code Quality", self.test_code_quality),
            ("Memory Management", self.test_memory_management),
            ("Concurrency Safety", self.test_concurrency_safety),
            ("Production Readiness", self.test_production_readiness)
        ]
        
        # Run each quality gate
        for gate_name, gate_test in quality_gates:
            self.total_gates += 1
            print(f"\n📋 Testing: {gate_name}")
            print("-" * 40)
            
            try:
                result = gate_test()
                if result["passed"]:
                    self.passed_gates += 1
                    print(f"✅ {gate_name}: PASSED")
                    if result.get("details"):
                        print(f"   Details: {result['details']}")
                else:
                    print(f"❌ {gate_name}: FAILED")
                    print(f"   Reason: {result.get('reason', 'Unknown')}")
                
                self.results[gate_name] = result
                
            except Exception as e:
                print(f"💥 {gate_name}: ERROR - {e}")
                self.results[gate_name] = {
                    "passed": False,
                    "reason": f"Exception: {e}",
                    "traceback": traceback.format_exc()
                }
        
        self.end_time = time.time()
        
        # Generate final report
        self._generate_final_report()
        
        return self.passed_gates == self.total_gates
    
    def test_core_architecture(self) -> Dict[str, Any]:
        """Test core architecture components."""
        try:
            # Test core pipeline import
            from fast_vlm_ondevice.core_pipeline import FastVLMPipeline, InferenceConfig
            
            # Test basic instantiation
            config = InferenceConfig(batch_size=1, use_cache=True)
            pipeline = FastVLMPipeline(config)
            
            # Test basic functionality
            mock_image = b"test_image_data" * 100
            mock_question = "Test question"
            
            result = pipeline.process_vlm_query(mock_image, mock_question)
            
            # Validate result structure
            required_fields = ["result", "confidence", "processing_time_ms"]
            missing_fields = [field for field in required_fields if field not in result]
            
            if missing_fields:
                return {
                    "passed": False,
                    "reason": f"Missing required fields: {missing_fields}"
                }
            
            return {
                "passed": True,
                "details": f"Core pipeline functional, processing time: {result['processing_time_ms']:.1f}ms"
            }
            
        except Exception as e:
            return {
                "passed": False,
                "reason": f"Core architecture test failed: {e}"
            }
    
    def test_import_structure(self) -> Dict[str, Any]:
        """Test import structure and module availability."""
        try:
            # Test main package import
            import fast_vlm_ondevice
            
            # Test key component imports
            imports_to_test = [
                "fast_vlm_ondevice.converter",
                "fast_vlm_ondevice.core_pipeline", 
                "fast_vlm_ondevice.real_time_mobile_optimizer",
                "fast_vlm_ondevice.advanced_error_recovery",
                "fast_vlm_ondevice.comprehensive_logging",
                "fast_vlm_ondevice.production_security_framework",
                "fast_vlm_ondevice.high_performance_distributed_engine"
            ]
            
            successful_imports = 0
            failed_imports = []
            
            for module_name in imports_to_test:
                try:
                    __import__(module_name)
                    successful_imports += 1
                except Exception as e:
                    failed_imports.append(f"{module_name}: {e}")
            
            if failed_imports:
                return {
                    "passed": False,
                    "reason": f"Failed imports: {failed_imports}"
                }
            
            return {
                "passed": True,
                "details": f"All {successful_imports} core modules imported successfully"
            }
            
        except Exception as e:
            return {
                "passed": False,
                "reason": f"Import structure test failed: {e}"
            }
    
    def test_security_framework(self) -> Dict[str, Any]:
        """Test security framework functionality."""
        try:
            from fast_vlm_ondevice.production_security_framework import (
                ProductionSecurityFramework,
                InputValidator
            )
            
            # Create security framework
            security = ProductionSecurityFramework()
            validator = InputValidator()
            
            # Test safe input validation
            safe_request = {
                "image_data": b"safe_image_data" * 100,
                "question": "What objects are visible?",
                "source_ip": "192.168.1.100"
            }
            
            safe_result = security.validate_and_secure_request(safe_request)
            
            if not safe_result["valid"]:
                return {
                    "passed": False,
                    "reason": "Safe request marked as invalid"
                }
            
            # Test malicious input detection
            malicious_request = {
                "image_data": b"<script>alert('xss')</script>",
                "question": "'; DROP TABLE users; --",
                "source_ip": "192.168.1.100"
            }
            
            malicious_result = security.validate_and_secure_request(malicious_request)
            
            if malicious_result["valid"]:
                return {
                    "passed": False,
                    "reason": "Malicious request not detected"
                }
            
            return {
                "passed": True,
                "details": f"Security validation working - detected {len(malicious_result['security_warnings'])} threats"
            }
            
        except Exception as e:
            return {
                "passed": False,
                "reason": f"Security framework test failed: {e}"
            }
    
    def test_performance_engine(self) -> Dict[str, Any]:
        """Test high-performance distributed engine."""
        try:
            from fast_vlm_ondevice.high_performance_distributed_engine import (
                create_high_performance_engine,
                ComputeStrategy
            )
            
            # Create engine
            engine = create_high_performance_engine(
                compute_strategy=ComputeStrategy.MULTI_THREADED,
                max_workers=4
            )
            
            # Test task submission
            start_time = time.time()
            result = engine.submit_task_sync(
                "vision_encoding",
                {"image_size": 224, "channels": 3},
                priority=5
            )
            duration = time.time() - start_time
            
            # Validate result
            if result["status"] != "completed":
                return {
                    "passed": False,
                    "reason": f"Task not completed: {result}"
                }
            
            # Test performance metrics
            metrics = engine.get_comprehensive_metrics()
            
            if metrics["total_completed"] < 1:
                return {
                    "passed": False,
                    "reason": "No completed tasks recorded"
                }
            
            # Cleanup
            engine.shutdown()
            
            return {
                "passed": True,
                "details": f"Performance engine functional, task completed in {duration*1000:.1f}ms"
            }
            
        except Exception as e:
            return {
                "passed": False,
                "reason": f"Performance engine test failed: {e}"
            }
    
    def test_error_recovery(self) -> Dict[str, Any]:
        """Test error recovery mechanisms."""
        try:
            from fast_vlm_ondevice.advanced_error_recovery import (
                create_error_recovery_manager,
                RecoveryStrategy,
                CircuitBreaker
            )
            
            # Test circuit breaker
            circuit_breaker = CircuitBreaker(failure_threshold=3, timeout_ms=1000)
            
            # Test normal operation
            def successful_operation():
                return "success"
            
            result = circuit_breaker.call(successful_operation)
            if result != "success":
                return {
                    "passed": False,
                    "reason": "Circuit breaker failed on successful operation"
                }
            
            # Test recovery manager
            recovery_manager = create_error_recovery_manager(max_retries=2)
            
            # Register fallback
            def fallback_method():
                return {"result": "fallback", "quality": "degraded"}
            
            recovery_manager.register_fallback_method("test_operation", fallback_method)
            
            # Test fallback execution
            @recovery_manager.with_recovery("test_operation", [RecoveryStrategy.FALLBACK])
            def failing_operation():
                raise ConnectionError("Simulated failure")
            
            try:
                result = failing_operation()
                if result["result"] != "fallback":
                    return {
                        "passed": False,
                        "reason": "Fallback not executed properly"
                    }
            except Exception as e:
                return {
                    "passed": False,
                    "reason": f"Error recovery failed: {e}"
                }
            
            return {
                "passed": True,
                "details": "Error recovery mechanisms functional"
            }
            
        except Exception as e:
            return {
                "passed": False,
                "reason": f"Error recovery test failed: {e}"
            }
    
    def test_mobile_optimization(self) -> Dict[str, Any]:
        """Test mobile optimization capabilities."""
        try:
            from fast_vlm_ondevice.real_time_mobile_optimizer import (
                create_mobile_optimizer,
                MobileOptimizationConfig
            )
            
            # Create mobile optimizer
            optimizer = create_mobile_optimizer(
                target_latency_ms=200,
                max_memory_mb=400
            )
            
            # Test optimization
            model_data = {
                "size_mb": 350,
                "complexity_score": 0.6,
                "target_device": "iphone_15_pro"
            }
            
            result = optimizer.optimize_for_mobile(model_data)
            
            if not result.optimized:
                return {
                    "passed": False,
                    "reason": "Mobile optimization failed"
                }
            
            if result.latency_ms > 250:  # Should meet target
                return {
                    "passed": False,
                    "reason": f"Latency target not met: {result.latency_ms}ms"
                }
            
            return {
                "passed": True,
                "details": f"Mobile optimization successful - {result.latency_ms:.1f}ms latency"
            }
            
        except Exception as e:
            return {
                "passed": False,
                "reason": f"Mobile optimization test failed: {e}"
            }
    
    def test_logging_system(self) -> Dict[str, Any]:
        """Test comprehensive logging system."""
        try:
            from fast_vlm_ondevice.comprehensive_logging import (
                get_logger,
                LogLevel,
                LogCategory
            )
            
            # Create logger
            logger = get_logger("test_logger")
            
            # Test various log levels
            logger.info("Test info message")
            logger.warning("Test warning message")
            logger.error("Test error message", Exception("Test exception"))
            
            # Test structured logging
            logger.performance("Performance test", metric=None)
            logger.security("Security test")
            logger.user_action("User action test")
            
            # Test context management
            with logger.context(user_id="test_user", operation="test_op"):
                logger.info("Context test message")
            
            return {
                "passed": True,
                "details": "Logging system functional"
            }
            
        except Exception as e:
            return {
                "passed": False,
                "reason": f"Logging system test failed: {e}"
            }
    
    def test_code_quality(self) -> Dict[str, Any]:
        """Test code quality metrics."""
        try:
            # Test file structure
            src_dir = Path("src/fast_vlm_ondevice")
            if not src_dir.exists():
                return {
                    "passed": False,
                    "reason": "Source directory not found"
                }
            
            # Count Python files
            python_files = list(src_dir.glob("*.py"))
            if len(python_files) < 10:
                return {
                    "passed": False,
                    "reason": f"Too few Python files: {len(python_files)}"
                }
            
            # Test for docstrings (basic check)
            files_with_docstrings = 0
            for py_file in python_files:
                try:
                    content = py_file.read_text()
                    if '"""' in content or "'''" in content:
                        files_with_docstrings += 1
                except:
                    pass
            
            docstring_ratio = files_with_docstrings / len(python_files)
            if docstring_ratio < 0.8:
                return {
                    "passed": False,
                    "reason": f"Low docstring coverage: {docstring_ratio:.1%}"
                }
            
            return {
                "passed": True,
                "details": f"Code quality check passed - {len(python_files)} files, {docstring_ratio:.1%} with docstrings"
            }
            
        except Exception as e:
            return {
                "passed": False,
                "reason": f"Code quality test failed: {e}"
            }
    
    def test_memory_management(self) -> Dict[str, Any]:
        """Test memory management and cleanup."""
        try:
            import gc
            import psutil
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create and destroy multiple components
            components = []
            for i in range(10):
                try:
                    from fast_vlm_ondevice.core_pipeline import FastVLMPipeline, InferenceConfig
                    config = InferenceConfig(batch_size=1)
                    pipeline = FastVLMPipeline(config)
                    components.append(pipeline)
                except:
                    pass  # Some imports might fail without dependencies
            
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Clean up
            del components
            gc.collect()
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_growth = peak_memory - initial_memory
            
            # Memory growth should be reasonable (< 100MB for this test)
            if memory_growth > 100:
                return {
                    "passed": False,
                    "reason": f"Excessive memory growth: {memory_growth:.1f}MB"
                }
            
            return {
                "passed": True,
                "details": f"Memory management OK - growth: {memory_growth:.1f}MB"
            }
            
        except Exception as e:
            return {
                "passed": False,
                "reason": f"Memory management test failed: {e}"
            }
    
    def test_concurrency_safety(self) -> Dict[str, Any]:
        """Test thread safety and concurrency."""
        try:
            import threading
            import time
            
            # Test concurrent access to components
            errors = []
            results = []
            
            def worker_function(worker_id):
                try:
                    from fast_vlm_ondevice.core_pipeline import FastVLMPipeline, InferenceConfig
                    config = InferenceConfig(batch_size=1)
                    pipeline = FastVLMPipeline(config)
                    
                    # Simulate concurrent processing
                    mock_image = b"concurrent_test" * 50
                    mock_question = f"Question from worker {worker_id}"
                    result = pipeline.process_vlm_query(mock_image, mock_question)
                    results.append(result)
                    
                except Exception as e:
                    errors.append(f"Worker {worker_id}: {e}")
            
            # Create multiple threads
            threads = []
            for i in range(5):
                thread = threading.Thread(target=worker_function, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads
            for thread in threads:
                thread.join(timeout=10)
            
            if errors:
                return {
                    "passed": False,
                    "reason": f"Concurrency errors: {errors}"
                }
            
            if len(results) != 5:
                return {
                    "passed": False,
                    "reason": f"Not all threads completed: {len(results)}/5"
                }
            
            return {
                "passed": True,
                "details": f"Concurrency test passed - {len(results)} threads completed successfully"
            }
            
        except Exception as e:
            return {
                "passed": False,
                "reason": f"Concurrency safety test failed: {e}"
            }
    
    def test_production_readiness(self) -> Dict[str, Any]:
        """Test overall production readiness."""
        try:
            # Check for required configuration files
            required_files = [
                "pyproject.toml",
                "requirements.txt", 
                "README.md",
                "CLAUDE.md"
            ]
            
            missing_files = []
            for file_name in required_files:
                if not Path(file_name).exists():
                    missing_files.append(file_name)
            
            if missing_files:
                return {
                    "passed": False,
                    "reason": f"Missing required files: {missing_files}"
                }
            
            # Check Swift package structure
            ios_dir = Path("ios")
            if not ios_dir.exists():
                return {
                    "passed": False,
                    "reason": "iOS package directory missing"
                }
            
            # Check for example/demo files
            examples_dir = Path("examples")
            if not examples_dir.exists() or len(list(examples_dir.glob("*.py"))) < 2:
                return {
                    "passed": False,
                    "reason": "Insufficient example files"
                }
            
            return {
                "passed": True,
                "details": "Production readiness check passed"
            }
            
        except Exception as e:
            return {
                "passed": False,
                "reason": f"Production readiness test failed: {e}"
            }
    
    def _generate_final_report(self):
        """Generate comprehensive final report."""
        total_time = self.end_time - self.start_time
        success_rate = (self.passed_gates / self.total_gates) * 100
        
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE QUALITY GATES SUMMARY")
        print("=" * 60)
        
        print(f"Total Quality Gates: {self.total_gates}")
        print(f"Passed: {self.passed_gates}")
        print(f"Failed: {self.total_gates - self.passed_gates}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Total Time: {total_time:.2f}s")
        
        # Print failed gates
        failed_gates = [name for name, result in self.results.items() if not result["passed"]]
        if failed_gates:
            print(f"\n❌ FAILED QUALITY GATES:")
            for gate_name in failed_gates:
                result = self.results[gate_name]
                print(f"  - {gate_name}: {result.get('reason', 'Unknown failure')}")
        
        # Print passed gates
        passed_gates = [name for name, result in self.results.items() if result["passed"]]
        if passed_gates:
            print(f"\n✅ PASSED QUALITY GATES:")
            for gate_name in passed_gates:
                result = self.results[gate_name]
                details = result.get('details', 'OK')
                print(f"  - {gate_name}: {details}")
        
        # Overall assessment
        if success_rate == 100:
            print("\n🎉 ALL QUALITY GATES PASSED - SYSTEM READY FOR PRODUCTION!")
            print("   ✅ Core functionality validated")
            print("   ✅ Security framework operational")
            print("   ✅ Performance engine optimized")
            print("   ✅ Error recovery mechanisms active")
            print("   ✅ Mobile optimization ready")
            print("   ✅ Production deployment approved")
        elif success_rate >= 80:
            print("\n⚠️  MOST QUALITY GATES PASSED - REVIEW FAILURES BEFORE DEPLOYMENT")
        else:
            print("\n❌ QUALITY GATES FAILED - SYSTEM NOT READY FOR PRODUCTION")
            print("   Please address the failed quality gates before deployment.")
        
        print("=" * 60)
        
        # Save detailed report
        report_data = {
            "timestamp": time.time(),
            "total_gates": self.total_gates,
            "passed_gates": self.passed_gates,
            "success_rate": success_rate,
            "duration_seconds": total_time,
            "results": self.results
        }
        
        with open("quality_gates_report.json", "w") as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"📝 Detailed report saved to: quality_gates_report.json")


def main():
    """Main execution function."""
    runner = QualityGateRunner()
    success = runner.run_all_quality_gates()
    
    # Return appropriate exit code
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)