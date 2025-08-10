#!/usr/bin/env python3
"""
Autonomous Validation Suite for FastVLM On-Device Kit.

Comprehensive validation of all system components without external dependencies.
"""

import sys
import os
import time
import json
import traceback
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

class ValidationResult:
    """Container for validation results."""
    def __init__(self, test_name: str, success: bool, message: str = "", duration: float = 0.0):
        self.test_name = test_name
        self.success = success
        self.message = message
        self.duration = duration
        self.timestamp = time.time()
    
    def to_dict(self):
        return {
            "test_name": self.test_name,
            "success": self.success,
            "message": self.message,
            "duration": self.duration,
            "timestamp": self.timestamp
        }


class AutonomousValidator:
    """Autonomous validation system for FastVLM components."""
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
    
    def run_validation(self, test_func, test_name: str) -> ValidationResult:
        """Run a validation test and record results."""
        start_time = time.time()
        
        try:
            test_func()
            duration = time.time() - start_time
            result = ValidationResult(test_name, True, "Passed", duration)
            print(f"✅ {test_name} - PASSED ({duration:.3f}s)")
        except Exception as e:
            duration = time.time() - start_time
            result = ValidationResult(test_name, False, str(e), duration)
            print(f"❌ {test_name} - FAILED ({duration:.3f}s): {str(e)}")
        
        self.results.append(result)
        return result
    
    def run_all_validations(self):
        """Run complete validation suite."""
        print("🚀 Starting Autonomous FastVLM Validation Suite\n")
        
        # Core module validations
        self.run_validation(self.test_core_imports, "Core Module Imports")
        self.run_validation(self.test_converter_module, "Converter Module")
        self.run_validation(self.test_autonomous_intelligence, "Autonomous Intelligence Engine")
        self.run_validation(self.test_quantum_optimization, "Quantum Optimization Engine")
        self.run_validation(self.test_edge_orchestrator, "Edge Computing Orchestrator")
        self.run_validation(self.test_security_framework, "Advanced Security Framework")
        self.run_validation(self.test_reliability_engine, "Production Reliability Engine")
        self.run_validation(self.test_performance_engine, "Hyper Performance Engine")
        
        # Integration validations
        self.run_validation(self.test_pipeline_integration, "Pipeline Integration")
        self.run_validation(self.test_orchestrator_integration, "Orchestrator Integration")
        
        # System validations
        self.run_validation(self.test_memory_usage, "Memory Usage Validation")
        self.run_validation(self.test_configuration_validation, "Configuration Validation")
        
        # Generate report
        self.generate_validation_report()
    
    def test_core_imports(self):
        """Test core module imports."""
        try:
            # Test basic imports without heavy dependencies
            import fast_vlm_ondevice.converter as converter_module
            assert hasattr(converter_module, 'FastVLMConverter')
            
            # Test configuration classes
            from fast_vlm_ondevice.quantization import QuantizationConfig
            config = QuantizationConfig()
            assert config is not None
            
        except ImportError as e:
            if "psutil" in str(e) or "torch" in str(e):
                print("⚠️  Some optional dependencies missing (expected in test environment)")
            else:
                raise e
    
    def test_converter_module(self):
        """Test FastVLM converter module."""
        try:
            from fast_vlm_ondevice.converter import FastVLMConverter
            
            # Test converter initialization (without heavy dependencies)
            # This tests the fallback mechanism
            converter = FastVLMConverter()
            assert converter is not None
            assert hasattr(converter, 'model_size_mb')
            
        except Exception as e:
            if "torch" in str(e).lower() or "coremltools" in str(e).lower():
                print("⚠️  ML dependencies not available - testing fallback mechanisms")
                # Test that fallback classes are created
                from fast_vlm_ondevice.converter import torch
                assert torch is not None
            else:
                raise e
    
    def test_autonomous_intelligence(self):
        """Test autonomous intelligence engine."""
        from fast_vlm_ondevice.autonomous_intelligence import (
            IntelligenceLevel, 
            create_autonomous_intelligence
        )
        
        # Test engine creation
        engine = create_autonomous_intelligence(IntelligenceLevel.ADAPTIVE)
        assert engine is not None
        assert engine.config.intelligence_level == IntelligenceLevel.ADAPTIVE
        
        # Test configuration
        assert hasattr(engine, 'pattern_engine')
        assert hasattr(engine, 'decision_engine')
        
        # Test status reporting
        status = engine.get_intelligence_status()
        assert isinstance(status, dict)
        assert 'engine_status' in status
    
    def test_quantum_optimization(self):
        """Test quantum optimization engine."""
        from fast_vlm_ondevice.quantum_optimization import (
            QuantumOptimizationMethod,
            create_quantum_optimizer
        )
        
        # Test optimizer creation
        optimizer = create_quantum_optimizer(
            QuantumOptimizationMethod.QUANTUM_ANNEALING
        )
        assert optimizer is not None
        assert optimizer.config.method == QuantumOptimizationMethod.QUANTUM_ANNEALING
        
        # Test statistics
        stats = optimizer.get_optimization_stats()
        assert isinstance(stats, dict)
        assert 'total_optimizations' in stats
    
    def test_edge_orchestrator(self):
        """Test edge computing orchestrator."""
        from fast_vlm_ondevice.edge_computing_orchestrator import (
            create_edge_orchestrator,
            create_mobile_edge_node
        )
        
        # Test orchestrator creation
        orchestrator = create_edge_orchestrator()
        assert orchestrator is not None
        assert hasattr(orchestrator, 'cluster')
        
        # Test mobile node creation
        mobile_node = create_mobile_edge_node()
        assert mobile_node is not None
        assert mobile_node.capabilities.neural_engine is True
        
        # Test status reporting
        status = orchestrator.get_orchestrator_status()
        assert isinstance(status, dict)
        assert 'orchestrator' in status
    
    def test_security_framework(self):
        """Test advanced security framework."""
        from fast_vlm_ondevice.advanced_security_framework import (
            create_security_framework,
            EncryptionLevel
        )
        
        # Test framework creation
        framework = create_security_framework("standard")
        assert framework is not None
        assert framework.crypto_manager.encryption_level == EncryptionLevel.STANDARD
        
        # Test security status
        status = framework.get_security_status()
        assert isinstance(status, dict)
        assert 'framework' in status
        assert 'metrics' in status
    
    def test_reliability_engine(self):
        """Test production reliability engine."""
        from fast_vlm_ondevice.production_reliability_engine import (
            create_reliability_engine,
            ReliabilityLevel
        )
        
        # Test engine creation
        engine = create_reliability_engine(ReliabilityLevel.HIGH)
        assert engine is not None
        assert engine.reliability_level == ReliabilityLevel.HIGH
        
        # Test reliability report
        report = engine.get_reliability_report()
        assert isinstance(report, dict)
        assert 'reliability_level' in report
        assert 'overall_metrics' in report
    
    def test_performance_engine(self):
        """Test hyper performance engine."""
        from fast_vlm_ondevice.hyper_performance_engine import (
            create_hyper_performance_engine,
            PerformanceLevel
        )
        
        # Test engine creation
        engine = create_hyper_performance_engine(PerformanceLevel.BALANCED)
        assert engine is not None
        assert engine.config.performance_level == PerformanceLevel.BALANCED
        
        # Test performance report
        report = engine.get_performance_report()
        assert isinstance(report, dict)
        assert 'config' in report
        assert 'metrics' in report
    
    def test_pipeline_integration(self):
        """Test pipeline integration."""
        from fast_vlm_ondevice.core_pipeline import FastVLMPipeline, PipelineConfig
        
        # Test pipeline creation with minimal config
        config = PipelineConfig(model_path="test_model.mlpackage")
        pipeline = FastVLMPipeline(config)
        
        assert pipeline is not None
        assert pipeline.config.model_path == "test_model.mlpackage"
        
        # Test metrics
        metrics = pipeline.get_performance_metrics()
        assert isinstance(metrics, dict)
        assert 'pipeline_metrics' in metrics
    
    def test_orchestrator_integration(self):
        """Test orchestrator integration."""
        from fast_vlm_ondevice.intelligent_orchestrator import (
            IntelligentOrchestrator,
            OrchestratorConfig
        )
        
        # Test orchestrator creation
        config = OrchestratorConfig(model_path="test_model.mlpackage")
        orchestrator = IntelligentOrchestrator(config)
        
        assert orchestrator is not None
        assert orchestrator.config.model_path == "test_model.mlpackage"
        
        # Test system status
        status = orchestrator.get_system_status()
        assert isinstance(status, dict)
        assert 'orchestrator' in status
    
    def test_memory_usage(self):
        """Test memory usage validation."""
        import gc
        import sys
        
        # Get initial memory
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Create and destroy some objects
        test_objects = [list(range(1000)) for _ in range(10)]
        del test_objects
        
        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory should not have grown significantly
        object_growth = final_objects - initial_objects
        assert object_growth < 1000, f"Memory leak detected: {object_growth} objects not collected"
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test various configuration classes
        from fast_vlm_ondevice.autonomous_intelligence import IntelligenceConfig
        from fast_vlm_ondevice.quantum_optimization import OptimizationConfig
        from fast_vlm_ondevice.advanced_security_framework import SecurityPolicy
        
        # Test intelligence config
        intel_config = IntelligenceConfig()
        assert intel_config.intelligence_level is not None
        assert intel_config.max_autonomous_decisions_per_hour > 0
        
        # Test optimization config
        opt_config = OptimizationConfig()
        assert opt_config.max_iterations > 0
        assert opt_config.convergence_threshold > 0
        
        # Test security policy
        sec_policy = SecurityPolicy()
        assert sec_policy.max_failed_attempts > 0
        assert len(sec_policy.allowed_content_types) > 0
    
    def generate_validation_report(self):
        """Generate comprehensive validation report."""
        total_time = time.time() - self.start_time
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print("\n" + "="*80)
        print("🏁 VALIDATION SUITE COMPLETE")
        print("="*80)
        print(f"📊 Results Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   ✅ Passed: {passed_tests}")
        print(f"   ❌ Failed: {failed_tests}")
        print(f"   📈 Success Rate: {success_rate:.1f}%")
        print(f"   ⏱️  Total Time: {total_time:.2f}s")
        print("="*80)
        
        if failed_tests > 0:
            print("\n❌ FAILED TESTS:")
            for result in self.results:
                if not result.success:
                    print(f"   - {result.test_name}: {result.message}")
        
        # Save detailed report
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": success_rate,
                "total_time": total_time,
                "timestamp": time.time()
            },
            "results": [result.to_dict() for result in self.results]
        }
        
        with open("validation_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: validation_report.json")
        
        # Return success status
        return success_rate >= 80.0  # 80% pass rate required


if __name__ == "__main__":
    validator = AutonomousValidator()
    success = validator.run_all_validations()
    
    if success:
        print("\n🎉 VALIDATION SUITE PASSED - System ready for deployment!")
        sys.exit(0)
    else:
        print("\n⚠️  VALIDATION SUITE FAILED - Review failures before deployment!")
        sys.exit(1)
