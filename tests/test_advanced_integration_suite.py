"""
Advanced Integration Test Suite
Comprehensive testing framework for FastVLM production systems.

This test suite validates:
- End-to-End Pipeline Integration
- Advanced Security Framework
- Production Reliability Patterns
- Quantum-Inspired Scaling Systems
- Performance Benchmarking
- Memory Management
- Error Recovery Mechanisms
- Real-world Scenario Testing
"""

import pytest
import asyncio
import time
import json
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, AsyncMock
import logging
import threading

# Test imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from fast_vlm_ondevice import (
        FastVLMCorePipeline, InferenceConfig, InferenceResult,
        create_demo_image
    )
    from fast_vlm_ondevice.production_reliability_framework import (
        ProductionReliabilityFramework, IntelligentCircuitBreaker,
        BulkheadIsolation, IntelligentRetryPolicy, HealthChecker,
        SelfHealingManager, ComponentState
    )
    from fast_vlm_ondevice.quantum_scale_orchestrator import (
        QuantumScaleOrchestrator, ScalingStrategy, QuantumState
    )
    from fast_vlm_ondevice.next_gen_mobile_optimizer import (
        NextGenMobileOptimizer, MobileDeviceProfile, AdaptiveQualitySettings
    )
except ImportError as e:
    pytest.skip(f"Advanced modules not available: {e}", allow_module_level=True)

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestAdvancedIntegration:
    """Advanced integration tests for production systems."""
    
    @pytest.fixture
    def demo_image(self):
        """Create demo image for testing."""
        return create_demo_image()
    
    @pytest.fixture
    def test_pipeline(self):
        """Create test pipeline with production configuration."""
        config = InferenceConfig(
            model_name="fast-vlm-test",
            enable_caching=True,
            timeout_seconds=30.0
        )
        return FastVLMCorePipeline(config)
    
    @pytest.fixture
    def reliability_framework(self):
        """Create production reliability framework."""
        framework = ProductionReliabilityFramework()
        yield framework
        framework.shutdown()
    
    @pytest.fixture
    def quantum_orchestrator(self):
        """Create quantum scale orchestrator."""
        from fast_vlm_ondevice.quantum_scale_orchestrator import create_quantum_scale_orchestrator
        orchestrator = create_quantum_scale_orchestrator(
            strategy=ScalingStrategy.HYBRID_CLASSICAL_QUANTUM,
            quantum_qubits=8,  # Smaller for testing
            swarm_size=10
        )
        yield orchestrator
        orchestrator.shutdown()
    
    @pytest.fixture
    def mobile_optimizer(self):
        """Create mobile optimizer."""
        from fast_vlm_ondevice.next_gen_mobile_optimizer import create_mobile_optimizer
        optimizer = create_mobile_optimizer(max_memory_mb=256)
        yield optimizer
        optimizer.shutdown()

class TestProductionPipeline:
    """Test production-grade pipeline functionality."""
    
    def test_enhanced_vision_encoding(self, demo_image):
        """Test enhanced vision encoder with multi-scale features."""
        from fast_vlm_ondevice.core_pipeline import EnhancedVisionEncoder
        
        encoder = EnhancedVisionEncoder("base")
        result = encoder.encode_image(demo_image)
        
        # Validate enhanced features
        assert "features" in result
        assert "spatial_features" in result
        assert "multi_scale_features" in result
        assert "object_proposals" in result
        assert "self_attention" in result
        assert "cross_modal_features" in result
        
        # Check multi-scale processing
        assert len(result["multi_scale_features"]) == 3  # Three patch sizes
        assert "patch_16" in result["multi_scale_features"]
        assert "patch_32" in result["multi_scale_features"]
        assert "patch_64" in result["multi_scale_features"]
        
        # Validate spatial hierarchy
        assert len(result["spatial_features"]) == 3  # Three hierarchy levels
        
        # Check object detection
        assert isinstance(result["object_proposals"], list)
        for proposal in result["object_proposals"]:
            assert "bbox" in proposal
            assert "confidence" in proposal
            assert len(proposal["bbox"]) == 4  # [x, y, width, height]
            assert 0 <= proposal["confidence"] <= 1
        
        # Validate self-attention
        attention_data = result["self_attention"]
        assert "weights" in attention_data
        assert "entropy" in attention_data
        assert attention_data["num_heads"] == 8
        
        # Cross-modal readiness
        cross_modal = result["cross_modal_features"]
        assert "features" in cross_modal
        assert "readiness_score" in cross_modal
        assert 0 <= cross_modal["readiness_score"] <= 1
    
    def test_production_inference_flow(self, test_pipeline, demo_image):
        """Test complete production inference flow."""
        questions = [
            "What objects are visible in this image?",
            "Describe the spatial relationships between elements.",
            "What is the overall scene composition?",
            "How would you categorize the visual complexity?"
        ]
        
        results = []
        for question in questions:
            result = test_pipeline.process_image_question(demo_image, question)
            
            # Validate result structure
            assert isinstance(result, InferenceResult)
            assert result.answer
            assert 0 <= result.confidence <= 1
            assert result.latency_ms > 0
            assert result.model_used == "fast-vlm-test"
            assert result.metadata
            
            # Validate metadata completeness
            metadata = result.metadata
            assert "vision_features_dim" in metadata
            assert "text_tokens" in metadata
            assert "fusion_dim" in metadata
            assert "processing_time_breakdown" in metadata
            assert "request_id" in metadata
            
            # Check processing breakdown
            breakdown = metadata["processing_time_breakdown"]
            total_time = breakdown["total_ms"]
            component_times = [
                breakdown["validation_ms"],
                breakdown["vision_encoding_ms"],
                breakdown["text_encoding_ms"],
                breakdown["fusion_ms"],
                breakdown["generation_ms"]
            ]
            
            # Components should sum roughly to total (allowing for small rounding differences)
            assert abs(sum(component_times) - total_time) < total_time * 0.1
            
            results.append(result)
        
        # Validate caching works
        cache_stats = test_pipeline.get_stats()
        assert "cache_entries" in cache_stats
        
        # Test cache hit by repeating a question
        repeat_result = test_pipeline.process_image_question(demo_image, questions[0])
        assert repeat_result.metadata["cache_used"] == True
    
    def test_error_handling_robustness(self, test_pipeline):
        """Test robust error handling across scenarios."""
        # Test with invalid image data
        with pytest.raises((ValueError, RuntimeError)):
            test_pipeline.process_image_question(b"invalid", "Test question")
        
        # Test with empty image
        with pytest.raises((ValueError, RuntimeError)):
            test_pipeline.process_image_question(b"", "Test question")
        
        # Test with invalid question
        with pytest.raises((ValueError, RuntimeError)):
            test_pipeline.process_image_question(create_demo_image(), "")
        
        # Test with malicious content
        malicious_patterns = [
            "<script>alert('xss')</script>",
            "javascript:void(0)",
            "data:text/html,<script>",
            "eval(malicious_code)"
        ]
        
        demo_image = create_demo_image()
        for malicious in malicious_patterns:
            with pytest.raises((ValueError, RuntimeError)):
                test_pipeline.process_image_question(demo_image, malicious)
    
    def test_performance_benchmarking(self, test_pipeline, demo_image):
        """Test performance meets production requirements."""
        # Warm-up runs
        for _ in range(3):
            test_pipeline.process_image_question(demo_image, "Warm-up question")
        
        # Benchmark runs
        benchmark_results = []
        for i in range(10):
            start_time = time.time()
            result = test_pipeline.process_image_question(
                demo_image, 
                f"Performance benchmark question {i}"
            )
            end_time = time.time()
            
            benchmark_results.append({
                "latency_ms": result.latency_ms,
                "confidence": result.confidence,
                "wall_clock_ms": (end_time - start_time) * 1000
            })
        
        # Analyze performance
        latencies = [r["latency_ms"] for r in benchmark_results]
        confidences = [r["confidence"] for r in benchmark_results]
        
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        avg_confidence = np.mean(confidences)
        
        # Performance assertions (adjust thresholds based on requirements)
        assert avg_latency < 1000, f"Average latency {avg_latency}ms exceeds 1000ms threshold"
        assert p95_latency < 2000, f"P95 latency {p95_latency}ms exceeds 2000ms threshold"
        assert avg_confidence > 0.5, f"Average confidence {avg_confidence} below 0.5 threshold"
        
        # Consistency check
        latency_variance = np.var(latencies)
        assert latency_variance < (avg_latency ** 2), "Latency variance too high - inconsistent performance"

class TestReliabilityFramework:
    """Test production reliability framework."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, reliability_framework):
        """Test intelligent circuit breaker patterns."""
        circuit_breaker = reliability_framework.create_circuit_breaker(
            "test_service",
            failure_threshold=3,
            recovery_timeout=1  # Short timeout for testing
        )
        
        # Mock service that fails initially
        failure_count = [0]
        async def failing_service():
            failure_count[0] += 1
            if failure_count[0] <= 3:
                raise Exception("Service temporarily failing")
            return {"status": "success", "attempt": failure_count[0]}
        
        # Test failure detection
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_service)
        
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_service)
        
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_service)
        
        # Circuit should be open now
        from fast_vlm_ondevice.production_reliability_framework import CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            await circuit_breaker.call(failing_service)
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # Should succeed after recovery
        result = await circuit_breaker.call(failing_service)
        assert result["status"] == "success"
        
        # Validate metrics
        metrics = circuit_breaker.get_metrics()
        assert metrics["total_calls"] > 0
        assert metrics["failed_calls"] >= 3
        assert metrics["state"] in ["closed", "half_open"]
    
    @pytest.mark.asyncio
    async def test_bulkhead_isolation(self, reliability_framework):
        """Test bulkhead isolation patterns."""
        bulkhead = reliability_framework.create_bulkhead(
            "test_bulkhead",
            max_concurrent_calls=2
        )
        
        # Mock service with controlled execution time
        async def slow_service(delay: float = 0.5):
            await asyncio.sleep(delay)
            return {"processed": True, "delay": delay}
        
        # Start concurrent tasks
        tasks = []
        for i in range(5):  # More than bulkhead capacity
            task = asyncio.create_task(bulkhead.execute(slow_service, 0.2))
            tasks.append(task)
        
        # Some should succeed, some should be rejected
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successes = [r for r in results if isinstance(r, dict)]
        failures = [r for r in results if isinstance(r, Exception)]
        
        assert len(successes) >= 2  # At least max_concurrent_calls should succeed
        assert len(failures) >= 1   # Some should be rejected
        
        # Validate metrics
        metrics = bulkhead.get_metrics()
        assert metrics["total_calls"] == 5
        assert metrics["rejected_calls"] > 0
        assert metrics["rejection_rate_percent"] > 0
    
    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff(self):
        """Test intelligent retry policies."""
        retry_policy = IntelligentRetryPolicy(
            max_attempts=4,
            base_delay=0.1,
            max_delay=2.0,
            jitter=True
        )
        
        # Service that fails first few attempts
        attempt_count = [0]
        async def flaky_service():
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise Exception(f"Attempt {attempt_count[0]} failed")
            return {"success": True, "attempts": attempt_count[0]}
        
        # Should succeed after retries
        start_time = time.time()
        result = await retry_policy.execute(flaky_service)
        end_time = time.time()
        
        assert result["success"] == True
        assert result["attempts"] == 3
        
        # Should have taken time due to backoff delays
        total_time = end_time - start_time
        expected_min_time = 0.1 + 0.2  # First two delays (base_delay * exponential_base^attempt)
        assert total_time >= expected_min_time * 0.5  # Allow some variance
        
        # Test metrics
        metrics = retry_policy.get_metrics()
        assert metrics["total_executions"] == 1
        assert metrics["avg_attempts_per_success"] == 3
    
    @pytest.mark.asyncio
    async def test_health_checking_system(self, reliability_framework):
        """Test comprehensive health checking."""
        health_checker = reliability_framework.health_checker
        
        # Register health checks
        def healthy_component():
            return {"status": ComponentState.HEALTHY, "metadata": {"uptime": 100}}
        
        def degraded_component():
            return {"status": ComponentState.DEGRADED, "metadata": {"warning": "high_latency"}}
        
        def failing_component():
            raise Exception("Component failed")
        
        health_checker.register_health_check("service_a", healthy_component)
        health_checker.register_health_check("service_b", degraded_component)
        health_checker.register_health_check("service_c", failing_component)
        
        # Check individual components
        result_a = await health_checker.check_component_health("service_a")
        assert result_a.status == ComponentState.HEALTHY
        assert result_a.error_message is None
        
        result_b = await health_checker.check_component_health("service_b")
        assert result_b.status == ComponentState.DEGRADED
        
        result_c = await health_checker.check_component_health("service_c")
        assert result_c.status == ComponentState.FAILED
        assert result_c.error_message is not None
        
        # Check overall health
        overall_health = health_checker.get_overall_health()
        assert overall_health["total_components"] == 3
        assert overall_health["healthy_components"] == 1
        assert "service_b" in overall_health["degraded_components"]
        assert "service_c" in overall_health["failed_components"]
        assert overall_health["overall_status"] in ["degraded", "failed"]
    
    @pytest.mark.asyncio
    async def test_self_healing_mechanisms(self, reliability_framework):
        """Test self-healing and recovery."""
        self_healing = reliability_framework.self_healing
        
        # Mock recovery strategy
        recovery_attempts = [0]
        async def mock_recovery_strategy(error_info):
            recovery_attempts[0] += 1
            if recovery_attempts[0] >= 2:  # Succeed on second attempt
                return True
            return False
        
        self_healing.register_recovery_strategy("test_component", mock_recovery_strategy)
        
        # Test recovery
        error_info = {"error": "Connection timeout", "error_type": "TimeoutError"}
        
        # First attempt should fail
        success = await self_healing.attempt_healing("test_component", error_info)
        assert success == False
        
        # Second attempt should succeed
        success = await self_healing.attempt_healing("test_component", error_info)
        assert success == True
        
        # Validate metrics
        metrics = self_healing.get_healing_metrics()
        assert metrics["total_attempts"] == 2
        assert metrics["successful_attempts"] == 1
        assert metrics["success_rate_percent"] == 50.0
    
    @pytest.mark.asyncio
    async def test_integrated_reliability_patterns(self, reliability_framework):
        """Test integrated reliability patterns working together."""
        # Create components
        circuit_breaker = reliability_framework.create_circuit_breaker("integrated_test")
        bulkhead = reliability_framework.create_bulkhead("integrated_test", max_concurrent_calls=3)
        retry_policy = reliability_framework.create_retry_policy(max_attempts=2)
        
        # Mock service with various failure modes
        call_count = [0]
        async def integrated_service():
            call_count[0] += 1
            # Fail every 3rd call to test circuit breaker
            if call_count[0] % 7 == 0:
                raise Exception("Periodic failure")
            return {"call_number": call_count[0], "timestamp": time.time()}
        
        # Execute with all reliability patterns
        results = []
        exceptions = []
        
        for i in range(20):
            try:
                result = await reliability_framework.execute_with_reliability(
                    integrated_service,
                    circuit_breaker_name="integrated_test",
                    bulkhead_name="integrated_test",
                    retry_policy=retry_policy,
                    enable_self_healing=False  # Disable for this test
                )
                results.append(result)
            except Exception as e:
                exceptions.append(e)
            
            # Small delay between calls
            await asyncio.sleep(0.01)
        
        # Should have mostly successful results with some failures handled
        assert len(results) > len(exceptions), "Should have more successes than failures"
        assert len(results) > 10, "Should have significant number of successful calls"
        
        # Generate comprehensive report
        report = reliability_framework.get_comprehensive_report()
        
        # Validate report structure
        assert "framework_uptime_seconds" in report
        assert "overall_health" in report
        assert "circuit_breakers" in report
        assert "bulkheads" in report
        assert "reliability_score" in report
        assert 0 <= report["reliability_score"] <= 1

class TestQuantumScaling:
    """Test quantum-inspired scaling systems."""
    
    @pytest.mark.asyncio
    async def test_quantum_resource_management(self, quantum_orchestrator):
        """Test quantum resource registration and entanglement."""
        # Register quantum resources
        resource_ids = []
        for i in range(4):
            resource_id = f"test_resource_{i}"
            quantum_resource = quantum_orchestrator.register_quantum_resource(
                resource_id,
                coherence_time=50.0,
                fidelity=0.9 + i * 0.02
            )
            resource_ids.append(resource_id)
            
            assert quantum_resource.resource_id == resource_id
            assert quantum_resource.quantum_state == QuantumState.SUPERPOSITION
            assert quantum_resource.fidelity >= 0.9
        
        # Test entanglement
        entanglement_success = quantum_orchestrator.entangle_resources(
            resource_ids[0], resource_ids[1], 0.85
        )
        assert entanglement_success == True
        
        # Verify entanglement
        resource_0 = quantum_orchestrator.quantum_resources[resource_ids[0]]
        resource_1 = quantum_orchestrator.quantum_resources[resource_ids[1]]
        
        assert resource_ids[1] in resource_0.entanglement_partners
        assert resource_ids[0] in resource_1.entanglement_partners
        assert resource_0.quantum_state == QuantumState.ENTANGLED
        assert resource_1.quantum_state == QuantumState.ENTANGLED
        
        # Test quantum coherence measurement
        coherence = quantum_orchestrator._measure_quantum_coherence()
        assert 0 <= coherence <= 1
    
    @pytest.mark.asyncio
    async def test_quantum_annealing_optimization(self, quantum_orchestrator):
        """Test quantum annealing for resource allocation."""
        # Prepare test resources
        resources = [
            {"id": "cpu_cluster", "cost": 5.0, "throughput": 1000, "latency": 20},
            {"id": "gpu_cluster", "cost": 8.0, "throughput": 2000, "latency": 15},
            {"id": "memory_pool", "cost": 3.0, "throughput": 500, "latency": 5},
            {"id": "storage_tier", "cost": 2.0, "throughput": 300, "latency": 50}
        ]
        
        # Test different optimization objectives
        scaling_requests = [
            {
                "resources": resources,
                "objective": "minimize_cost",
                "constraints": {"max_budget": 15.0},
                "urgency": 0.5
            },
            {
                "resources": resources,
                "objective": "maximize_throughput",
                "constraints": {"max_budget": 20.0},
                "urgency": 0.8
            }
        ]
        
        for request in scaling_requests:
            result = await quantum_orchestrator.quantum_scale_operation(request)
            
            # Validate result structure
            assert "allocation" in result
            assert "cost" in result
            assert "optimization_method" in result
            
            # Validate allocation
            allocation = result["allocation"]
            assert isinstance(allocation, dict)
            assert len(allocation) <= len(resources)
            
            # All allocations should be between 0 and 1
            for resource_id, alloc_value in allocation.items():
                assert 0 <= alloc_value <= 1
            
            # Check constraint satisfaction
            if "max_budget" in request["constraints"]:
                total_cost = sum(
                    allocation.get(r["id"], 0) * r["cost"] 
                    for r in resources
                )
                # Allow some tolerance for optimization algorithms
                max_budget = request["constraints"]["max_budget"]
                assert total_cost <= max_budget * 1.1  # 10% tolerance
    
    @pytest.mark.asyncio 
    async def test_neuromorphic_swarm_optimization(self, quantum_orchestrator):
        """Test neuromorphic swarm optimization."""
        # This test focuses on the swarm optimizer functionality
        if quantum_orchestrator.swarm_optimizer:
            swarm = quantum_orchestrator.swarm_optimizer
            
            # Define simple test objective function
            def test_objective(position):
                # Simple quadratic function with minimum at origin
                return np.sum(position ** 2)
            
            # Run optimization
            result = swarm.optimize(test_objective, max_iterations=10)  # Short run for testing
            
            # Validate optimization result
            assert "best_position" in result
            assert "best_fitness" in result
            assert "optimization_history" in result
            
            # Should find solution close to origin (global minimum)
            best_position = result["best_position"]
            best_fitness = result["best_fitness"]
            
            # Fitness should be relatively low (close to 0)
            assert best_fitness < 10.0, f"Best fitness {best_fitness} too high for simple quadratic"
            
            # Position should be reasonably close to origin
            position_magnitude = np.linalg.norm(best_position)
            assert position_magnitude < 5.0, f"Best position magnitude {position_magnitude} too far from origin"
            
            # Get swarm metrics
            metrics = swarm.get_swarm_metrics()
            assert "swarm_size" in metrics
            assert "global_best_fitness" in metrics
            assert "swarm_diversity" in metrics
            assert metrics["swarm_size"] > 0
    
    @pytest.mark.asyncio
    async def test_hybrid_classical_quantum_scaling(self, quantum_orchestrator):
        """Test hybrid classical-quantum scaling strategy."""
        # This test requires the orchestrator to have hybrid strategy
        if quantum_orchestrator.strategy != ScalingStrategy.HYBRID_CLASSICAL_QUANTUM:
            pytest.skip("Test requires hybrid classical-quantum strategy")
        
        resources = [
            {"id": "hybrid_cpu", "cost": 4.0, "throughput": 800, "latency": 25},
            {"id": "hybrid_gpu", "cost": 7.0, "throughput": 1500, "latency": 18},
            {"id": "hybrid_mem", "cost": 2.5, "throughput": 400, "latency": 8}
        ]
        
        scaling_request = {
            "resources": resources,
            "objective": "minimize_cost",
            "constraints": {"max_budget": 12.0},
            "urgency": 0.7
        }
        
        result = await quantum_orchestrator.quantum_scale_operation(scaling_request)
        
        # Hybrid method should be indicated
        assert result["optimization_method"] == "hybrid_classical_quantum"
        
        # Should have both quantum and swarm metrics
        if "quantum_metrics" in result:
            quantum_metrics = result["quantum_metrics"]
            assert "quantum_coherence" in quantum_metrics
            assert "total_optimizations" in quantum_metrics
        
        if "swarm_metrics" in result:
            swarm_metrics = result["swarm_metrics"]
            assert "swarm_diversity" in swarm_metrics
            assert "average_fitness" in swarm_metrics
        
        # Should have fusion coherence
        if "fusion_coherence" in result:
            assert 0 <= result["fusion_coherence"] <= 1
    
    @pytest.mark.asyncio
    async def test_quantum_error_correction(self, quantum_orchestrator):
        """Test quantum error correction mechanisms."""
        # Register resources with varying fidelities
        for i in range(3):
            resource_id = f"error_test_resource_{i}"
            fidelity = 0.7 + i * 0.1  # Low to high fidelity
            quantum_orchestrator.register_quantum_resource(
                resource_id,
                fidelity=fidelity,
                coherence_time=30.0
            )
        
        # Create test allocation result
        test_result = {
            "allocation": {
                "error_test_resource_0": 0.8,  # Low fidelity resource
                "error_test_resource_1": 0.6,  # Medium fidelity resource
                "error_test_resource_2": 0.9   # High fidelity resource
            },
            "cost": 10.0
        }
        
        # Apply error correction
        corrected_result = quantum_orchestrator._apply_quantum_error_correction(test_result)
        
        assert "error_correction_applied" in corrected_result
        assert corrected_result["error_correction_applied"] == True
        
        # Low fidelity resource should have reduced allocation
        original_alloc = test_result["allocation"]["error_test_resource_0"]
        corrected_alloc = corrected_result["allocation"]["error_test_resource_0"]
        assert corrected_alloc <= original_alloc
        
        # High fidelity resource should be less affected
        high_fidelity_original = test_result["allocation"]["error_test_resource_2"]
        high_fidelity_corrected = corrected_result["allocation"]["error_test_resource_2"]
        reduction_ratio = high_fidelity_corrected / high_fidelity_original
        assert reduction_ratio > 0.9  # Minimal correction for high fidelity

class TestMobileOptimization:
    """Test next-generation mobile optimization."""
    
    def test_mobile_device_profiling(self, mobile_optimizer):
        """Test mobile device profiling and characterization."""
        device_profile = mobile_optimizer.device_profile
        
        # Validate device profile
        assert device_profile.device_id
        assert device_profile.cpu_cores > 0
        assert device_profile.memory_total_mb > 0
        assert device_profile.battery_level_percent >= 0
        assert device_profile.thermal_state in ["nominal", "fair", "serious", "critical"]
        
        # Test profile updates
        original_battery = device_profile.battery_level_percent
        # Trigger system state gathering (which updates battery)
        system_state = mobile_optimizer._gather_system_state()
        
        assert "battery_level" in system_state
        assert "thermal_state" in system_state
        assert "memory_available_mb" in system_state
    
    def test_intelligent_memory_management(self, mobile_optimizer):
        """Test AI-driven memory management."""
        memory_manager = mobile_optimizer.memory_manager
        
        # Test memory allocation
        allocation_success = memory_manager.allocate_memory("test_model", 100.0)
        assert allocation_success == True
        
        # Test memory stats
        stats = memory_manager.get_memory_stats()
        assert stats["total_allocated_mb"] >= 100.0
        assert stats["utilization_percent"] > 0
        assert stats["num_allocations"] >= 1
        
        # Test memory pressure handling
        # Allocate until near capacity
        for i in range(5):
            memory_manager.allocate_memory(f"pressure_test_{i}", 50.0)
        
        updated_stats = memory_manager.get_memory_stats()
        assert updated_stats["utilization_percent"] > stats["utilization_percent"]
        
        # Test intelligent cleanup
        freed_memory = memory_manager._intelligent_cleanup(100.0)
        assert freed_memory >= 0
        
        # Test memory prediction
        context = {
            "image_size_mb": 2.0,
            "model_complexity": 0.7,
            "batch_size": 1
        }
        predicted_usage = memory_manager.predict_memory_usage(context)
        assert predicted_usage >= 0
    
    def test_thermal_management(self, mobile_optimizer):
        """Test thermal management and throttling."""
        thermal_manager = mobile_optimizer.thermal_manager
        
        # Test thermal state updates
        initial_state = thermal_manager.current_state
        
        # Simulate high temperature
        thermal_state = thermal_manager.update_thermal_state(80.0)  # High temp
        assert thermal_state in ["fair", "serious", "critical"]
        
        # Test performance multiplier
        multiplier = thermal_manager.get_performance_multiplier()
        assert 0 < multiplier <= 1.0
        
        # Higher temperatures should result in lower multipliers
        critical_state = thermal_manager.update_thermal_state(90.0)
        critical_multiplier = thermal_manager.get_performance_multiplier()
        assert critical_multiplier <= multiplier
        
        # Test thermal prediction
        prediction = thermal_manager.predict_thermal_impact(0.8)  # High workload
        assert "current_temp_c" in prediction
        assert "predicted_temp_c" in prediction
        assert "time_to_critical_min" in prediction
        assert prediction["predicted_temp_c"] >= prediction["current_temp_c"]
    
    def test_battery_optimization(self, mobile_optimizer):
        """Test battery-aware optimization."""
        battery_optimizer = mobile_optimizer.battery_optimizer
        
        # Test battery level detection
        battery_level = battery_optimizer.get_battery_level()
        assert 0 <= battery_level <= 100
        
        # Test optimization for different battery levels
        critical_settings = battery_optimizer.optimize_for_battery(5)  # Critical battery
        low_settings = battery_optimizer.optimize_for_battery(15)      # Low battery
        normal_settings = battery_optimizer.optimize_for_battery(80)   # Normal battery
        
        # Critical battery should have most aggressive optimizations
        assert critical_settings.quantization_bits <= low_settings.quantization_bits
        assert critical_settings.attention_heads <= low_settings.attention_heads
        assert critical_settings.image_resolution[0] <= low_settings.image_resolution[0]
        
        # Low battery should be more aggressive than normal
        assert low_settings.quantization_bits <= normal_settings.quantization_bits
        assert low_settings.attention_heads <= normal_settings.attention_heads
        
        # Test power consumption prediction
        workload = {
            "cpu_intensity": 0.7,
            "gpu_usage": 0.5,
            "memory_usage_mb": 300,
            "estimated_duration_s": 15
        }
        
        prediction = battery_optimizer.predict_battery_consumption(workload)
        assert "estimated_power_mw" in prediction
        assert "estimated_energy_mwh" in prediction
        assert "battery_impact_percent" in prediction
        assert prediction["estimated_power_mw"] > 0
        assert prediction["estimated_energy_mwh"] > 0
    
    def test_adaptive_quality_optimization(self, mobile_optimizer):
        """Test multi-objective adaptive quality optimization."""
        # Test optimization for different scenarios
        scenarios = [
            {
                "name": "high_performance",
                "image_size_mb": 5.0,
                "model_complexity": 0.9,
                "question": "Detailed analysis question requiring high accuracy"
            },
            {
                "name": "power_saving",
                "image_size_mb": 1.0,
                "model_complexity": 0.3,
                "question": "Simple question"
            },
            {
                "name": "balanced",
                "image_size_mb": 2.5,
                "model_complexity": 0.6,
                "question": "Moderate complexity question"
            }
        ]
        
        optimization_results = []
        for scenario in scenarios:
            settings, record = mobile_optimizer.optimize_for_request(scenario)
            
            # Validate settings
            assert isinstance(settings, AdaptiveQualitySettings)
            assert settings.image_resolution[0] > 0
            assert settings.image_resolution[1] > 0
            assert settings.quantization_bits in [2, 3, 4, 8]
            assert settings.attention_heads > 0
            assert 0 <= settings.layer_pruning_ratio <= 0.5
            
            # Validate optimization record
            assert "timestamp" in record
            assert "optimization_time_ms" in record
            assert "predicted_resources" in record
            assert record["optimization_time_ms"] > 0
            
            optimization_results.append({
                "scenario": scenario["name"],
                "settings": settings,
                "record": record
            })
        
        # Power saving should have more aggressive optimizations than high performance
        power_settings = next(r["settings"] for r in optimization_results if r["scenario"] == "power_saving")
        performance_settings = next(r["settings"] for r in optimization_results if r["scenario"] == "high_performance")
        
        # Power saving should use smaller image resolution
        assert power_settings.image_resolution[0] <= performance_settings.image_resolution[0]
        
        # Power saving might use more aggressive quantization
        assert power_settings.quantization_bits <= performance_settings.quantization_bits
    
    def test_mobile_performance_benchmarking(self, mobile_optimizer):
        """Test mobile performance benchmarking system."""
        from fast_vlm_ondevice.next_gen_mobile_optimizer import benchmark_mobile_performance
        
        # Create test workloads
        test_workloads = [
            {
                "name": "Lightweight Processing",
                "image_size_mb": 1.0,
                "model_complexity": 0.3,
                "question": "What color is dominant in this image?"
            },
            {
                "name": "Standard Processing",
                "image_size_mb": 2.5,
                "model_complexity": 0.6,
                "question": "Describe the objects and their relationships in this scene."
            },
            {
                "name": "Heavy Processing",
                "image_size_mb": 4.0,
                "model_complexity": 0.9,
                "question": "Provide a detailed spatial analysis of the visual hierarchy and composition principles demonstrated in this complex image."
            }
        ]
        
        # Run benchmarks
        benchmarks = benchmark_mobile_performance(mobile_optimizer, test_workloads)
        
        assert len(benchmarks) == len(test_workloads)
        
        for benchmark in benchmarks:
            # Validate benchmark structure
            assert benchmark.latency_ms > 0
            assert benchmark.memory_peak_mb > 0
            assert benchmark.energy_consumed_mwh >= 0
            assert 0 <= benchmark.accuracy_score <= 1
            assert benchmark.thermal_increase_c >= 0
            assert benchmark.timestamp
            assert benchmark.configuration
        
        # Heavier workloads should generally have higher resource usage
        lightweight_bench = benchmarks[0]
        heavy_bench = benchmarks[2]
        
        assert heavy_bench.memory_peak_mb >= lightweight_bench.memory_peak_mb
        assert heavy_bench.energy_consumed_mwh >= lightweight_bench.energy_consumed_mwh

class TestIntegrationScenarios:
    """Test real-world integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_high_load_scenario(self, reliability_framework):
        """Test system behavior under high load."""
        # Create multiple concurrent requests
        async def simulate_high_load():
            tasks = []
            for i in range(50):  # High concurrent load
                task = asyncio.create_task(self._mock_inference_request(i))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            successes = [r for r in results if not isinstance(r, Exception)]
            failures = [r for r in results if isinstance(r, Exception)]
            
            return len(successes), len(failures)
        
        # Execute with reliability patterns
        circuit_breaker = reliability_framework.create_circuit_breaker("high_load_test", failure_threshold=10)
        bulkhead = reliability_framework.create_bulkhead("high_load_test", max_concurrent_calls=10)
        
        async def protected_high_load():
            return await reliability_framework.execute_with_reliability(
                simulate_high_load,
                circuit_breaker_name="high_load_test",
                bulkhead_name="high_load_test"
            )
        
        successes, failures = await protected_high_load()
        
        # Should handle load gracefully
        total_requests = successes + failures
        success_rate = successes / total_requests if total_requests > 0 else 0
        
        assert success_rate > 0.7, f"Success rate {success_rate:.2%} too low for high load scenario"
    
    async def _mock_inference_request(self, request_id: int):
        """Mock inference request for load testing."""
        # Simulate processing time
        processing_time = np.random.uniform(0.01, 0.1)
        await asyncio.sleep(processing_time)
        
        # Simulate occasional failures
        if np.random.random() < 0.1:  # 10% failure rate
            raise Exception(f"Mock failure for request {request_id}")
        
        return {
            "request_id": request_id,
            "result": f"Mock inference result {request_id}",
            "processing_time": processing_time
        }
    
    @pytest.mark.asyncio
    async def test_failure_recovery_scenario(self, reliability_framework):
        """Test system recovery from cascading failures."""
        # Simulate cascading failure scenario
        failure_components = ["service_a", "service_b", "service_c"]
        
        # Register health checks that will fail
        failure_states = {comp: True for comp in failure_components}
        
        def create_failing_health_check(component):
            def health_check():
                if failure_states[component]:
                    raise Exception(f"{component} is down")
                return {"status": ComponentState.HEALTHY}
            return health_check
        
        # Register recovery strategies
        def create_recovery_strategy(component):
            async def recovery_strategy(error_info):
                # Simulate recovery process
                await asyncio.sleep(0.1)
                failure_states[component] = False  # Mark as recovered
                return True
            return recovery_strategy
        
        for component in failure_components:
            reliability_framework.register_health_check(component, create_failing_health_check(component))
            reliability_framework.register_recovery_strategy(component, create_recovery_strategy(component))
        
        # Check initial health (should be failing)
        initial_health = reliability_framework.health_checker.get_overall_health()
        assert initial_health["overall_status"] in ["failed", "degraded"]
        assert len(initial_health["failed_components"]) > 0
        
        # Trigger recovery for each component
        for component in failure_components:
            await reliability_framework.self_healing.attempt_healing(
                component, 
                {"error": f"{component} failure", "error_type": "ServiceDown"}
            )
        
        # Wait for health checks to update
        await asyncio.sleep(0.5)
        
        # Check health after recovery
        recovered_health = reliability_framework.health_checker.get_overall_health()
        
        # Should show improvement
        assert len(recovered_health["failed_components"]) < len(initial_health["failed_components"])
    
    def test_memory_pressure_scenario(self, mobile_optimizer):
        """Test system behavior under memory pressure."""
        memory_manager = mobile_optimizer.memory_manager
        
        # Simulate memory pressure by allocating near capacity
        allocated_blocks = []
        allocation_size = 50.0  # MB
        
        while True:
            success = memory_manager.allocate_memory(f"pressure_block_{len(allocated_blocks)}", allocation_size)
            if success:
                allocated_blocks.append(f"pressure_block_{len(allocated_blocks)}")
            else:
                break
        
        # Should have allocated significant memory
        stats = memory_manager.get_memory_stats()
        assert stats["utilization_percent"] > 70  # High utilization
        
        # Test that new allocation triggers cleanup
        before_cleanup_blocks = len(allocated_blocks)
        cleanup_success = memory_manager.allocate_memory("post_cleanup_block", allocation_size)
        
        if cleanup_success:
            # Cleanup should have occurred
            after_stats = memory_manager.get_memory_stats()
            assert after_stats["total_allocated_mb"] <= stats["total_allocated_mb"]
        
        # Test emergency cleanup
        memory_manager._emergency_cache_cleanup()
        emergency_stats = memory_manager.get_memory_stats()
        assert emergency_stats["total_allocated_mb"] < stats["total_allocated_mb"]
    
    @pytest.mark.asyncio
    async def test_end_to_end_production_workflow(self, test_pipeline, reliability_framework, mobile_optimizer):
        """Test complete end-to-end production workflow."""
        # Create comprehensive workflow combining all systems
        demo_image = create_demo_image()
        test_question = "Analyze this image comprehensively including objects, spatial relationships, and visual hierarchy."
        
        # Step 1: Mobile optimization
        optimization_context = {
            "image_size_mb": len(demo_image) / 1024 / 1024,
            "model_complexity": 0.7,
            "question": test_question
        }
        
        optimized_settings, opt_record = mobile_optimizer.optimize_for_request(optimization_context)
        
        # Step 2: Execute inference with reliability patterns
        circuit_breaker = reliability_framework.create_circuit_breaker("e2e_test")
        
        async def protected_inference():
            # Simulate applying optimized settings
            start_time = time.time()
            result = test_pipeline.process_image_question(demo_image, test_question)
            end_time = time.time()
            
            # Apply mobile optimization effects
            actual_latency = (end_time - start_time) * 1000
            optimized_latency = actual_latency * optimized_settings.quantization_bits / 8  # Simulate quantization speedup
            
            return {
                "inference_result": result,
                "actual_latency_ms": actual_latency,
                "optimized_latency_ms": optimized_latency,
                "settings_applied": optimized_settings.__dict__
            }
        
        # Execute with reliability
        workflow_result = await reliability_framework.execute_with_reliability(
            protected_inference,
            circuit_breaker_name="e2e_test"
        )
        
        # Step 3: Validate end-to-end result
        assert "inference_result" in workflow_result
        assert "optimized_latency_ms" in workflow_result
        assert "settings_applied" in workflow_result
        
        inference_result = workflow_result["inference_result"]
        assert isinstance(inference_result, InferenceResult)
        assert inference_result.answer
        assert inference_result.confidence > 0
        
        # Optimized latency should be less than or equal to actual
        assert workflow_result["optimized_latency_ms"] <= workflow_result["actual_latency_ms"] * 1.1
        
        # Step 4: Validate system health after workflow
        overall_health = reliability_framework.health_checker.get_overall_health()
        mobile_report = mobile_optimizer.get_optimization_report()
        
        # Systems should remain healthy after workflow
        assert overall_health["overall_status"] != "failed"
        assert mobile_report["system_state"]["memory_pressure"] < 0.95  # Not critically high

# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmark tests for production validation."""
    
    @pytest.mark.performance
    def test_inference_latency_benchmark(self, test_pipeline):
        """Benchmark inference latency across different scenarios."""
        demo_image = create_demo_image()
        
        # Different complexity questions
        test_cases = [
            ("Simple", "What color is this?"),
            ("Medium", "What objects are in this image?"),
            ("Complex", "Describe the spatial relationships and visual hierarchy in detail."),
            ("Very Complex", "Provide a comprehensive analysis of the scene including objects, spatial relationships, color theory, composition principles, and contextual meaning.")
        ]
        
        benchmark_results = []
        
        for complexity, question in test_cases:
            latencies = []
            
            # Warm-up
            for _ in range(3):
                test_pipeline.process_image_question(demo_image, question)
            
            # Benchmark runs
            for _ in range(10):
                start = time.perf_counter()
                result = test_pipeline.process_image_question(demo_image, question)
                end = time.perf_counter()
                
                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)
            
            # Statistical analysis
            avg_latency = np.mean(latencies)
            p50_latency = np.percentile(latencies, 50)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            
            benchmark_results.append({
                "complexity": complexity,
                "avg_latency_ms": avg_latency,
                "p50_latency_ms": p50_latency,
                "p95_latency_ms": p95_latency,
                "p99_latency_ms": p99_latency,
                "std_dev": np.std(latencies)
            })
        
        # Print benchmark results
        print("\n📊 Inference Latency Benchmarks:")
        print("Complexity\t\tAvg\t\tP50\t\tP95\t\tP99")
        for result in benchmark_results:
            print(f"{result['complexity']:<15}\t"
                  f"{result['avg_latency_ms']:<8.1f}\t"
                  f"{result['p50_latency_ms']:<8.1f}\t"
                  f"{result['p95_latency_ms']:<8.1f}\t"
                  f"{result['p99_latency_ms']:<8.1f}")
        
        # Performance assertions
        for result in benchmark_results:
            assert result["p95_latency_ms"] < 2000, f"{result['complexity']} P95 latency too high"
            assert result["std_dev"] < result["avg_latency_ms"], f"{result['complexity']} latency too variable"
    
    @pytest.mark.performance
    def test_memory_usage_benchmark(self, test_pipeline):
        """Benchmark memory usage patterns."""
        import psutil
        import gc
        
        process = psutil.Process()
        demo_image = create_demo_image()
        
        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_measurements = []
        
        # Memory usage during inference
        for i in range(20):
            before_memory = process.memory_info().rss / 1024 / 1024
            
            result = test_pipeline.process_image_question(
                demo_image, 
                f"Test question {i} for memory benchmarking"
            )
            
            after_memory = process.memory_info().rss / 1024 / 1024
            peak_memory = max(before_memory, after_memory)
            
            memory_measurements.append({
                "iteration": i,
                "before_mb": before_memory,
                "after_mb": after_memory,
                "peak_mb": peak_memory,
                "delta_mb": after_memory - before_memory
            })
        
        # Analyze memory patterns
        avg_delta = np.mean([m["delta_mb"] for m in memory_measurements])
        max_delta = max([m["delta_mb"] for m in memory_measurements])
        final_memory = memory_measurements[-1]["after_mb"]
        
        print(f"\n🧠 Memory Benchmark Results:")
        print(f"Baseline Memory: {baseline_memory:.1f} MB")
        print(f"Final Memory: {final_memory:.1f} MB")
        print(f"Average Delta: {avg_delta:.1f} MB")
        print(f"Max Delta: {max_delta:.1f} MB")
        print(f"Total Growth: {final_memory - baseline_memory:.1f} MB")
        
        # Memory usage assertions
        assert max_delta < 200, f"Peak memory delta {max_delta:.1f}MB too high"
        memory_growth = final_memory - baseline_memory
        assert memory_growth < 500, f"Total memory growth {memory_growth:.1f}MB too high"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_performance(self, test_pipeline):
        """Test performance under concurrent load."""
        demo_image = create_demo_image()
        concurrency_levels = [1, 2, 5, 10, 20]
        
        async def single_inference(request_id: int):
            start_time = time.perf_counter()
            result = test_pipeline.process_image_question(
                demo_image, 
                f"Concurrent request {request_id}"
            )
            end_time = time.perf_counter()
            
            return {
                "request_id": request_id,
                "latency_ms": (end_time - start_time) * 1000,
                "success": True
            }
        
        concurrency_results = []
        
        for concurrency in concurrency_levels:
            print(f"\n🔄 Testing concurrency level: {concurrency}")
            
            # Create concurrent tasks
            tasks = [single_inference(i) for i in range(concurrency)]
            
            start_time = time.perf_counter()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.perf_counter()
            
            # Analyze results
            successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
            failed_results = [r for r in results if not (isinstance(r, dict) and r.get("success"))]
            
            if successful_results:
                latencies = [r["latency_ms"] for r in successful_results]
                avg_latency = np.mean(latencies)
                p95_latency = np.percentile(latencies, 95)
            else:
                avg_latency = 0
                p95_latency = 0
            
            total_time = (end_time - start_time) * 1000
            throughput = len(successful_results) / (total_time / 1000) if total_time > 0 else 0
            
            concurrency_result = {
                "concurrency": concurrency,
                "total_requests": concurrency,
                "successful_requests": len(successful_results),
                "failed_requests": len(failed_results),
                "success_rate": len(successful_results) / concurrency,
                "avg_latency_ms": avg_latency,
                "p95_latency_ms": p95_latency,
                "total_time_ms": total_time,
                "throughput_rps": throughput
            }
            
            concurrency_results.append(concurrency_result)
            
            print(f"   Success Rate: {concurrency_result['success_rate']:.1%}")
            print(f"   Avg Latency: {avg_latency:.1f}ms")
            print(f"   Throughput: {throughput:.2f} RPS")
        
        # Performance analysis
        print(f"\n📊 Concurrency Performance Summary:")
        print("Concurrency\tSuccess Rate\tAvg Latency\tThroughput")
        for result in concurrency_results:
            print(f"{result['concurrency']}\t\t"
                  f"{result['success_rate']:.1%}\t\t"
                  f"{result['avg_latency_ms']:.1f}ms\t\t"
                  f"{result['throughput_rps']:.2f} RPS")
        
        # Performance assertions
        for result in concurrency_results:
            # Success rate should remain reasonable even under load
            assert result["success_rate"] > 0.8, f"Success rate {result['success_rate']:.1%} too low at concurrency {result['concurrency']}"
            
            # Latency shouldn't degrade too much
            single_thread_latency = concurrency_results[0]["avg_latency_ms"]
            latency_degradation = result["avg_latency_ms"] / single_thread_latency
            assert latency_degradation < 3.0, f"Latency degradation {latency_degradation:.1f}x too high at concurrency {result['concurrency']}"

if __name__ == "__main__":
    # Run with pytest-asyncio for async tests
    pytest.main([__file__, "-v", "--tb=short"])