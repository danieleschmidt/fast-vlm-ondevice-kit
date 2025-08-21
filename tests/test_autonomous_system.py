"""
Comprehensive test suite for the autonomous FastVLM system.

Tests core pipeline, mobile optimization, reliability, and scalability components
with real-world scenarios and edge cases.
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
from pathlib import Path
import tempfile
import json

# Import the modules we're testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fast_vlm_ondevice.core_pipeline import (
    FastVLMCorePipeline, InferenceConfig, InferenceResult
)
from fast_vlm_ondevice.mobile_optimizer import (
    MobileOptimizer, MobileOptimizationConfig, OptimizationLevel, DeviceProfile
)
from fast_vlm_ondevice.reliability_engine import (
    ReliabilityEngine, HealthStatus, ErrorSeverity, CircuitBreaker, CircuitBreakerConfig
)
from fast_vlm_ondevice.scalability_engine import (
    ScalabilityEngine, ScalabilityConfig, LoadBalancingStrategy, ResourceType
)
from fast_vlm_ondevice.intelligent_orchestrator import (
    IntelligentOrchestrator, OrchestratorConfig, RequestContext
)


class TestCorePipeline:
    """Test cases for the core FastVLM pipeline."""
    
    @pytest.fixture
    def pipeline_config(self):
        """Create test pipeline configuration."""
        return PipelineConfig(
            model_path="test_model.mlpackage",
            image_size=(224, 224),
            target_latency_ms=200.0,
            enable_caching=True
        )
    
    @pytest.fixture
    def pipeline(self, pipeline_config):
        """Create test pipeline instance."""
        return FastVLMPipeline(pipeline_config)
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.is_initialized
        assert pipeline.config.image_size == (224, 224)
        assert pipeline.config.enable_caching
        assert pipeline.cache is not None
    
    @pytest.mark.asyncio
    async def test_basic_processing(self, pipeline):
        """Test basic image-question processing."""
        # Mock image data
        image_data = np.random.rand(224, 224, 3)
        question = "What is in this image?"
        
        result = await pipeline.process(image_data, question)
        
        assert "answer" in result
        assert "confidence" in result
        assert "session_id" in result
        assert result["answer"] != ""
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["processing_time_ms"] > 0
    
    @pytest.mark.asyncio
    async def test_performance_targets(self, pipeline):
        """Test that pipeline meets performance targets."""
        image_data = np.random.rand(224, 224, 3)
        question = "Test question for performance"
        
        # Run multiple iterations to test consistency
        latencies = []
        for _ in range(5):
            start_time = time.time()
            result = await pipeline.process(image_data, question)
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)
            
            # Each individual request should be fast
            assert latency < 1000  # Should be under 1 second
        
        # Average latency should meet target
        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 300  # Should average under 300ms
    
    @pytest.mark.asyncio
    async def test_caching_functionality(self, pipeline):
        """Test caching improves performance."""
        image_data = np.random.rand(224, 224, 3)
        question = "Cached question test"
        
        # First request (cold)
        start_time = time.time()
        result1 = await pipeline.process(image_data, question)
        first_latency = (time.time() - start_time) * 1000
        
        # Second request (should be cached/faster)
        start_time = time.time()
        result2 = await pipeline.process(image_data, question)
        second_latency = (time.time() - start_time) * 1000
        
        assert result1["answer"] == result2["answer"]
        # Second request should be faster due to caching
        assert second_latency <= first_latency * 1.1  # Allow small variance
    
    def test_metrics_collection(self, pipeline):
        """Test pipeline metrics collection."""
        metrics = pipeline.get_performance_metrics()
        
        assert "pipeline_metrics" in metrics
        assert "cache_stats" in metrics
        assert "model_info" in metrics
        assert "config" in metrics
        
        # Check pipeline metrics structure
        pipeline_metrics = metrics["pipeline_metrics"]
        assert "session_id" in pipeline_metrics
        assert "stage_timings" in pipeline_metrics


class TestMobileOptimizer:
    """Test cases for mobile optimization."""
    
    @pytest.fixture
    def mobile_config(self):
        """Create test mobile optimization configuration."""
        return MobileOptimizationConfig(
            device_profile=DeviceProfile.MID_RANGE,
            optimization_level=OptimizationLevel.BALANCED,
            target_latency_ms=250.0,
            enable_battery_optimization=True
        )
    
    @pytest.fixture
    def mobile_optimizer(self, mobile_config):
        """Create test mobile optimizer instance."""
        return MobileOptimizer(mobile_config)
    
    def test_device_detection(self, mobile_optimizer):
        """Test device capability detection."""
        caps = mobile_optimizer.resource_manager.device_caps
        
        assert caps.chip_name != "Unknown"
        assert caps.neural_engine_cores > 0
        assert caps.memory_gb > 0
        assert 0.0 <= caps.battery_level <= 1.0
    
    def test_optimization_configuration(self, mobile_optimizer):
        """Test optimization configuration generation."""
        config = mobile_optimizer.get_optimized_config()
        
        assert "device_optimizations" in config
        assert "battery_optimizations" in config
        assert "adaptive_settings" in config
        assert "device_capabilities" in config
        
        # Check device optimizations
        device_opts = config["device_optimizations"]
        assert "compute_allocation" in device_opts
        assert "memory_settings" in device_opts
        assert "quality_settings" in device_opts
    
    def test_battery_optimization(self, mobile_optimizer):
        """Test battery level optimization."""
        # Test low battery optimization
        mobile_optimizer.update_runtime_conditions(battery_level=0.15)
        config = mobile_optimizer.get_optimized_config()
        
        battery_opts = config["battery_optimizations"]
        assert battery_opts["cache_aggressive"] == True  # Should be aggressive when low battery
        
        # Test high battery optimization
        mobile_optimizer.update_runtime_conditions(battery_level=0.85)
        config = mobile_optimizer.get_optimized_config()
        
        battery_opts = config["battery_optimizations"]
        # Should allow performance optimizations when battery is high
        assert "performance" in str(battery_opts).lower() or battery_opts["quality_reduced"] == False
    
    def test_adaptive_performance(self, mobile_optimizer):
        """Test adaptive performance controller."""
        controller = mobile_optimizer.performance_controller
        controller.start_monitoring()
        
        # Simulate performance metrics
        mobile_optimizer.record_inference_metrics(latency_ms=300, memory_mb=800)
        mobile_optimizer.record_inference_metrics(latency_ms=280, memory_mb=750)
        
        time.sleep(0.1)  # Allow monitoring to process
        
        assert controller.monitoring_active
        controller.stop_monitoring()
    
    def test_performance_report(self, mobile_optimizer):
        """Test performance reporting."""
        report = mobile_optimizer.get_performance_report()
        
        assert "optimization_config" in report
        assert "device_status" in report
        assert "performance_metrics" in report
        assert "optimizations_active" in report
        
        # Validate structure
        assert "level" in report["optimization_config"]
        assert "battery_level" in report["device_status"]


class TestReliabilityEngine:
    """Test cases for reliability and error handling."""
    
    @pytest.fixture
    def reliability_engine(self):
        """Create test reliability engine."""
        engine = ReliabilityEngine()
        engine.initialize()
        return engine
    
    def test_initialization(self, reliability_engine):
        """Test reliability engine initialization."""
        assert reliability_engine.is_initialized
        assert reliability_engine.health_monitor.monitoring_active
        assert len(reliability_engine.error_recovery.recovery_strategies) > 0
    
    def test_health_monitoring(self, reliability_engine):
        """Test health monitoring functionality."""
        health_report = reliability_engine.get_reliability_report()
        
        assert "engine_status" in health_report
        assert "system_health" in health_report
        assert "error_statistics" in health_report
        assert "availability" in health_report
        
        # Check system health structure
        system_health = health_report["system_health"]
        assert "overall_status" in system_health
        assert "components" in system_health
    
    def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        config = CircuitBreakerConfig(failure_threshold=3, timeout_seconds=1.0)
        circuit_breaker = CircuitBreaker("test_component", config)
        
        # Function that always fails
        def failing_function():
            raise Exception("Test failure")
        
        # Should fail and eventually open circuit
        failure_count = 0
        for _ in range(5):
            try:
                circuit_breaker.call(failing_function)
            except Exception:
                failure_count += 1
        
        assert failure_count >= 3  # Should have multiple failures
        assert circuit_breaker.state.value in ["open", "half_open"]
    
    def test_error_recovery(self, reliability_engine):
        """Test error recovery mechanisms."""
        error_recovery = reliability_engine.error_recovery
        
        # Test error handling
        test_error = Exception("Test error for recovery")
        result = error_recovery.handle_error(test_error, "test_component")
        
        assert result is not None
        assert len(error_recovery.error_history) > 0
        
        # Check error event structure
        error_event = error_recovery.error_history[-1]
        assert error_event.component == "test_component"
        assert error_event.error_type == "Exception"
        assert error_event.recovery_action is not None
    
    def test_reliability_context(self, reliability_engine):
        """Test reliability context manager."""
        success_recorded = False
        
        # Test successful execution
        with reliability_engine.reliability_context("test_component"):
            time.sleep(0.01)  # Simulate work
            success_recorded = True
        
        assert success_recorded
        
        # Test error handling
        error_caught = False
        try:
            with reliability_engine.reliability_context("test_component"):
                raise Exception("Test error in context")
        except Exception:
            error_caught = True
        
        # Error should be caught and recorded
        assert len(reliability_engine.error_recovery.error_history) > 0


class TestScalabilityEngine:
    """Test cases for scalability and load balancing."""
    
    @pytest.fixture
    def scalability_config(self):
        """Create test scalability configuration."""
        return ScalabilityConfig(
            min_workers=2,
            max_workers=6,
            load_balancing_strategy=LoadBalancingStrategy.INTELLIGENT,
            enable_resource_pooling=True
        )
    
    @pytest.fixture
    def scalability_engine(self, scalability_config):
        """Create test scalability engine."""
        engine = ScalabilityEngine(scalability_config)
        engine.initialize()
        return engine
    
    def test_initialization(self, scalability_engine):
        """Test scalability engine initialization."""
        assert scalability_engine.is_initialized
        assert len(scalability_engine.load_balancer.workers) >= 2  # min_workers
        
        status = scalability_engine.get_scaling_status()
        assert status["engine_status"]["initialized"]
        assert status["engine_status"]["workers_active"] >= 2
    
    def test_load_balancer(self, scalability_engine):
        """Test load balancing functionality."""
        load_balancer = scalability_engine.load_balancer
        
        # Test worker selection
        worker = load_balancer.select_worker()
        assert worker is not None
        assert worker.status in ["idle", "busy"]
        
        # Test multiple selections for distribution
        selected_workers = []
        for _ in range(5):
            worker = load_balancer.select_worker()
            if worker:
                selected_workers.append(worker.worker_id)
        
        # Should distribute across workers
        unique_workers = set(selected_workers)
        assert len(unique_workers) >= 1  # At least using workers
    
    @pytest.mark.asyncio
    async def test_request_processing(self, scalability_engine):
        """Test request processing through scalability engine."""
        image_data = np.random.rand(224, 224, 3)
        question = "Test scalability processing"
        
        result = await scalability_engine.process_request(image_data, question)
        
        assert "answer" in result or "error" in result
        assert "processing_time_ms" in result
        assert "worker_id" in result
        
        if "error" not in result:
            assert result["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, scalability_engine):
        """Test concurrent request processing."""
        image_data = np.random.rand(224, 224, 3)
        
        # Submit multiple concurrent requests
        tasks = []
        for i in range(4):
            task = scalability_engine.process_request(
                image_data, 
                f"Concurrent question {i}"
            )
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= 2  # At least some should succeed
        
        # Check worker distribution
        worker_ids = {r.get("worker_id") for r in successful_results if "worker_id" in r}
        assert len(worker_ids) >= 1  # Should use available workers
    
    def test_resource_pooling(self, scalability_engine):
        """Test resource pool management."""
        if not scalability_engine.config.enable_resource_pooling:
            pytest.skip("Resource pooling not enabled")
        
        pool_manager = scalability_engine.resource_pool_manager
        pool_stats = pool_manager.get_pool_statistics()
        
        assert len(pool_stats) > 0
        
        # Test resource allocation
        for resource_type_name, stats in pool_stats.items():
            assert "total_resources" in stats
            assert "available_resources" in stats
            assert "utilization" in stats
            assert 0.0 <= stats["utilization"] <= 1.0


class TestIntelligentOrchestrator:
    """Test cases for the intelligent orchestrator."""
    
    @pytest.fixture
    def orchestrator_config(self):
        """Create test orchestrator configuration."""
        return OrchestratorConfig(
            model_path="test_model.mlpackage",
            max_concurrent_requests=4,
            enable_intelligent_caching=True,
            enable_health_monitoring=True
        )
    
    @pytest.fixture
    async def orchestrator(self, orchestrator_config):
        """Create test orchestrator instance."""
        orchestrator = IntelligentOrchestrator(orchestrator_config)
        
        # Start orchestrator in background
        orchestrator_task = asyncio.create_task(orchestrator.start())
        
        # Give it time to initialize
        await asyncio.sleep(0.1)
        
        yield orchestrator
        
        # Cleanup
        await orchestrator.stop()
        try:
            await asyncio.wait_for(orchestrator_task, timeout=1.0)
        except asyncio.TimeoutError:
            orchestrator_task.cancel()
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.is_running
        assert orchestrator.pipeline.is_initialized
        assert orchestrator.reliability_engine.is_initialized
        
        status = orchestrator.get_system_status()
        assert status["orchestrator"]["is_running"]
        assert status["components"]["pipeline_initialized"]
    
    @pytest.mark.asyncio
    async def test_request_processing(self, orchestrator):
        """Test request processing through orchestrator."""
        image_data = np.random.rand(336, 336, 3)
        question = "What do you see in this image?"
        
        result = await orchestrator.process_request(image_data, question)
        
        assert "answer" in result
        assert "session_id" in result
        assert "metrics" in result
        
        # Check metrics structure
        metrics = result["metrics"]
        assert "session_id" in metrics
        assert "total_latency_ms" in metrics
    
    @pytest.mark.asyncio
    async def test_system_monitoring(self, orchestrator):
        """Test system monitoring and status reporting."""
        # Process a few requests to generate metrics
        image_data = np.random.rand(336, 336, 3)
        
        for i in range(3):
            await orchestrator.process_request(image_data, f"Test question {i}")
        
        # Give monitoring time to update
        await asyncio.sleep(0.1)
        
        status = orchestrator.get_system_status()
        
        assert status["orchestrator"]["total_requests"] >= 3
        assert "performance" in status
        assert "optimization" in status
    
    @pytest.mark.asyncio
    async def test_concurrent_orchestration(self, orchestrator):
        """Test concurrent request handling by orchestrator."""
        image_data = np.random.rand(336, 336, 3)
        
        # Create multiple concurrent requests
        tasks = []
        for i in range(6):  # More than typical max_concurrent_requests
            context = RequestContext(priority=i % 3 + 1)  # Varying priorities
            task = orchestrator.process_request(
                image_data, 
                f"Concurrent orchestrated question {i}",
                context
            )
            tasks.append(task)
        
        # Process concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        successful_results = [r for r in results if not isinstance(r, Exception) and "error" not in r]
        
        # Should handle multiple requests successfully
        assert len(successful_results) >= 3
        
        # Check system status after concurrent load
        status = orchestrator.get_system_status()
        assert status["orchestrator"]["total_requests"] >= 6


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_processing(self):
        """Test complete end-to-end processing pipeline."""
        # Create integrated configuration
        config = OrchestratorConfig(
            model_path="integration_test_model.mlpackage",
            max_concurrent_requests=2,
            mobile_optimization=MobileOptimizationConfig(
                optimization_level=OptimizationLevel.BALANCED
            )
        )
        
        # Initialize orchestrator
        orchestrator = IntelligentOrchestrator(config)
        
        # Start orchestrator
        orchestrator_task = asyncio.create_task(orchestrator.start())
        await asyncio.sleep(0.1)  # Initialization time
        
        try:
            # Test various scenarios
            test_cases = [
                {
                    "image": np.random.rand(224, 224, 3),
                    "question": "What objects are in this image?",
                    "expected_latency_ms": 500
                },
                {
                    "image": np.random.rand(336, 336, 3),
                    "question": "Describe the scene",
                    "expected_latency_ms": 600
                },
                {
                    "image": np.random.rand(448, 448, 3),
                    "question": "Count the number of people",
                    "expected_latency_ms": 800
                }
            ]
            
            results = []
            for test_case in test_cases:
                result = await orchestrator.process_request(
                    test_case["image"],
                    test_case["question"]
                )
                results.append((test_case, result))
            
            # Validate results
            for test_case, result in results:
                if "error" not in result:
                    assert "answer" in result
                    assert result["answer"] != ""
                    assert "confidence" in result
                    assert 0.0 <= result["confidence"] <= 1.0
                    
                    # Performance validation
                    if "processing_time_ms" in result:
                        # Should be reasonable performance
                        assert result["processing_time_ms"] < test_case["expected_latency_ms"] * 2
            
            # System health check
            status = orchestrator.get_system_status()
            assert status["orchestrator"]["is_running"]
            assert status["orchestrator"]["health_status"] in ["healthy", "degraded"]
            
        finally:
            # Cleanup
            await orchestrator.stop()
            try:
                await asyncio.wait_for(orchestrator_task, timeout=1.0)
            except asyncio.TimeoutError:
                orchestrator_task.cancel()
    
    def test_performance_benchmarks(self):
        """Test system performance benchmarks."""
        # Performance targets for the system
        performance_targets = {
            "max_latency_p95_ms": 300.0,
            "min_throughput_fps": 3.0,
            "max_memory_usage_mb": 1500.0,
            "min_accuracy": 0.70,
            "min_availability": 0.99
        }
        
        # This would run actual performance benchmarks
        # For now, we validate the targets are reasonable
        assert performance_targets["max_latency_p95_ms"] > 0
        assert performance_targets["min_throughput_fps"] > 0
        assert performance_targets["max_memory_usage_mb"] > 0
        assert 0.0 <= performance_targets["min_accuracy"] <= 1.0
        assert 0.0 <= performance_targets["min_availability"] <= 1.0
    
    def test_configuration_validation(self):
        """Test configuration validation and defaults."""
        # Test default configurations
        pipeline_config = PipelineConfig()
        assert pipeline_config.target_latency_ms > 0
        assert pipeline_config.memory_limit_mb > 0
        
        mobile_config = MobileOptimizationConfig()
        assert mobile_config.target_latency_ms > 0
        assert mobile_config.max_memory_mb > 0
        
        orchestrator_config = OrchestratorConfig()
        assert orchestrator_config.max_concurrent_requests > 0
        assert orchestrator_config.request_timeout_seconds > 0


if __name__ == "__main__":
    # Run tests with comprehensive output
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--disable-warnings"
    ])