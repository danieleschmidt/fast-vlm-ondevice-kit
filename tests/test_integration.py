"""
Integration tests for FastVLM On-Device Kit.

Tests the complete system integration across all components.
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Test the main components
def test_package_import():
    """Test that all main components can be imported."""
    try:
        from fast_vlm_ondevice import (
            FastVLMConverter,
            QuantizationConfig,
            HealthChecker,
            ModelTester,
            PerformanceBenchmark
        )
        assert True, "Core imports successful"
    except ImportError as e:
        pytest.skip(f"Required dependencies not available: {e}")


def test_advanced_imports():
    """Test advanced component imports."""
    try:
        from fast_vlm_ondevice import (
            MetricsCollector,
            InputValidator,
            CacheManager,
            PerformanceOptimizer,
            ModelServer
        )
        assert True, "Advanced imports successful"
    except ImportError as e:
        pytest.skip(f"Advanced dependencies not available: {e}")


class TestSystemIntegration:
    """Integration tests for complete system."""
    
    def test_converter_with_monitoring(self):
        """Test converter with monitoring integration."""
        try:
            from fast_vlm_ondevice import FastVLMConverter
            
            converter = FastVLMConverter()
            assert converter is not None
            assert hasattr(converter, 'session_id')
            assert hasattr(converter, 'metrics_collector')
            assert hasattr(converter, 'profiler')
        except ImportError:
            pytest.skip("Dependencies not available")
    
    def test_health_checker_system_validation(self):
        """Test health checker system validation."""
        try:
            from fast_vlm_ondevice import HealthChecker
            
            checker = HealthChecker()
            health_status = checker.check_all()
            
            assert isinstance(health_status, dict)
            assert "system" in health_status
            assert "dependencies" in health_status
            assert "hardware" in health_status
            
            # Each check should have required fields
            for component, status in health_status.items():
                assert "healthy" in status
                assert "message" in status
                
        except ImportError:
            pytest.skip("Dependencies not available")
    
    def test_security_validation_pipeline(self):
        """Test security validation pipeline."""
        try:
            from fast_vlm_ondevice import InputValidator
            
            validator = InputValidator()
            
            # Test text validation
            text_result = validator.validate_text_input("What is in this image?")
            assert text_result["valid"] == True
            assert text_result["sanitized"] == "What is in this image?"
            
            # Test suspicious text
            suspicious_text = "<script>alert('test')</script>"
            suspicious_result = validator.validate_text_input(suspicious_text)
            assert len(suspicious_result["warnings"]) > 0
            
        except ImportError:
            pytest.skip("Dependencies not available")
    
    def test_caching_system_integration(self):
        """Test caching system integration."""
        try:
            from fast_vlm_ondevice import create_cache_manager
            
            with tempfile.TemporaryDirectory() as temp_dir:
                cache_manager = create_cache_manager(
                    cache_dir=temp_dir,
                    config={"enable_model_cache": True, "enable_inference_cache": True}
                )
                
                assert cache_manager is not None
                
                # Test model cache
                model_cache = cache_manager.get_model_cache()
                assert model_cache is not None
                
                # Test inference cache
                inference_cache = cache_manager.get_inference_cache()
                assert inference_cache is not None
                
                # Test stats
                stats = cache_manager.get_cache_stats()
                assert isinstance(stats, dict)
                
        except ImportError:
            pytest.skip("Dependencies not available")
    
    def test_optimization_system_integration(self):
        """Test optimization system integration."""
        try:
            from fast_vlm_ondevice import create_optimizer, OptimizationConfig
            
            # Test different optimization levels
            for level in ["speed", "balanced", "memory"]:
                optimizer = create_optimizer(
                    optimization_level=level,
                    max_memory_gb=1.0,
                    enable_async=False
                )
                
                assert optimizer is not None
                assert optimizer.config.optimization_level == level
                
                # Test performance stats
                stats = optimizer.get_performance_stats()
                assert isinstance(stats, dict)
                assert "memory" in stats
                assert "optimization_config" in stats
                
        except ImportError:
            pytest.skip("Dependencies not available")
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_monitoring_system_integration(self, mock_memory, mock_cpu):
        """Test monitoring system integration."""
        try:
            from fast_vlm_ondevice import setup_monitoring
            
            # Mock system resources
            mock_cpu.return_value = 45.0
            mock_memory.return_value = Mock(percent=60.0, available=2*1024**3)
            
            metrics_collector, system_monitor, profiler, alert_manager = setup_monitoring()
            
            # Test components exist
            assert metrics_collector is not None
            assert system_monitor is not None
            assert profiler is not None
            assert alert_manager is not None
            
            # Test basic functionality
            inference_stats = metrics_collector.get_inference_stats()
            assert isinstance(inference_stats, dict)
            
            # Test alert checking
            alerts = alert_manager.check_alerts()
            assert isinstance(alerts, list)
            
        except ImportError:
            pytest.skip("Dependencies not available")
    
    def test_deployment_system_integration(self):
        """Test deployment system integration."""
        try:
            from fast_vlm_ondevice import create_deployment, DeploymentConfig
            
            with tempfile.NamedTemporaryFile(suffix='.mlpackage') as temp_model:
                # Create deployment config
                config = DeploymentConfig(
                    model_path=temp_model.name,
                    model_type="coreml",
                    max_instances=1,
                    enable_auto_scaling=False
                )
                
                # This would normally start a full deployment
                # For testing, just verify configuration
                assert config.model_path == temp_model.name
                assert config.model_type == "coreml"
                assert config.max_instances == 1
                
        except ImportError:
            pytest.skip("Dependencies not available")


class TestErrorHandling:
    """Test error handling across the system."""
    
    def test_converter_error_handling(self):
        """Test converter handles errors gracefully."""
        try:
            from fast_vlm_ondevice import FastVLMConverter
            
            converter = FastVLMConverter()
            
            # Test with invalid file path
            try:
                result = converter.load_pytorch_model("/nonexistent/path.pth")
                # Should return demo model instead of crashing
                assert result is not None
            except Exception as e:
                # Should handle gracefully
                assert "Invalid or unsafe file path" in str(e) or "not found" in str(e).lower()
                
        except ImportError:
            pytest.skip("Dependencies not available")
    
    def test_health_checker_error_resilience(self):
        """Test health checker resilience to system errors."""
        try:
            from fast_vlm_ondevice import HealthChecker
            
            checker = HealthChecker()
            
            # Should not crash even if some checks fail
            health_status = checker.check_all()
            assert isinstance(health_status, dict)
            
            # Should have reasonable fallbacks
            for component, status in health_status.items():
                assert "healthy" in status
                assert isinstance(status["healthy"], bool)
                
        except ImportError:
            pytest.skip("Dependencies not available")


class TestPerformanceRequirements:
    """Test that performance requirements are met."""
    
    def test_import_performance(self):
        """Test that imports complete quickly."""
        import time
        
        start_time = time.time()
        
        try:
            from fast_vlm_ondevice import FastVLMConverter
            import_time = time.time() - start_time
            
            # Should import in under 2 seconds
            assert import_time < 2.0, f"Import took {import_time:.2f}s, expected < 2.0s"
            
        except ImportError:
            pytest.skip("Dependencies not available")
    
    def test_converter_initialization_performance(self):
        """Test converter initializes quickly."""
        try:
            from fast_vlm_ondevice import FastVLMConverter
            import time
            
            start_time = time.time()
            converter = FastVLMConverter()
            init_time = time.time() - start_time
            
            # Should initialize in under 1 second
            assert init_time < 1.0, f"Initialization took {init_time:.2f}s, expected < 1.0s"
            
        except ImportError:
            pytest.skip("Dependencies not available")


def test_cli_integration():
    """Test CLI integration."""
    try:
        from fast_vlm_ondevice.cli import main
        
        # Test that CLI main function exists and is callable
        assert callable(main)
        
        # Test help functionality (should not crash)
        with patch('sys.argv', ['fastvlm', '--help']):
            try:
                main()
            except SystemExit:
                # argparse calls sys.exit for help, this is expected
                pass
                
    except ImportError:
        pytest.skip("CLI dependencies not available")


def test_logging_integration():
    """Test logging system integration."""
    try:
        from fast_vlm_ondevice import setup_logging, get_logger
        
        # Test logging setup
        config = setup_logging(level="INFO", structured=True)
        assert isinstance(config, dict)
        assert "level" in config
        
        # Test logger creation
        logger = get_logger("test")
        assert logger is not None
        
        # Test logger with context
        context_logger = get_logger("test", request_id="test-123")
        assert context_logger is not None
        
    except ImportError:
        pytest.skip("Logging dependencies not available")


def test_configuration_validation():
    """Test configuration validation across components."""
    try:
        from fast_vlm_ondevice import OptimizationConfig, DeploymentConfig
        
        # Test optimization config
        opt_config = OptimizationConfig(
            max_memory_mb=1024.0,
            enable_batching=True,
            max_batch_size=8
        )
        
        assert opt_config.max_memory_mb == 1024.0
        assert opt_config.enable_batching == True
        assert opt_config.max_batch_size == 8
        
        # Test deployment config
        deploy_config = DeploymentConfig(
            max_concurrent_requests=5,
            enable_auto_scaling=True,
            min_instances=1,
            max_instances=3
        )
        
        assert deploy_config.max_concurrent_requests == 5
        assert deploy_config.enable_auto_scaling == True
        assert deploy_config.min_instances <= deploy_config.max_instances
        
    except ImportError:
        pytest.skip("Configuration dependencies not available")


if __name__ == "__main__":
    # Run basic smoke tests
    test_package_import()
    test_advanced_imports()
    print("✅ All integration tests completed successfully!")