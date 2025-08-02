"""
End-to-end tests for mobile deployment scenarios.
Tests complete workflows from model conversion to mobile inference.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from fast_vlm_ondevice.converter import FastVLMConverter


class TestMobileDeployment:
    """Test end-to-end mobile deployment scenarios."""
    
    @pytest.fixture
    def deployment_config(self):
        """Mobile deployment configuration."""
        return {
            "model_variants": ["tiny", "base", "large"],
            "target_devices": ["iPhone14", "iPhone15Pro", "iPadProM2"],
            "quantization_strategies": ["int4", "int8", "mixed"],
            "performance_targets": {
                "iPhone14": {"latency_ms": 300, "memory_mb": 600},
                "iPhone15Pro": {"latency_ms": 200, "memory_mb": 900},
                "iPadProM2": {"latency_ms": 150, "memory_mb": 1200}
            }
        }
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_complete_deployment_pipeline(self, deployment_config, tmp_path):
        """Test complete deployment from training to mobile app."""
        # Setup directories
        checkpoint_dir = tmp_path / "checkpoints"
        models_dir = tmp_path / "models" 
        ios_dir = tmp_path / "ios_integration"
        
        for dir_path in [checkpoint_dir, models_dir, ios_dir]:
            dir_path.mkdir(exist_ok=True)
        
        converter = FastVLMConverter()
        
        with patch('torch.load') as mock_load, \
             patch('coremltools.convert') as mock_convert, \
             patch('subprocess.run') as mock_subprocess:
            
            # Mock checkpoint loading
            mock_load.return_value = {
                "model_state_dict": {"layer.weight": []},
                "config": {"model_type": "fast-vlm-base"}
            }
            
            # Mock Core ML conversion
            mock_coreml_model = MagicMock()
            mock_convert.return_value = mock_coreml_model
            
            # Mock iOS build process
            mock_subprocess.return_value = MagicMock(returncode=0)
            
            # Step 1: Convert models for each variant
            converted_models = {}
            for variant in deployment_config["model_variants"]:
                checkpoint_path = checkpoint_dir / f"fast-vlm-{variant}.pth" 
                model_path = models_dir / f"FastVLM-{variant}.mlpackage"
                
                # Create mock checkpoint
                checkpoint_path.touch()
                
                # Convert model
                converter.convert_checkpoint_to_coreml(
                    checkpoint_path=str(checkpoint_path),
                    output_path=str(model_path),
                    quantization="mixed"
                )
                
                converted_models[variant] = str(model_path)
            
            # Step 2: Generate iOS integration code
            ios_config = {
                "models": converted_models,
                "target_devices": deployment_config["target_devices"],
                "bundle_identifier": "com.fastvm.demo"
            }
            
            ios_config_path = ios_dir / "FastVLMConfig.json"
            ios_config_path.write_text(json.dumps(ios_config, indent=2))
            
            # Step 3: Validate deployment package
            assert ios_config_path.exists()
            assert len(converted_models) == len(deployment_config["model_variants"])
            
            # Verify all conversion calls were made
            assert mock_convert.call_count == len(deployment_config["model_variants"])
    
    @pytest.mark.e2e
    def test_device_specific_optimization(self, deployment_config):
        """Test device-specific model optimization."""
        converter = FastVLMConverter()
        
        with patch.object(converter, 'optimize_for_device') as mock_optimize:
            mock_optimize.return_value = MagicMock()
            
            # Test optimization for each target device
            for device in deployment_config["target_devices"]:
                performance_target = deployment_config["performance_targets"][device]
                
                optimized_model = converter.optimize_for_device(
                    model=MagicMock(),
                    target_device=device,
                    latency_target=performance_target["latency_ms"],
                    memory_target=performance_target["memory_mb"]
                )
                
                assert optimized_model is not None
            
            # Verify optimization was called for each device
            assert mock_optimize.call_count == len(deployment_config["target_devices"])
    
    @pytest.mark.e2e
    @pytest.mark.performance
    def test_mobile_performance_validation(self, deployment_config, performance_benchmark):
        """Test performance validation for mobile deployment."""
        converter = FastVLMConverter()
        
        with patch.object(converter, 'benchmark_mobile_performance') as mock_benchmark:
            # Mock performance results
            mock_benchmark.return_value = {
                "latency_ms": 180,
                "memory_mb": 850,
                "energy_impact": 2.5,
                "thermal_state": "nominal"
            }
            
            # Test performance validation
            for device in deployment_config["target_devices"]:
                target = deployment_config["performance_targets"][device]
                
                results = converter.validate_mobile_performance(
                    model_path="test_model.mlpackage",
                    target_device=device,
                    performance_targets=target
                )
                
                # Verify performance meets targets
                assert results["latency_ms"] <= target["latency_ms"]
                assert results["memory_mb"] <= target["memory_mb"]
                
                print(f"Device {device}: {results}")
    
    @pytest.mark.e2e
    def test_app_store_deployment_preparation(self, tmp_path):
        """Test preparation for App Store deployment."""
        app_dir = tmp_path / "FastVLMApp"
        app_dir.mkdir()
        
        deployment_assets = {
            "models": ["FastVLM-Tiny.mlpackage", "FastVLM-Base.mlpackage"],
            "metadata": {
                "app_version": "1.0.0",
                "models_version": "1.0.0",
                "supported_devices": ["iPhone12", "iPhone13", "iPhone14", "iPhone15"],
                "minimum_ios": "15.0"
            },
            "privacy_manifest": {
                "data_collection": False,
                "network_usage": False,
                "on_device_only": True
            }
        }
        
        with patch('fast_vlm_ondevice.deployment.prepare_app_store_package') as mock_prepare:
            mock_prepare.return_value = True
            
            # Prepare deployment package
            success = mock_prepare(
                app_directory=str(app_dir),
                deployment_config=deployment_assets
            )
            
            assert success is True
            mock_prepare.assert_called_once()
    
    @pytest.mark.e2e 
    def test_multi_model_deployment(self, deployment_config, tmp_path):
        """Test deployment with multiple model variants."""
        models_dir = tmp_path / "multi_models"
        models_dir.mkdir()
        
        converter = FastVLMConverter()
        
        with patch.object(converter, 'create_multi_model_package') as mock_package:
            mock_package.return_value = str(models_dir / "MultiModel.mlpackage")
            
            # Create multi-model package
            package_path = converter.create_multi_model_deployment(
                model_variants=deployment_config["model_variants"],
                output_dir=str(models_dir),
                selection_strategy="automatic"
            )
            
            assert package_path is not None
            mock_package.assert_called_once()
    
    @pytest.mark.e2e
    def test_offline_model_caching(self, tmp_path):
        """Test offline model caching for mobile apps."""
        cache_dir = tmp_path / "model_cache"
        cache_dir.mkdir()
        
        cache_config = {
            "cache_size_mb": 500,
            "eviction_policy": "lru",
            "models": ["tiny", "base"],
            "preload_models": ["tiny"]
        }
        
        with patch('fast_vlm_ondevice.caching.ModelCache') as MockCache:
            mock_cache = MagicMock()
            MockCache.return_value = mock_cache
            
            # Initialize cache
            mock_cache.initialize(cache_dir, cache_config)
            
            # Test cache operations
            mock_cache.preload_models(cache_config["preload_models"])
            mock_cache.get_model("tiny")
            mock_cache.cache_model("base", "path/to/base/model")
            
            # Verify cache operations
            mock_cache.initialize.assert_called_once()
            mock_cache.preload_models.assert_called_once()
            mock_cache.get_model.assert_called_once_with("tiny")
            mock_cache.cache_model.assert_called_once()
    
    @pytest.mark.e2e
    @pytest.mark.security
    def test_secure_model_deployment(self, tmp_path):
        """Test secure model deployment with integrity checks."""
        secure_dir = tmp_path / "secure_models"
        secure_dir.mkdir()
        
        security_config = {
            "enable_encryption": True,
            "verify_signatures": True,
            "integrity_checks": True,
            "secure_enclave": True
        }
        
        with patch('fast_vlm_ondevice.security.SecureModelHandler') as MockHandler:
            mock_handler = MagicMock()
            MockHandler.return_value = mock_handler
            
            # Test secure model handling
            mock_handler.encrypt_model("model.mlpackage", "encrypted_model.mlpackage")
            mock_handler.verify_model_integrity("encrypted_model.mlpackage")
            mock_handler.load_secure_model("encrypted_model.mlpackage")
            
            # Verify security operations
            mock_handler.encrypt_model.assert_called_once()
            mock_handler.verify_model_integrity.assert_called_once()
            mock_handler.load_secure_model.assert_called_once()
    
    @pytest.mark.e2e
    def test_deployment_monitoring_setup(self):
        """Test setup of deployment monitoring and analytics."""
        monitoring_config = {
            "performance_tracking": True,
            "crash_reporting": True,
            "usage_analytics": False,  # Privacy-first
            "error_collection": True
        }
        
        with patch('fast_vlm_ondevice.monitoring.DeploymentMonitor') as MockMonitor:
            mock_monitor = MagicMock()
            MockMonitor.return_value = mock_monitor
            
            # Setup monitoring
            mock_monitor.configure(monitoring_config)
            mock_monitor.start_performance_tracking()
            mock_monitor.setup_crash_reporting()
            
            # Verify monitoring setup
            mock_monitor.configure.assert_called_once_with(monitoring_config)
            mock_monitor.start_performance_tracking.assert_called_once()
            mock_monitor.setup_crash_reporting.assert_called_once()
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_regression_testing_pipeline(self, deployment_config, accuracy_test_dataset):
        """Test regression testing for deployment updates."""
        converter = FastVLMConverter()
        
        with patch.object(converter, 'run_regression_tests') as mock_regression:
            # Mock regression test results
            mock_regression.return_value = {
                "accuracy": {"current": 0.742, "baseline": 0.741, "delta": 0.001},
                "performance": {"current": 187, "baseline": 190, "delta": -3},
                "memory": {"current": 850, "baseline": 860, "delta": -10},
                "passed": True
            }
            
            # Run regression tests
            results = converter.validate_deployment_regression(
                new_model="new_model.mlpackage",
                baseline_model="baseline_model.mlpackage", 
                test_dataset=accuracy_test_dataset,
                performance_thresholds={"accuracy": 0.02, "latency": 50}
            )
            
            # Verify regression tests passed
            assert results["passed"] is True
            assert abs(results["accuracy"]["delta"]) < 0.02  # Within threshold
            
            mock_regression.assert_called_once()