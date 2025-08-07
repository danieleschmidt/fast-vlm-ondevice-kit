"""
Fast VLM On-Device Kit

Production-ready Vision-Language Models for mobile devices.
Optimized for Apple Neural Engine with <250ms inference.
"""

__version__ = "1.0.0"
__author__ = "Daniel Schmidt"

from .converter import FastVLMConverter
from .quantization import QuantizationConfig
from .health import HealthChecker, quick_health_check
from .testing import ModelTester
from .benchmarking import PerformanceBenchmark, compare_models
from .monitoring import MetricsCollector, PerformanceProfiler, AlertManager, setup_monitoring
from .security import InputValidator, SecureFileHandler, SecurityScanner, setup_security_validation
from .logging_config import setup_logging, get_logger
from .caching import CacheManager, ModelCache, InferenceCache, create_cache_manager
from .optimization import PerformanceOptimizer, OptimizationConfig, create_optimizer
from .deployment import ModelServer, DeploymentConfig, create_deployment
from .neuromorphic import NeuromorphicFastVLM, SpikeConfig, SpikingNeuron, SpikingNetwork, create_neuromorphic_config
from .research import ExperimentRunner, ExperimentConfig, ResearchExperimentType, create_research_experiment, run_comprehensive_research_suite
from .model_manager import ModelManager, ModelMetadata, ModelFormat, DeploymentTarget, create_advanced_model_manager

__all__ = [
    # Core functionality
    "FastVLMConverter", 
    "QuantizationConfig",
    
    # Health and testing
    "HealthChecker",
    "quick_health_check", 
    "ModelTester",
    "PerformanceBenchmark",
    "compare_models",
    
    # Monitoring and observability
    "MetricsCollector",
    "PerformanceProfiler", 
    "AlertManager",
    "setup_monitoring",
    
    # Security
    "InputValidator",
    "SecureFileHandler",
    "SecurityScanner",
    "setup_security_validation",
    
    # Logging
    "setup_logging",
    "get_logger",
    
    # Caching
    "CacheManager",
    "ModelCache",
    "InferenceCache", 
    "create_cache_manager",
    
    # Optimization
    "PerformanceOptimizer",
    "OptimizationConfig",
    "create_optimizer",
    
    # Deployment
    "ModelServer",
    "DeploymentConfig",
    "create_deployment",
    
    # Neuromorphic computing
    "NeuromorphicFastVLM",
    "SpikeConfig", 
    "SpikingNeuron",
    "SpikingNetwork",
    "create_neuromorphic_config",
    
    # Research framework
    "ExperimentRunner",
    "ExperimentConfig",
    "ResearchExperimentType",
    "create_research_experiment",
    "run_comprehensive_research_suite",
    
    # Advanced model management
    "ModelManager",
    "ModelMetadata",
    "ModelFormat",
    "DeploymentTarget",
    "create_advanced_model_manager"
]