"""
Fast VLM On-Device Kit

Production-ready Vision-Language Models for mobile devices.
Optimized for Apple Neural Engine with <250ms inference.
"""

__version__ = "1.0.0"
__author__ = "Daniel Schmidt"

# Core functionality - always available
from .core_pipeline import FastVLMCorePipeline, InferenceConfig, InferenceResult, quick_inference, create_demo_image

# Optional imports with graceful fallbacks
try:
    from .converter import FastVLMConverter
except ImportError:
    FastVLMConverter = None

try:
    from .quantization import QuantizationConfig
except ImportError:
    QuantizationConfig = None

try:
    from .health import HealthChecker, quick_health_check
except ImportError:
    HealthChecker = None
    quick_health_check = None

try:
    from .testing import ModelTester
except ImportError:
    ModelTester = None

try:
    from .benchmarking import PerformanceBenchmark, compare_models
except ImportError:
    PerformanceBenchmark = None
    compare_models = None

try:
    from .monitoring import MetricsCollector, PerformanceProfiler, AlertManager, setup_monitoring
except ImportError:
    MetricsCollector = None
    PerformanceProfiler = None
    AlertManager = None
    setup_monitoring = None

try:
    from .security import InputValidator, SecureFileHandler, SecurityScanner, setup_security_validation
except ImportError:
    InputValidator = None
    SecureFileHandler = None
    SecurityScanner = None
    setup_security_validation = None

try:
    from .logging_config import setup_logging, get_logger
except ImportError:
    setup_logging = None
    get_logger = None

try:
    from .caching import CacheManager, ModelCache, InferenceCache, create_cache_manager
except ImportError:
    CacheManager = None
    ModelCache = None
    InferenceCache = None
    create_cache_manager = None

try:
    from .optimization import PerformanceOptimizer, OptimizationConfig, create_optimizer
except ImportError:
    PerformanceOptimizer = None
    OptimizationConfig = None
    create_optimizer = None

try:
    from .deployment import ModelServer, DeploymentConfig, create_deployment
except ImportError:
    ModelServer = None
    DeploymentConfig = None
    create_deployment = None

try:
    from .neuromorphic import NeuromorphicFastVLM, SpikeConfig, SpikingNeuron, SpikingNetwork, create_neuromorphic_config
except ImportError:
    NeuromorphicFastVLM = None
    SpikeConfig = None
    SpikingNeuron = None
    SpikingNetwork = None
    create_neuromorphic_config = None

try:
    from .research import ExperimentRunner, ExperimentConfig, ResearchExperimentType, create_research_experiment, run_comprehensive_research_suite
except ImportError:
    ExperimentRunner = None
    ExperimentConfig = None
    ResearchExperimentType = None
    create_research_experiment = None
    run_comprehensive_research_suite = None

try:
    from .model_manager import ModelManager, ModelMetadata, ModelFormat, DeploymentTarget, create_advanced_model_manager
except ImportError:
    ModelManager = None
    ModelMetadata = None
    ModelFormat = None
    DeploymentTarget = None
    create_advanced_model_manager = None

try:
    from .autonomous_intelligence import AutonomousIntelligenceEngine, PatternRecognitionEngine, AutonomousDecisionEngine, create_autonomous_intelligence
except ImportError:
    AutonomousIntelligenceEngine = None
    PatternRecognitionEngine = None
    AutonomousDecisionEngine = None
    create_autonomous_intelligence = None

try:
    from .quantum_optimization import QuantumOptimizationEngine, QuantumAnnealer, VariationalQuantumOptimizer, create_quantum_optimizer
except ImportError:
    QuantumOptimizationEngine = None
    QuantumAnnealer = None
    VariationalQuantumOptimizer = None
    create_quantum_optimizer = None

try:
    from .edge_computing_orchestrator import EdgeComputingOrchestrator, IntelligentLoadBalancer, EdgeAutoScaler, create_edge_orchestrator
except ImportError:
    EdgeComputingOrchestrator = None
    IntelligentLoadBalancer = None
    EdgeAutoScaler = None
    create_edge_orchestrator = None

try:
    from .advanced_security_framework import AdvancedSecurityFramework, CryptographicManager, ThreatDetectionEngine, create_security_framework
except ImportError:
    AdvancedSecurityFramework = None
    CryptographicManager = None
    ThreatDetectionEngine = None
    create_security_framework = None

try:
    from .production_reliability_engine import ProductionReliabilityEngine, CircuitBreaker, Bulkhead, SelfHealingManager, create_reliability_engine
except ImportError:
    ProductionReliabilityEngine = None
    CircuitBreaker = None
    Bulkhead = None
    SelfHealingManager = None
    create_reliability_engine = None

try:
    from .hyper_performance_engine import HyperPerformanceEngine, HyperCache, JITCompiler, GPUAccelerator, create_hyper_performance_engine
except ImportError:
    HyperPerformanceEngine = None
    HyperCache = None
    JITCompiler = None
    GPUAccelerator = None
    create_hyper_performance_engine = None

__all__ = [
    # Core functionality - always available
    "FastVLMCorePipeline",
    "InferenceConfig", 
    "InferenceResult",
    "quick_inference",
    "create_demo_image",
    
    # Optional functionality
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
    "create_advanced_model_manager",
    
    # Autonomous Intelligence
    "AutonomousIntelligenceEngine",
    "PatternRecognitionEngine", 
    "AutonomousDecisionEngine",
    "create_autonomous_intelligence",
    
    # Quantum Optimization
    "QuantumOptimizationEngine",
    "QuantumAnnealer",
    "VariationalQuantumOptimizer",
    "create_quantum_optimizer",
    
    # Edge Computing
    "EdgeComputingOrchestrator",
    "IntelligentLoadBalancer",
    "EdgeAutoScaler", 
    "create_edge_orchestrator",
    
    # Advanced Security
    "AdvancedSecurityFramework",
    "CryptographicManager",
    "ThreatDetectionEngine",
    "create_security_framework",
    
    # Production Reliability
    "ProductionReliabilityEngine",
    "CircuitBreaker",
    "Bulkhead",
    "SelfHealingManager",
    "create_reliability_engine",
    
    # Hyper Performance
    "HyperPerformanceEngine",
    "HyperCache",
    "JITCompiler",
    "GPUAccelerator",
    "create_hyper_performance_engine"
]