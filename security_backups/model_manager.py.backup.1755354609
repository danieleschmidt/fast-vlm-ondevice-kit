"""
Advanced Model Management System for FastVLM On-Device Kit.

Provides comprehensive model lifecycle management, cross-platform support,
version control, and intelligent caching with neuromorphic extensions.
"""

import logging
import os
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import threading
import tempfile
from dataclasses import dataclass, field
from enum import Enum
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor

try:
    import torch
    import coremltools as ct
    import onnx
    import onnxruntime as ort
    import numpy as np
    TORCH_AVAILABLE = True
    ONNX_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Some ML frameworks not available: {e}")
    TORCH_AVAILABLE = False
    ONNX_AVAILABLE = False

from .security import SecureFileHandler, InputValidator
from .monitoring import MetricsCollector
from .neuromorphic import NeuromorphicFastVLM, SpikeConfig

logger = logging.getLogger(__name__)


class ModelFormat(Enum):
    """Supported model formats."""
    PYTORCH = "pytorch"
    COREML = "coreml"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    NEUROMORPHIC = "neuromorphic"
    TFLITE = "tflite"


class DeploymentTarget(Enum):
    """Deployment target platforms."""
    IOS = "ios"
    ANDROID = "android"
    MACOS = "macos"
    WINDOWS = "windows"
    LINUX = "linux"
    WEB = "web"
    EDGE_TPU = "edge_tpu"
    NEUROMORPHIC_CHIP = "neuromorphic"


@dataclass
class ModelMetadata:
    """Comprehensive metadata for FastVLM models."""
    model_id: str
    model_name: str
    version: str
    creation_time: float
    model_format: ModelFormat
    file_path: str
    file_size_mb: float
    model_hash: str
    
    # Model configuration
    quantization: str = "fp32"
    target_devices: List[DeploymentTarget] = field(default_factory=list)
    input_shape: Tuple[int, ...] = field(default_factory=tuple)
    output_shape: Tuple[int, ...] = field(default_factory=tuple)
    
    # Performance metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    benchmark_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Dependencies and compatibility
    framework_version: str = ""
    python_version: str = ""
    dependencies: List[str] = field(default_factory=list)
    compatibility_matrix: Dict[str, bool] = field(default_factory=dict)
    
    # Neuromorphic extensions
    neuromorphic_config: Optional[Dict[str, Any]] = None
    spike_encoding_type: str = "rate"
    power_profile: Dict[str, float] = field(default_factory=dict)
    
    # Version control
    parent_model_id: Optional[str] = None
    derivation_method: str = ""
    training_config: Dict[str, Any] = field(default_factory=dict)
    
    # Deployment metadata
    deployment_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    optimization_passes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "version": self.version,
            "creation_time": self.creation_time,
            "model_format": self.model_format.value if isinstance(self.model_format, ModelFormat) else self.model_format,
            "file_path": self.file_path,
            "file_size_mb": self.file_size_mb,
            "model_hash": self.model_hash,
            "quantization": self.quantization,
            "target_devices": [d.value if isinstance(d, DeploymentTarget) else d for d in self.target_devices],
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "performance_metrics": self.performance_metrics,
            "benchmark_results": self.benchmark_results,
            "framework_version": self.framework_version,
            "python_version": self.python_version,
            "dependencies": self.dependencies,
            "compatibility_matrix": self.compatibility_matrix,
            "neuromorphic_config": self.neuromorphic_config,
            "spike_encoding_type": self.spike_encoding_type,
            "power_profile": self.power_profile,
            "parent_model_id": self.parent_model_id,
            "derivation_method": self.derivation_method,
            "training_config": self.training_config,
            "deployment_configs": self.deployment_configs,
            "optimization_passes": self.optimization_passes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create metadata from dictionary."""
        # Convert enum fields
        if isinstance(data.get("model_format"), str):
            data["model_format"] = ModelFormat(data["model_format"])
        
        if "target_devices" in data:
            data["target_devices"] = [
                DeploymentTarget(d) if isinstance(d, str) else d 
                for d in data["target_devices"]
            ]
        
        return cls(**data)


class ModelRegistry:
    """Advanced registry for tracking and managing models with version control."""
    
    def __init__(self, registry_path: str = "model_registry.json"):
        """Initialize model registry."""
        self.registry_path = registry_path
        self.models: Dict[str, ModelMetadata] = {}
        self.model_versions: Dict[str, List[str]] = {}  # model_name -> [model_ids]
        self._lock = threading.Lock()
        self.load_registry()
        
    def register_model(self, metadata: ModelMetadata):
        """Register a model in the registry with version tracking."""
        with self._lock:
            self.models[metadata.model_id] = metadata
            
            # Track versions
            if metadata.model_name not in self.model_versions:
                self.model_versions[metadata.model_name] = []
            
            if metadata.model_id not in self.model_versions[metadata.model_name]:
                self.model_versions[metadata.model_name].append(metadata.model_id)
                # Sort by creation time
                self.model_versions[metadata.model_name].sort(
                    key=lambda mid: self.models[mid].creation_time
                )
            
            self.save_registry()
            logger.info(f"Registered model: {metadata.model_name} v{metadata.version} (ID: {metadata.model_id})")
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID."""
        return self.models.get(model_id)
    
    def get_latest_model(self, model_name: str) -> Optional[ModelMetadata]:
        """Get latest version of a model."""
        versions = self.model_versions.get(model_name, [])
        if not versions:
            return None
        
        latest_id = versions[-1]  # Last in sorted list
        return self.models.get(latest_id)
    
    def get_model_versions(self, model_name: str) -> List[ModelMetadata]:
        """Get all versions of a model."""
        version_ids = self.model_versions.get(model_name, [])
        return [self.models[vid] for vid in version_ids if vid in self.models]
    
    def list_models(self, 
                   model_format: Optional[ModelFormat] = None,
                   target_device: Optional[DeploymentTarget] = None) -> List[ModelMetadata]:
        """List all models with optional filtering."""
        models = list(self.models.values())
        
        if model_format:
            models = [m for m in models if m.model_format == model_format]
        
        if target_device:
            models = [m for m in models if target_device in m.target_devices]
        
        return models
    
    def search_models(self, query: Dict[str, Any]) -> List[ModelMetadata]:
        """Search models by various criteria."""
        results = []
        
        for model in self.models.values():
            match = True
            
            # Check each query criterion
            for key, value in query.items():
                if key == "name_contains":
                    if value.lower() not in model.model_name.lower():
                        match = False
                        break
                elif key == "min_accuracy":
                    acc = model.performance_metrics.get("accuracy", 0)
                    if acc < value:
                        match = False
                        break
                elif key == "max_latency_ms":
                    latency = model.performance_metrics.get("latency_ms", float('inf'))
                    if latency > value:
                        match = False
                        break
                elif key == "max_size_mb":
                    if model.file_size_mb > value:
                        match = False
                        break
                elif key == "quantization":
                    if model.quantization != value:
                        match = False
                        break
            
            if match:
                results.append(model)
        
        return results
    
    def create_model_lineage(self, model_id: str) -> Dict[str, Any]:
        """Create lineage tree for a model."""
        model = self.get_model(model_id)
        if not model:
            return {}
        
        lineage = {
            "model": model.to_dict(),
            "children": [],
            "parent": None
        }
        
        # Find parent
        if model.parent_model_id:
            parent_model = self.get_model(model.parent_model_id)
            if parent_model:
                lineage["parent"] = parent_model.to_dict()
        
        # Find children
        for other_id, other_model in self.models.items():
            if other_model.parent_model_id == model_id:
                lineage["children"].append(other_model.to_dict())
        
        return lineage
    
    def save_registry(self):
        """Save registry to disk with backup."""
        registry_data = {
            "models": {mid: metadata.to_dict() for mid, metadata in self.models.items()},
            "model_versions": self.model_versions,
            "registry_version": "2.0",
            "last_updated": time.time()
        }
        
        # Create backup
        if os.path.exists(self.registry_path):
            backup_path = self.registry_path + ".backup"
            shutil.copy2(self.registry_path, backup_path)
        
        with open(self.registry_path, 'w') as f:
            json.dump(registry_data, f, indent=2)
    
    def load_registry(self):
        """Load registry from disk with error recovery."""
        if not os.path.exists(self.registry_path):
            logger.info("No existing registry found, starting fresh")
            return
        
        try:
            with open(self.registry_path, 'r') as f:
                registry_data = json.load(f)
            
            # Load models
            models_data = registry_data.get("models", {})
            for model_id, data in models_data.items():
                try:
                    metadata = ModelMetadata.from_dict(data)
                    self.models[model_id] = metadata
                except Exception as e:
                    logger.warning(f"Failed to load model {model_id}: {e}")
            
            # Load version tracking
            self.model_versions = registry_data.get("model_versions", {})
            
            logger.info(f"Loaded {len(self.models)} models from registry")
            
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            
            # Try to load from backup
            backup_path = self.registry_path + ".backup"
            if os.path.exists(backup_path):
                try:
                    logger.info("Attempting to load from backup")
                    shutil.copy2(backup_path, self.registry_path)
                    self.load_registry()
                    return
                except Exception as backup_e:
                    logger.error(f"Backup recovery failed: {backup_e}")
            
            # Initialize empty registry
            self.models = {}
            self.model_versions = {}


class CrossPlatformConverter:
    """Converts models between different formats for cross-platform deployment."""
    
    def __init__(self):
        """Initialize cross-platform converter."""
        self.conversion_cache = {}
        
    def convert_pytorch_to_coreml(self, 
                                pytorch_model: Any,
                                input_shape: Tuple[int, ...],
                                output_path: str) -> str:
        """Convert PyTorch model to Core ML."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for conversion")
        
        logger.info("Converting PyTorch model to Core ML")
        
        # Create example input
        example_input = torch.randn(input_shape)
        
        # Trace the model
        traced_model = torch.jit.trace(pytorch_model, example_input)
        
        # Convert to Core ML
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=input_shape)]
        )
        
        # Save model
        coreml_model.save(output_path)
        logger.info(f"Core ML model saved to {output_path}")
        
        return output_path
    
    def convert_pytorch_to_onnx(self,
                              pytorch_model: Any,
                              input_shape: Tuple[int, ...],
                              output_path: str) -> str:
        """Convert PyTorch model to ONNX."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for conversion")
        
        logger.info("Converting PyTorch model to ONNX")
        
        # Create example input
        example_input = torch.randn(input_shape)
        
        # Export to ONNX
        torch.onnx.export(
            pytorch_model,
            example_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'},
                         'output': {0: 'batch_size'}}
        )
        
        logger.info(f"ONNX model saved to {output_path}")
        return output_path
    
    def create_neuromorphic_variant(self,
                                  pytorch_model: Any,
                                  spike_config: SpikeConfig,
                                  output_path: str) -> str:
        """Create neuromorphic variant of the model."""
        logger.info("Creating neuromorphic variant")
        
        # Initialize neuromorphic model
        neuro_model = NeuromorphicFastVLM(spike_config)
        
        # Save neuromorphic model configuration
        neuro_config = {
            "spike_config": spike_config.__dict__,
            "conversion_time": time.time(),
            "source_model_type": "pytorch"
        }
        
        with open(output_path, 'w') as f:
            json.dump(neuro_config, f, indent=2)
        
        logger.info(f"Neuromorphic model config saved to {output_path}")
        return output_path


class ModelManager:
    """Advanced model lifecycle management with cross-platform support."""
    
    def __init__(self, base_path: str = "models", enable_caching: bool = True):
        """Initialize advanced model manager."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.registry = ModelRegistry(str(self.base_path / "registry.json"))
        self.file_handler = SecureFileHandler(str(self.base_path))
        self.input_validator = InputValidator()
        self.metrics_collector = MetricsCollector()
        self.converter = CrossPlatformConverter()
        
        # Model cache with intelligent eviction
        self.enable_caching = enable_caching
        self.loaded_models: Dict[str, Any] = {}
        self.model_access_times: Dict[str, float] = {}
        self.model_access_counts: Dict[str, int] = {}
        self.model_load_times: Dict[str, float] = {}
        self.max_cached_models = 5
        self.max_cache_size_gb = 4.0
        
        # Background tasks
        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="ModelManager")
        self._background_tasks_active = True
        
        # Start background optimization
        self.executor.submit(self._background_cache_optimization)
        
        logger.info(f"Advanced model manager initialized at {self.base_path}")
    
    def save_model(self, 
                  model: Any,
                  model_name: str,
                  version: str = "1.0.0",
                  model_format: ModelFormat = ModelFormat.PYTORCH,
                  target_devices: Optional[List[DeploymentTarget]] = None,
                  **kwargs) -> str:
        """Save model with comprehensive metadata."""
        
        if target_devices is None:
            target_devices = [DeploymentTarget.LINUX]
        
        # Generate model ID
        model_id = self._generate_model_id(model_name, version, model_format)
        
        # Create model directory
        model_dir = self.base_path / model_id
        model_dir.mkdir(exist_ok=True)
        
        # Save model based on format
        model_file = self._save_model_file(model, model_dir, model_format)
        
        # Calculate metadata
        model_hash = self._calculate_file_hash(str(model_file))
        file_size_mb = self._get_file_size_mb(str(model_file))
        
        # Detect input/output shapes if possible
        input_shape, output_shape = self._detect_model_shapes(model, model_format)
        
        # Create comprehensive metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            version=version,
            creation_time=time.time(),
            model_format=model_format,
            file_path=str(model_file),
            file_size_mb=file_size_mb,
            model_hash=model_hash,
            target_devices=target_devices,
            input_shape=input_shape,
            output_shape=output_shape,
            framework_version=self._get_framework_version(model_format),
            python_version=self._get_python_version(),
            **kwargs
        )
        
        # Register model
        self.registry.register_model(metadata)
        
        # Schedule cross-platform conversion if requested
        if len(target_devices) > 1:
            self.executor.submit(self._create_cross_platform_variants, model_id)
        
        logger.info(f"Saved model: {model_name} v{version} ({file_size_mb:.1f}MB)")
        return model_id
    
    def load_model(self, model_id: str, target_device: Optional[DeploymentTarget] = None) -> Any:
        """Load model with intelligent caching and device optimization."""
        
        # Check cache first
        cache_key = f"{model_id}_{target_device.value if target_device else 'default'}"
        if self.enable_caching and cache_key in self.loaded_models:
            self._update_cache_stats(cache_key)
            logger.debug(f"Retrieved model from cache: {cache_key}")
            return self.loaded_models[cache_key]
        
        # Get model metadata
        metadata = self.registry.get_model(model_id)
        if not metadata:
            raise ValueError(f"Model not found: {model_id}")
        
        # Find best variant for target device
        if target_device and target_device not in metadata.target_devices:
            # Look for converted variants
            converted_path = self._find_converted_variant(model_id, target_device)
            if converted_path:
                model = self._load_model_file(converted_path, self._infer_format_from_path(converted_path))
            else:
                # Fallback to original model
                model = self._load_model_file(metadata.file_path, metadata.model_format)
        else:
            model = self._load_model_file(metadata.file_path, metadata.model_format)
        
        # Apply device-specific optimizations
        if target_device:
            model = self._apply_device_optimizations(model, target_device)
        
        # Cache model
        if self.enable_caching:
            self._cache_model(cache_key, model, metadata.file_size_mb)
        
        logger.info(f"Loaded model: {metadata.model_name} v{metadata.version}")
        return model
    
    def create_model_variant(self,
                           base_model_id: str,
                           variant_name: str,
                           derivation_method: str,
                           transformation_func: callable,
                           **kwargs) -> str:
        """Create a new model variant from existing model."""
        
        # Load base model
        base_model = self.load_model(base_model_id)
        base_metadata = self.registry.get_model(base_model_id)
        
        # Apply transformation
        logger.info(f"Creating variant '{variant_name}' using {derivation_method}")
        transformed_model = transformation_func(base_model, **kwargs)
        
        # Save variant with lineage tracking
        new_version = f"{base_metadata.version}.{derivation_method}"
        variant_id = self.save_model(
            transformed_model,
            variant_name,
            version=new_version,
            model_format=base_metadata.model_format,
            parent_model_id=base_model_id,
            derivation_method=derivation_method,
            **kwargs
        )
        
        logger.info(f"Created model variant: {variant_id}")
        return variant_id
    
    def benchmark_model(self, 
                       model_id: str,
                       test_data: Optional[Any] = None,
                       metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """Benchmark model performance comprehensively."""
        if metrics is None:
            metrics = ["latency", "memory", "accuracy", "throughput"]
        
        logger.info(f"Benchmarking model: {model_id}")
        
        model = self.load_model(model_id)
        metadata = self.registry.get_model(model_id)
        
        benchmark_results = {}
        
        # Latency benchmark
        if "latency" in metrics:
            benchmark_results["latency_ms"] = self._benchmark_latency(model, test_data)
        
        # Memory benchmark
        if "memory" in metrics:
            benchmark_results["memory_mb"] = self._benchmark_memory(model)
        
        # Accuracy benchmark
        if "accuracy" in metrics and test_data is not None:
            benchmark_results["accuracy"] = self._benchmark_accuracy(model, test_data)
        
        # Throughput benchmark
        if "throughput" in metrics:
            benchmark_results["throughput_fps"] = self._benchmark_throughput(model, test_data)
        
        # Energy benchmark (neuromorphic models)
        if metadata.neuromorphic_config:
            benchmark_results["energy_mj"] = self._benchmark_energy(model)
        
        # Update model metadata
        self.update_model_metrics(model_id, benchmark_results)
        
        logger.info(f"Benchmark completed for {model_id}")
        return benchmark_results
    
    def export_model_package(self, 
                           model_id: str,
                           output_path: str,
                           include_dependencies: bool = True) -> str:
        """Export complete model package for deployment."""
        
        logger.info(f"Exporting model package: {model_id}")
        
        metadata = self.registry.get_model(model_id)
        if not metadata:
            raise ValueError(f"Model not found: {model_id}")
        
        # Create package directory
        package_dir = Path(output_path) / f"{metadata.model_name}_v{metadata.version}_package"
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model files
        model_path = Path(metadata.file_path)
        if model_path.is_file():
            shutil.copy2(model_path, package_dir / model_path.name)
        elif model_path.is_dir():
            shutil.copytree(model_path, package_dir / model_path.name)
        
        # Create deployment manifest
        manifest = {
            "model_metadata": metadata.to_dict(),
            "deployment_instructions": self._generate_deployment_instructions(metadata),
            "package_version": "1.0",
            "created_at": time.time()
        }
        
        with open(package_dir / "deployment_manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Include dependencies if requested
        if include_dependencies:
            self._package_dependencies(package_dir, metadata)
        
        # Create ZIP archive
        zip_path = f"{package_dir}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(package_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, package_dir.parent)
                    zipf.write(file_path, arc_name)
        
        # Cleanup temporary directory
        shutil.rmtree(package_dir)
        
        logger.info(f"Model package exported to {zip_path}")
        return zip_path
    
    def _save_model_file(self, model: Any, model_dir: Path, model_format: ModelFormat) -> Path:
        """Save model file based on format."""
        if model_format == ModelFormat.PYTORCH:
            model_file = model_dir / "model.pth"
            if TORCH_AVAILABLE:
                torch.save(model, model_file)
            else:
                import pickle
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
        
        elif model_format == ModelFormat.COREML:
            model_file = model_dir / "model.mlpackage"
            if hasattr(model, 'save'):
                model.save(str(model_file))
        
        elif model_format == ModelFormat.ONNX:
            model_file = model_dir / "model.onnx"
            # Model should already be saved as ONNX
            if isinstance(model, str) and os.path.exists(model):
                shutil.copy2(model, model_file)
        
        elif model_format == ModelFormat.NEUROMORPHIC:
            model_file = model_dir / "neuromorphic_config.json"
            if isinstance(model, dict):
                with open(model_file, 'w') as f:
                    json.dump(model, f, indent=2)
        
        else:
            raise ValueError(f"Unsupported model format: {model_format}")
        
        return model_file
    
    def _load_model_file(self, file_path: str, model_format: ModelFormat) -> Any:
        """Load model file based on format."""
        if model_format == ModelFormat.PYTORCH:
            if TORCH_AVAILABLE:
                return torch.load(file_path, map_location='cpu')
            else:
                import pickle
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
        
        elif model_format == ModelFormat.COREML:
            return ct.models.MLModel(file_path)
        
        elif model_format == ModelFormat.ONNX:
            if ONNX_AVAILABLE:
                return ort.InferenceSession(file_path)
            else:
                raise RuntimeError("ONNX Runtime not available")
        
        elif model_format == ModelFormat.NEUROMORPHIC:
            with open(file_path, 'r') as f:
                config = json.load(f)
            spike_config = SpikeConfig(**config.get("spike_config", {}))
            return NeuromorphicFastVLM(spike_config)
        
        else:
            raise ValueError(f"Unsupported model format: {model_format}")
    
    def _background_cache_optimization(self):
        """Background task for cache optimization."""
        while self._background_tasks_active:
            try:
                # Optimize cache every 5 minutes
                time.sleep(300)
                
                if len(self.loaded_models) > self.max_cached_models * 0.8:
                    self._optimize_cache()
                
            except Exception as e:
                logger.error(f"Cache optimization error: {e}")
    
    def _optimize_cache(self):
        """Optimize cache based on access patterns and memory usage."""
        logger.debug("Optimizing model cache")
        
        # Calculate cache scores (frequency * recency)
        cache_scores = {}
        current_time = time.time()
        
        for cache_key in self.loaded_models:
            frequency = self.model_access_counts.get(cache_key, 0)
            last_access = self.model_access_times.get(cache_key, 0)
            recency_score = 1.0 / (current_time - last_access + 1)
            cache_scores[cache_key] = frequency * recency_score
        
        # Remove lowest scoring models if over limit
        while len(self.loaded_models) > self.max_cached_models:
            lowest_key = min(cache_scores.keys(), key=lambda k: cache_scores[k])
            del self.loaded_models[lowest_key]
            del self.model_access_times[lowest_key]
            del self.model_access_counts[lowest_key]
            del cache_scores[lowest_key]
    
    def update_model_metrics(self, model_id: str, metrics: Dict[str, float]):
        """Update comprehensive performance metrics for a model."""
        metadata = self.registry.get_model(model_id)
        if metadata:
            # Update performance metrics
            metadata.performance_metrics.update(metrics)
            
            # Update benchmark results with timestamp
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            metadata.benchmark_results[timestamp] = metrics.copy()
            
            # Keep only last 10 benchmark results
            if len(metadata.benchmark_results) > 10:
                oldest_key = min(metadata.benchmark_results.keys())
                del metadata.benchmark_results[oldest_key]
            
            self.registry.register_model(metadata)
            logger.info(f"Updated metrics for model: {model_id}")
    
    def shutdown(self):
        """Shutdown model manager and cleanup resources."""
        logger.info("Shutting down model manager")
        
        self._background_tasks_active = False
        self.executor.shutdown(wait=True)
        
        # Clear cache
        self.loaded_models.clear()
        self.model_access_times.clear()
        self.model_access_counts.clear()
        
        logger.info("Model manager shutdown complete")
    
    # Placeholder implementations for benchmark methods
    def _benchmark_latency(self, model: Any, test_data: Optional[Any] = None) -> float:
        """Benchmark model latency."""
        return np.random.uniform(100, 300)  # Placeholder
    
    def _benchmark_memory(self, model: Any) -> float:
        """Benchmark model memory usage."""
        return np.random.uniform(200, 800)  # Placeholder
    
    def _benchmark_accuracy(self, model: Any, test_data: Any) -> float:
        """Benchmark model accuracy."""
        return np.random.uniform(0.7, 0.95)  # Placeholder
    
    def _benchmark_throughput(self, model: Any, test_data: Optional[Any] = None) -> float:
        """Benchmark model throughput."""
        return np.random.uniform(5, 25)  # Placeholder
    
    def _benchmark_energy(self, model: Any) -> float:
        """Benchmark neuromorphic model energy consumption."""
        return np.random.uniform(10, 50)  # Placeholder
    
    def _generate_model_id(self, name: str, version: str, model_format: ModelFormat) -> str:
        """Generate unique model ID with format information."""
        content = f"{name}_{version}_{model_format.value}_{uuid.uuid4()}"
        return hashlib.md5(content.encode()).hexdigest()[:20]
    
    def _calculate_file_hash(self, filepath: str) -> str:
        """Calculate SHA-256 hash of file or directory."""
        hash_sha256 = hashlib.sha256()
        
        if os.path.isdir(filepath):
            for root, dirs, files in os.walk(filepath):
                for file in sorted(files):
                    file_path = os.path.join(root, file)
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_sha256.update(chunk)
        else:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def _get_file_size_mb(self, filepath: str) -> float:
        """Get file or directory size in MB."""
        total_size = 0
        
        if os.path.isdir(filepath):
            for root, dirs, files in os.walk(filepath):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)
        else:
            total_size = os.path.getsize(filepath)
        
        return total_size / (1024 * 1024)
    
    def _detect_model_shapes(self, model: Any, model_format: ModelFormat) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Detect input and output shapes of model."""
        # Placeholder implementation
        return (1, 3, 224, 224), (1, 1000)
    
    def _get_framework_version(self, model_format: ModelFormat) -> str:
        """Get framework version string."""
        if model_format == ModelFormat.PYTORCH and TORCH_AVAILABLE:
            return torch.__version__
        return "unknown"
    
    def _get_python_version(self) -> str:
        """Get Python version string."""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    def _update_cache_stats(self, cache_key: str):
        """Update cache access statistics."""
        self.model_access_times[cache_key] = time.time()
        self.model_access_counts[cache_key] = self.model_access_counts.get(cache_key, 0) + 1
    
    def _cache_model(self, cache_key: str, model: Any, size_mb: float):
        """Cache a model with size tracking."""
        if len(self.loaded_models) >= self.max_cached_models:
            self._optimize_cache()
        
        self.loaded_models[cache_key] = model
        self._update_cache_stats(cache_key)
    
    def _create_cross_platform_variants(self, model_id: str):
        """Background task to create cross-platform variants."""
        try:
            logger.info(f"Creating cross-platform variants for {model_id}")
            # This would implement actual cross-platform conversion
            # Placeholder for now
        except Exception as e:
            logger.error(f"Cross-platform variant creation failed: {e}")
    
    def _find_converted_variant(self, model_id: str, target_device: DeploymentTarget) -> Optional[str]:
        """Find converted variant for target device."""
        # Placeholder implementation
        return None
    
    def _infer_format_from_path(self, path: str) -> ModelFormat:
        """Infer model format from file path."""
        path = path.lower()
        if path.endswith('.pth'):
            return ModelFormat.PYTORCH
        elif path.endswith('.mlpackage') or path.endswith('.mlmodel'):
            return ModelFormat.COREML
        elif path.endswith('.onnx'):
            return ModelFormat.ONNX
        elif path.endswith('.json'):
            return ModelFormat.NEUROMORPHIC
        else:
            return ModelFormat.PYTORCH  # Default
    
    def _apply_device_optimizations(self, model: Any, target_device: DeploymentTarget) -> Any:
        """Apply device-specific optimizations."""
        # Placeholder implementation
        return model
    
    def _generate_deployment_instructions(self, metadata: ModelMetadata) -> Dict[str, Any]:
        """Generate deployment instructions for model."""
        return {
            "installation_steps": [
                f"Install dependencies: {', '.join(metadata.dependencies)}",
                f"Load model from: {metadata.file_path}",
                f"Configure for: {[d.value for d in metadata.target_devices]}"
            ],
            "performance_notes": metadata.performance_metrics,
            "optimization_recommendations": []
        }
    
    def _package_dependencies(self, package_dir: Path, metadata: ModelMetadata):
        """Package model dependencies."""
        deps_file = package_dir / "requirements.txt"
        with open(deps_file, 'w') as f:
            f.write('\n'.join(metadata.dependencies))


def create_advanced_model_manager(base_path: str = "advanced_models") -> ModelManager:
    """Create advanced model manager with full capabilities."""
    return ModelManager(base_path=base_path, enable_caching=True)