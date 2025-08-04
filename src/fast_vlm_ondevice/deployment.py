"""
Production deployment utilities for FastVLM models.

Provides model serving, auto-scaling, and deployment management.
"""

import logging
import time
import json
import threading
from typing import Dict, Any, Optional, List, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import uuid
from concurrent.futures import ThreadPoolExecutor
import queue

try:
    import torch
    import coremltools as ct
    import numpy as np
    from PIL import Image
    DEPLOYMENT_DEPS = True
except ImportError:
    DEPLOYMENT_DEPS = False

from .monitoring import MetricsCollector, PerformanceProfiler, AlertManager
from .caching import CacheManager
from .optimization import PerformanceOptimizer, OptimizationConfig
from .security import InputValidator

logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""
    
    # Server configuration
    max_concurrent_requests: int = 10
    request_timeout_seconds: float = 30.0
    health_check_interval: float = 60.0
    
    # Auto-scaling configuration
    enable_auto_scaling: bool = True
    min_instances: int = 1
    max_instances: int = 5
    scale_up_threshold: float = 0.8  # CPU/memory threshold
    scale_down_threshold: float = 0.2
    scale_decision_window: float = 300.0  # 5 minutes
    
    # Model configuration
    model_path: str = ""
    model_type: str = "coreml"  # "pytorch" or "coreml"
    warm_up_requests: int = 3
    
    # Resource limits
    max_memory_mb: float = 2048.0
    max_cpu_percent: float = 80.0
    
    # Deployment features
    enable_caching: bool = True
    enable_monitoring: bool = True
    enable_load_balancing: bool = True


@dataclass
class InferenceRequest:
    """Container for inference requests."""
    request_id: str
    image: Any
    question: str
    model_version: str
    timestamp: float
    priority: int = 0  # Higher number = higher priority
    
    def __lt__(self, other):
        """For priority queue ordering."""
        return self.priority > other.priority


@dataclass
class InferenceResponse:
    """Container for inference responses."""
    request_id: str
    answer: str
    confidence: float
    latency_ms: float
    model_version: str
    timestamp: float
    error: Optional[str] = None


class ModelInstance:
    """Single model instance for serving."""
    
    def __init__(
        self,
        instance_id: str,
        model_path: str,
        model_type: str,
        config: DeploymentConfig
    ):
        """Initialize model instance.
        
        Args:
            instance_id: Unique instance identifier
            model_path: Path to model file
            model_type: Type of model ("pytorch" or "coreml")
            config: Deployment configuration
        """
        self.instance_id = instance_id
        self.model_path = model_path
        self.model_type = model_type
        self.config = config
        
        self.model = None
        self.is_loaded = False
        self.is_healthy = True
        self.load_time = 0.0
        self.request_count = 0
        self.error_count = 0
        self.last_request_time = 0.0
        
        # Initialize components
        self.input_validator = InputValidator()
        self.optimizer = PerformanceOptimizer(OptimizationConfig(
            max_memory_mb=config.max_memory_mb,
            optimization_level="balanced"
        ))
        self.metrics_collector = MetricsCollector()
        self.profiler = PerformanceProfiler(self.metrics_collector, instance_id)
        
        self._lock = threading.Lock()
        
    def load_model(self) -> bool:
        """Load model into memory."""
        with self._lock:
            if self.is_loaded:
                return True
            
            try:
                start_time = time.time()
                logger.info(f"Loading model instance {self.instance_id}")
                
                if self.model_type == "pytorch":
                    self.model = torch.load(self.model_path, map_location='cpu')
                    if hasattr(self.model, 'eval'):
                        self.model.eval()
                    self.model = self.optimizer.optimize_model(self.model, "pytorch")
                    
                elif self.model_type == "coreml":
                    self.model = ct.models.MLModel(self.model_path)
                    self.model = self.optimizer.optimize_model(self.model, "coreml")
                
                else:
                    raise ValueError(f"Unsupported model type: {self.model_type}")
                
                self.load_time = time.time() - start_time
                self.is_loaded = True
                
                # Warm up model
                self._warm_up_model()
                
                logger.info(f"Model instance {self.instance_id} loaded in {self.load_time:.2f}s")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load model instance {self.instance_id}: {e}")
                self.is_healthy = False
                return False
    
    def _warm_up_model(self):
        """Warm up model with dummy requests."""
        logger.info(f"Warming up model instance {self.instance_id}")
        
        try:
            for i in range(self.config.warm_up_requests):
                dummy_request = InferenceRequest(
                    request_id=f"warmup-{i}",
                    image=np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8),
                    question="What is in this image?",
                    model_version="1.0",
                    timestamp=time.time()
                )
                
                # Run dummy inference
                self._run_inference(dummy_request)
                
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")
    
    def process_request(self, request: InferenceRequest) -> InferenceResponse:
        """Process inference request.
        
        Args:
            request: Inference request
            
        Returns:
            Inference response
        """
        with self.profiler.profile_inference():
            start_time = time.time()
            
            try:
                # Validate inputs
                image_validation = self.input_validator.validate_image_input(request.image)
                if not image_validation["valid"]:
                    raise ValueError(f"Invalid image: {image_validation['errors']}")
                
                text_validation = self.input_validator.validate_text_input(request.question)
                if not text_validation["valid"]:
                    raise ValueError(f"Invalid question: {text_validation['errors']}")
                
                # Run inference
                result = self._run_inference(request)
                
                # Update statistics
                self.request_count += 1
                self.last_request_time = time.time()
                
                latency_ms = (time.time() - start_time) * 1000
                
                return InferenceResponse(
                    request_id=request.request_id,
                    answer=result["answer"],
                    confidence=result.get("confidence", 0.0),
                    latency_ms=latency_ms,
                    model_version=request.model_version,
                    timestamp=time.time()
                )
                
            except Exception as e:
                self.error_count += 1
                logger.error(f"Inference failed for request {request.request_id}: {e}")
                
                return InferenceResponse(
                    request_id=request.request_id,
                    answer="",
                    confidence=0.0,
                    latency_ms=(time.time() - start_time) * 1000,
                    model_version=request.model_version,
                    timestamp=time.time(),
                    error=str(e)
                )
    
    def _run_inference(self, request: InferenceRequest) -> Dict[str, Any]:
        """Run model inference."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        if self.model_type == "pytorch":
            return self._run_pytorch_inference(request)
        elif self.model_type == "coreml":
            return self._run_coreml_inference(request)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _run_pytorch_inference(self, request: InferenceRequest) -> Dict[str, Any]:
        """Run PyTorch model inference."""
        # This would implement actual PyTorch inference
        # For demo, return placeholder
        time.sleep(0.1)  # Simulate inference time
        return {
            "answer": f"PyTorch answer for: {request.question[:50]}...",
            "confidence": 0.85
        }
    
    def _run_coreml_inference(self, request: InferenceRequest) -> Dict[str, Any]:
        """Run Core ML model inference."""
        # This would implement actual Core ML inference
        # For demo, return placeholder
        time.sleep(0.08)  # Simulate faster inference time
        return {
            "answer": f"Core ML answer for: {request.question[:50]}...",
            "confidence": 0.90
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get instance health status."""
        return {
            "instance_id": self.instance_id,
            "is_loaded": self.is_loaded,
            "is_healthy": self.is_healthy,
            "load_time": self.load_time,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "last_request_time": self.last_request_time,
            "uptime": time.time() - (self.last_request_time - self.load_time) if self.last_request_time > 0 else 0
        }
    
    def shutdown(self):
        """Shutdown model instance."""
        logger.info(f"Shutting down model instance {self.instance_id}")
        
        with self._lock:
            self.is_loaded = False
            self.is_healthy = False
            self.model = None
            
        self.optimizer.shutdown()


class LoadBalancer:
    """Load balancer for distributing requests across model instances."""
    
    def __init__(self, config: DeploymentConfig):
        """Initialize load balancer."""
        self.config = config
        self.instances: Dict[str, ModelInstance] = {}
        self.request_counts: Dict[str, int] = {}
        self._lock = threading.Lock()
        
    def add_instance(self, instance: ModelInstance):
        """Add model instance to load balancer."""
        with self._lock:
            self.instances[instance.instance_id] = instance
            self.request_counts[instance.instance_id] = 0
            
        logger.info(f"Added instance {instance.instance_id} to load balancer")
    
    def remove_instance(self, instance_id: str):
        """Remove model instance from load balancer."""
        with self._lock:
            if instance_id in self.instances:
                del self.instances[instance_id]
                del self.request_counts[instance_id]
                
        logger.info(f"Removed instance {instance_id} from load balancer")
    
    def select_instance(self) -> Optional[ModelInstance]:
        """Select best instance for request using round-robin with health checks."""
        with self._lock:
            healthy_instances = [
                (instance_id, instance)
                for instance_id, instance in self.instances.items()
                if instance.is_healthy and instance.is_loaded
            ]
            
            if not healthy_instances:
                return None
            
            # Select instance with lowest request count (weighted round-robin)
            selected_id, selected_instance = min(
                healthy_instances,
                key=lambda x: self.request_counts[x[0]]
            )
            
            self.request_counts[selected_id] += 1
            return selected_instance
    
    def get_load_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        with self._lock:
            return {
                "total_instances": len(self.instances),
                "healthy_instances": sum(1 for i in self.instances.values() if i.is_healthy),
                "request_distribution": self.request_counts.copy(),
                "instance_health": {
                    instance_id: instance.get_health_status()
                    for instance_id, instance in self.instances.items()
                }
            }


class AutoScaler:
    """Automatic scaling of model instances based on load."""
    
    def __init__(self, config: DeploymentConfig, load_balancer: LoadBalancer):
        """Initialize auto scaler."""
        self.config = config
        self.load_balancer = load_balancer
        self.scaling_decisions: List[Dict[str, Any]] = []
        self.last_scale_time = 0.0
        self._lock = threading.Lock()
        
        # Start monitoring thread
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start auto-scaling monitoring."""
        if self.monitoring or not self.config.enable_auto_scaling:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Auto-scaling monitoring started")
    
    def stop_monitoring(self):
        """Stop auto-scaling monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("Auto-scaling monitoring stopped")
    
    def _monitor_loop(self):
        """Auto-scaling monitoring loop."""
        while self.monitoring:
            try:
                # Check if scaling decision needed
                if self._should_make_scaling_decision():
                    scaling_decision = self._make_scaling_decision()
                    if scaling_decision:
                        self._execute_scaling_decision(scaling_decision)
                
                time.sleep(30.0)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Auto-scaling monitoring error: {e}")
                time.sleep(30.0)
    
    def _should_make_scaling_decision(self) -> bool:
        """Check if scaling decision should be made."""
        time_since_last = time.time() - self.last_scale_time
        return time_since_last >= self.config.scale_decision_window
    
    def _make_scaling_decision(self) -> Optional[Dict[str, Any]]:
        """Make scaling decision based on current metrics."""
        try:
            import psutil
            
            # Get current resource usage
            cpu_percent = psutil.cpu_percent(interval=1.0)
            memory_percent = psutil.virtual_memory().percent
            
            # Get load balancer stats
            load_stats = self.load_balancer.get_load_stats()
            current_instances = load_stats["healthy_instances"]
            
            # Calculate average load per instance
            total_requests = sum(load_stats["request_distribution"].values())
            avg_load = total_requests / max(current_instances, 1)
            
            # Make scaling decision
            scale_action = None
            
            if (cpu_percent > self.config.scale_up_threshold * 100 or
                memory_percent > self.config.scale_up_threshold * 100 or
                avg_load > 100):  # More than 100 requests per instance
                
                if current_instances < self.config.max_instances:
                    scale_action = "scale_up"
            
            elif (cpu_percent < self.config.scale_down_threshold * 100 and
                  memory_percent < self.config.scale_down_threshold * 100 and
                  avg_load < 50):  # Less than 50 requests per instance
                
                if current_instances > self.config.min_instances:
                    scale_action = "scale_down"
            
            if scale_action:
                decision = {
                    "action": scale_action,
                    "reason": f"CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%, Load: {avg_load:.1f}",
                    "current_instances": current_instances,
                    "timestamp": time.time()
                }
                
                logger.info(f"Scaling decision: {decision}")
                return decision
            
            return None
            
        except ImportError:
            logger.warning("psutil not available for auto-scaling")
            return None
        except Exception as e:
            logger.error(f"Scaling decision failed: {e}")
            return None
    
    def _execute_scaling_decision(self, decision: Dict[str, Any]):
        """Execute scaling decision."""
        with self._lock:
            self.scaling_decisions.append(decision)
            self.last_scale_time = time.time()
        
        # Note: Actual scaling implementation would depend on deployment environment
        # This is a placeholder for the scaling logic
        logger.info(f"Would execute scaling decision: {decision['action']}")
    
    def get_scaling_history(self) -> List[Dict[str, Any]]:
        """Get scaling decision history."""
        with self._lock:
            return self.scaling_decisions.copy()


class ModelServer:
    """Main model server for production deployment."""
    
    def __init__(self, config: DeploymentConfig):
        """Initialize model server."""
        self.config = config
        self.server_id = str(uuid.uuid4())[:8]
        
        # Initialize components
        self.load_balancer = LoadBalancer(config)
        self.auto_scaler = AutoScaler(config, self.load_balancer)
        self.request_queue = queue.PriorityQueue(maxsize=config.max_concurrent_requests * 2)
        
        # Initialize monitoring
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        
        # Initialize caching if enabled
        if config.enable_caching:
            self.cache_manager = CacheManager()
        else:
            self.cache_manager = None
        
        # Request processing
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_requests)
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        
        logger.info(f"Model server {self.server_id} initialized")
    
    def start(self) -> bool:
        """Start model server."""
        try:
            # Load initial model instance
            initial_instance = ModelInstance(
                instance_id=f"{self.server_id}-0",
                model_path=self.config.model_path,
                model_type=self.config.model_type,
                config=self.config
            )
            
            if not initial_instance.load_model():
                logger.error("Failed to load initial model instance")
                return False
            
            self.load_balancer.add_instance(initial_instance)
            
            # Start monitoring
            if self.config.enable_monitoring:
                self.auto_scaler.start_monitoring()
            
            logger.info(f"Model server {self.server_id} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start model server: {e}")
            return False
    
    def process_request_async(self, request: InferenceRequest) -> str:
        """Process inference request asynchronously.
        
        Args:
            request: Inference request
            
        Returns:
            Request ID for tracking
        """
        try:
            # Add to request queue
            self.request_queue.put(request, timeout=1.0)
            
            # Submit for processing
            future = self.executor.submit(self._process_request, request)
            
            with self._lock:
                self.active_requests[request.request_id] = {
                    "future": future,
                    "request": request,
                    "submitted_at": time.time()
                }
            
            logger.debug(f"Queued request {request.request_id}")
            return request.request_id
            
        except queue.Full:
            logger.warning(f"Request queue full, rejecting request {request.request_id}")
            raise RuntimeError("Server overloaded")
        except Exception as e:
            logger.error(f"Failed to queue request {request.request_id}: {e}")
            raise
    
    def _process_request(self, request: InferenceRequest) -> InferenceResponse:
        """Process inference request."""
        try:
            # Check cache first
            if self.cache_manager:
                inference_cache = self.cache_manager.get_inference_cache()
                if inference_cache:
                    cached_result = inference_cache.get_result(
                        request.image, request.question, self.config.model_type
                    )
                    if cached_result:
                        logger.debug(f"Cache hit for request {request.request_id}")
                        return InferenceResponse(
                            request_id=request.request_id,
                            answer=cached_result,
                            confidence=1.0,
                            latency_ms=1.0,  # Cache retrieval time
                            model_version=request.model_version,
                            timestamp=time.time()
                        )
            
            # Select model instance
            instance = self.load_balancer.select_instance()
            if not instance:
                raise RuntimeError("No healthy model instances available")
            
            # Process request
            response = instance.process_request(request)
            
            # Cache result if successful
            if self.cache_manager and not response.error:
                inference_cache = self.cache_manager.get_inference_cache()
                if inference_cache:
                    inference_cache.put_result(
                        request.image, request.question, response.answer,
                        self.config.model_type
                    )
            
            return response
            
        except Exception as e:
            logger.error(f"Request processing failed: {e}")
            return InferenceResponse(
                request_id=request.request_id,
                answer="",
                confidence=0.0,
                latency_ms=0.0,
                model_version=request.model_version,
                timestamp=time.time(),
                error=str(e)
            )
        finally:
            # Clean up request tracking
            with self._lock:
                self.active_requests.pop(request.request_id, None)
    
    def get_result(self, request_id: str, timeout: Optional[float] = None) -> Optional[InferenceResponse]:
        """Get result for async request."""
        with self._lock:
            request_info = self.active_requests.get(request_id)
        
        if not request_info:
            return None
        
        try:
            return request_info["future"].result(timeout=timeout)
        except Exception as e:
            logger.error(f"Failed to get result for {request_id}: {e}")
            return None
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get comprehensive server statistics."""
        with self._lock:
            active_request_count = len(self.active_requests)
        
        stats = {
            "server_id": self.server_id,
            "active_requests": active_request_count,
            "queue_size": self.request_queue.qsize(),
            "load_balancer": self.load_balancer.get_load_stats(),
            "scaling_history": self.auto_scaler.get_scaling_history()[-10:],  # Last 10 decisions
        }
        
        if self.cache_manager:
            stats["cache"] = self.cache_manager.get_cache_stats()
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Perform server health check."""
        load_stats = self.load_balancer.get_load_stats()
        
        return {
            "server_id": self.server_id,
            "status": "healthy" if load_stats["healthy_instances"] > 0 else "unhealthy",
            "healthy_instances": load_stats["healthy_instances"],
            "total_instances": load_stats["total_instances"],
            "queue_size": self.request_queue.qsize(),
            "timestamp": time.time()
        }
    
    def shutdown(self):
        """Shutdown model server."""
        logger.info(f"Shutting down model server {self.server_id}")
        
        # Stop monitoring
        self.auto_scaler.stop_monitoring()
        
        # Shutdown instances
        for instance in self.load_balancer.instances.values():
            instance.shutdown()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Shutdown caching
        if self.cache_manager:
            self.cache_manager.shutdown()


def create_deployment(
    model_path: str,
    model_type: str = "coreml",
    max_instances: int = 3,
    enable_auto_scaling: bool = True
) -> ModelServer:
    """Create production deployment for FastVLM model.
    
    Args:
        model_path: Path to model file
        model_type: Type of model ("pytorch" or "coreml")
        max_instances: Maximum number of model instances
        enable_auto_scaling: Enable automatic scaling
        
    Returns:
        Configured model server
    """
    config = DeploymentConfig(
        model_path=model_path,
        model_type=model_type,
        max_instances=max_instances,
        enable_auto_scaling=enable_auto_scaling,
        enable_caching=True,
        enable_monitoring=True,
        enable_load_balancing=True
    )
    
    return ModelServer(config)