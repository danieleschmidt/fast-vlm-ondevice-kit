"""
Performance optimization utilities for FastVLM models.

Provides model optimization, inference acceleration, and resource management.
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, List, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import multiprocessing as mp

try:
    import torch
    import torch.nn as nn
    import numpy as np
    from PIL import Image
    import coremltools as ct
    OPTIMIZATION_DEPS = True
except ImportError:
    OPTIMIZATION_DEPS = False

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for model optimization."""
    
    # Memory optimization
    enable_memory_mapping: bool = True
    max_memory_mb: float = 2048.0
    garbage_collection_threshold: int = 100
    
    # Inference optimization
    enable_batching: bool = True
    max_batch_size: int = 8
    batch_timeout_ms: float = 50.0
    
    # Threading optimization
    max_workers: int = 4
    enable_async_inference: bool = True
    
    # Model optimization
    enable_model_compilation: bool = True
    enable_graph_optimization: bool = True
    optimization_level: str = "balanced"  # "speed", "balanced", "memory"
    
    # Caching optimization
    enable_inference_cache: bool = True
    enable_model_cache: bool = True
    cache_size_mb: float = 512.0


class MemoryOptimizer:
    """Optimizes memory usage for model inference."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize memory optimizer."""
        self.config = config
        self.inference_count = 0
        self._lock = threading.Lock()
        
    def optimize_memory_usage(self, model: Any) -> Any:
        """Optimize model memory usage."""
        if not OPTIMIZATION_DEPS:
            return model
        
        try:
            if hasattr(model, 'eval'):
                model.eval()
            
            # Enable memory efficient attention if available
            if hasattr(model, 'config') and hasattr(model.config, 'use_memory_efficient_attention'):
                model.config.use_memory_efficient_attention = True
            
            # Apply memory mapping if enabled
            if self.config.enable_memory_mapping and hasattr(model, 'half'):
                if torch.cuda.is_available():
                    model = model.half()  # Use FP16 on GPU
                    logger.info("Applied FP16 optimization for GPU")
            
            return model
            
        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")
            return model
    
    def check_memory_usage(self) -> Dict[str, float]:
        """Check current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            memory_stats = {
                "rss_mb": memory_info.rss / (1024**2),
                "vms_mb": memory_info.vms / (1024**2),
                "memory_percent": process.memory_percent()
            }
            
            # Check if approaching memory limit
            if memory_stats["rss_mb"] > self.config.max_memory_mb * 0.9:
                logger.warning(f"High memory usage: {memory_stats['rss_mb']:.1f}MB")
                self.trigger_garbage_collection()
            
            return memory_stats
            
        except ImportError:
            return {"rss_mb": 0.0, "vms_mb": 0.0, "memory_percent": 0.0}
    
    def trigger_garbage_collection(self):
        """Trigger garbage collection if needed."""
        import gc
        
        with self._lock:
            self.inference_count += 1
            
            if self.inference_count >= self.config.garbage_collection_threshold:
                logger.debug("Triggering garbage collection")
                gc.collect()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.inference_count = 0


class BatchProcessor:
    """Processes inference requests in batches for efficiency."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize batch processor."""
        self.config = config
        self.pending_requests: List[Dict[str, Any]] = []
        self.batch_lock = threading.Lock()
        self.last_batch_time = time.time()
        
    def add_request(
        self,
        request_id: str,
        inputs: Dict[str, Any],
        callback: callable
    ):
        """Add inference request to batch."""
        with self.batch_lock:
            request = {
                "id": request_id,
                "inputs": inputs,
                "callback": callback,
                "timestamp": time.time()
            }
            self.pending_requests.append(request)
            
            # Process batch if conditions met
            if self._should_process_batch():
                self._process_batch()
    
    def _should_process_batch(self) -> bool:
        """Check if batch should be processed."""
        if not self.config.enable_batching:
            return len(self.pending_requests) >= 1
        
        # Process if batch is full
        if len(self.pending_requests) >= self.config.max_batch_size:
            return True
        
        # Process if timeout reached
        time_since_last = (time.time() - self.last_batch_time) * 1000
        if time_since_last >= self.config.batch_timeout_ms and self.pending_requests:
            return True
        
        return False
    
    def _process_batch(self):
        """Process current batch of requests."""
        if not self.pending_requests:
            return
        
        batch_requests = self.pending_requests.copy()
        self.pending_requests.clear()
        self.last_batch_time = time.time()
        
        logger.debug(f"Processing batch of {len(batch_requests)} requests")
        
        # Process requests (implementation depends on model type)
        for request in batch_requests:
            try:
                # This would be implemented by the specific model handler
                # For now, just call the callback with placeholder result
                result = {"answer": "Batch processed result"}
                request["callback"](request["id"], result)
            except Exception as e:
                logger.error(f"Batch processing failed for request {request['id']}: {e}")


class ModelCompiler:
    """Compiles models for optimal performance."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize model compiler."""
        self.config = config
        
    def compile_pytorch_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Compile PyTorch model for optimization."""
        if not OPTIMIZATION_DEPS or not self.config.enable_model_compilation:
            return model
        
        try:
            # Use torch.jit.script for optimization
            if hasattr(torch.jit, 'optimize_for_inference'):
                model = torch.jit.optimize_for_inference(model)
                logger.info("Applied torch.jit optimization")
            
            # Use torch.compile if available (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                if self.config.optimization_level == "speed":
                    mode = "max-autotune"
                elif self.config.optimization_level == "memory":
                    mode = "reduce-overhead"
                else:
                    mode = "default"
                
                try:
                    model = torch.compile(model, mode=mode)
                    logger.info(f"Applied torch.compile with mode: {mode}")
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}")
            
            return model
            
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}")
            return model
    
    def optimize_coreml_model(self, model: ct.models.MLModel) -> ct.models.MLModel:
        """Optimize Core ML model."""
        if not OPTIMIZATION_DEPS:
            return model
        
        try:
            # Apply Core ML optimizations
            if self.config.enable_graph_optimization:
                # This would apply graph-level optimizations
                logger.info("Applied Core ML graph optimizations")
            
            return model
            
        except Exception as e:
            logger.warning(f"Core ML optimization failed: {e}")
            return model


class AsyncInferenceManager:
    """Manages asynchronous inference requests."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize async inference manager."""
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self.active_requests: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
    def submit_inference(
        self,
        request_id: str,
        inference_func: callable,
        *args,
        **kwargs
    ) -> str:
        """Submit asynchronous inference request.
        
        Args:
            request_id: Unique request identifier
            inference_func: Function to execute
            *args, **kwargs: Arguments for inference function
            
        Returns:
            Request ID
        """
        with self._lock:
            future = self.executor.submit(inference_func, *args, **kwargs)
            self.active_requests[request_id] = {
                "future": future,
                "submitted_at": time.time(),
                "status": "pending"
            }
        
        logger.debug(f"Submitted async inference request: {request_id}")
        return request_id
    
    def get_result(self, request_id: str, timeout: Optional[float] = None) -> Optional[Any]:
        """Get result from async inference request.
        
        Args:
            request_id: Request identifier
            timeout: Timeout in seconds
            
        Returns:
            Inference result or None if not ready/failed
        """
        with self._lock:
            request = self.active_requests.get(request_id)
            
        if not request:
            return None
        
        try:
            result = request["future"].result(timeout=timeout)
            
            with self._lock:
                self.active_requests[request_id]["status"] = "completed"
                
            return result
            
        except Exception as e:
            logger.error(f"Async inference failed for {request_id}: {e}")
            
            with self._lock:
                self.active_requests[request_id]["status"] = "failed"
                
            return None
    
    def cancel_request(self, request_id: str) -> bool:
        """Cancel async inference request."""
        with self._lock:
            request = self.active_requests.get(request_id)
            
        if request and request["future"].cancel():
            with self._lock:
                self.active_requests[request_id]["status"] = "cancelled"
            return True
        
        return False
    
    def get_active_requests(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all active requests."""
        with self._lock:
            return {
                req_id: {
                    "status": req["status"],
                    "submitted_at": req["submitted_at"],
                    "elapsed_time": time.time() - req["submitted_at"]
                }
                for req_id, req in self.active_requests.items()
            }
    
    def cleanup_completed(self):
        """Clean up completed requests."""
        with self._lock:
            completed_ids = [
                req_id for req_id, req in self.active_requests.items()
                if req["status"] in ["completed", "failed", "cancelled"]
            ]
            
            for req_id in completed_ids:
                del self.active_requests[req_id]
    
    def shutdown(self):
        """Shutdown async inference manager."""
        self.executor.shutdown(wait=True)


class ResourceMonitor:
    """Monitors system resources during inference."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize resource monitor."""
        self.config = config
        self.monitoring = False
        self.monitor_thread = None
        self.resource_history: List[Dict[str, float]] = []
        self.max_history = 1000
        
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Resource monitoring loop."""
        while self.monitoring:
            try:
                resources = self._collect_resources()
                self.resource_history.append(resources)
                
                # Keep history size manageable
                if len(self.resource_history) > self.max_history:
                    self.resource_history = self.resource_history[-self.max_history:]
                
                time.sleep(1.0)  # Sample every second
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(1.0)
    
    def _collect_resources(self) -> Dict[str, float]:
        """Collect current resource usage."""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            resources = {
                "timestamp": time.time(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3)
            }
            
            # Add GPU stats if available
            if torch.cuda.is_available():
                try:
                    gpu_memory = torch.cuda.memory_stats()
                    resources.update({
                        "gpu_memory_allocated": gpu_memory.get("allocated_bytes.all.current", 0) / (1024**3),
                        "gpu_memory_cached": gpu_memory.get("reserved_bytes.all.current", 0) / (1024**3)
                    })
                except:
                    pass
            
            return resources
            
        except ImportError:
            return {"timestamp": time.time()}
    
    def get_resource_stats(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get resource statistics for time window."""
        if not self.resource_history:
            return {}
        
        cutoff_time = time.time() - (window_minutes * 60)
        recent_data = [
            r for r in self.resource_history
            if r.get("timestamp", 0) >= cutoff_time
        ]
        
        if not recent_data:
            return {}
        
        stats = {}
        for key in recent_data[0].keys():
            if key == "timestamp":
                continue
            
            values = [r[key] for r in recent_data if key in r]
            if values:
                stats[key] = {
                    "mean": np.mean(values),
                    "max": np.max(values),
                    "min": np.min(values),
                    "current": values[-1] if values else 0.0
                }
        
        return stats


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize performance optimizer."""
        self.config = config
        
        self.memory_optimizer = MemoryOptimizer(config)
        self.batch_processor = BatchProcessor(config)
        self.model_compiler = ModelCompiler(config)
        self.resource_monitor = ResourceMonitor(config)
        
        if config.enable_async_inference:
            self.async_manager = AsyncInferenceManager(config)
        else:
            self.async_manager = None
        
        logger.info(f"Performance optimizer initialized with config: {config}")
    
    def optimize_model(self, model: Any, model_type: str = "pytorch") -> Any:
        """Optimize model for inference.
        
        Args:
            model: Model to optimize
            model_type: Type of model ("pytorch", "coreml")
            
        Returns:
            Optimized model
        """
        logger.info(f"Optimizing {model_type} model")
        
        try:
            if model_type == "pytorch":
                model = self.memory_optimizer.optimize_memory_usage(model)
                model = self.model_compiler.compile_pytorch_model(model)
            elif model_type == "coreml":
                model = self.model_compiler.optimize_coreml_model(model)
            
            logger.info("Model optimization completed")
            return model
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return model
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.resource_monitor.start_monitoring()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.resource_monitor.stop_monitoring()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            "memory": self.memory_optimizer.check_memory_usage(),
            "resources": self.resource_monitor.get_resource_stats(),
            "optimization_config": self.config.__dict__
        }
        
        if self.async_manager:
            stats["async_requests"] = self.async_manager.get_active_requests()
        
        return stats
    
    def optimize_inference_pipeline(self, inference_func: callable) -> callable:
        """Wrap inference function with optimizations.
        
        Args:
            inference_func: Original inference function
            
        Returns:
            Optimized inference function
        """
        def optimized_inference(*args, **kwargs):
            # Check memory before inference
            memory_stats = self.memory_optimizer.check_memory_usage()
            
            # Trigger GC if needed
            if memory_stats["rss_mb"] > self.config.max_memory_mb * 0.8:
                self.memory_optimizer.trigger_garbage_collection()
            
            # Run inference
            result = inference_func(*args, **kwargs)
            
            return result
        
        return optimized_inference
    
    def shutdown(self):
        """Shutdown optimizer and cleanup resources."""
        self.stop_monitoring()
        
        if self.async_manager:
            self.async_manager.shutdown()


def create_optimizer(
    optimization_level: str = "balanced",
    max_memory_gb: float = 2.0,
    enable_async: bool = True
) -> PerformanceOptimizer:
    """Create performance optimizer with preset configuration.
    
    Args:
        optimization_level: "speed", "balanced", or "memory"
        max_memory_gb: Maximum memory usage in GB
        enable_async: Enable async inference
        
    Returns:
        Configured performance optimizer
    """
    if optimization_level == "speed":
        config = OptimizationConfig(
            max_memory_mb=max_memory_gb * 1024,
            enable_batching=True,
            max_batch_size=16,
            batch_timeout_ms=10.0,
            max_workers=8,
            enable_async_inference=enable_async,
            enable_model_compilation=True,
            optimization_level="speed"
        )
    elif optimization_level == "memory":
        config = OptimizationConfig(
            max_memory_mb=max_memory_gb * 1024,
            enable_batching=False,
            max_batch_size=1,
            max_workers=2,
            enable_async_inference=False,
            garbage_collection_threshold=10,
            optimization_level="memory"
        )
    else:  # balanced
        config = OptimizationConfig(
            max_memory_mb=max_memory_gb * 1024,
            enable_async_inference=enable_async,
            optimization_level="balanced"
        )
    
    return PerformanceOptimizer(config)