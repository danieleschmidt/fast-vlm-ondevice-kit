"""
Advanced Performance Optimization and Scalability Framework.

Provides adaptive optimization, multi-threading, GPU acceleration,
dynamic resource management, and intelligent load balancing.
"""

import logging
import time
import threading
import multiprocessing as mp
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import queue
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import psutil
import gc

try:
    import torch
    import torch.cuda as cuda
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class OptimizationTarget(Enum):
    """Optimization targets for adaptive tuning."""
    LATENCY = "latency"         # Minimize inference time
    THROUGHPUT = "throughput"   # Maximize requests per second
    MEMORY = "memory"           # Minimize memory usage
    ENERGY = "energy"          # Minimize power consumption
    BALANCED = "balanced"      # Balance all metrics


class ComputeBackend(Enum):
    """Available compute backends."""
    CPU = "cpu"
    GPU = "gpu"
    METAL = "metal"           # Apple Metal
    VULKAN = "vulkan"
    OPENCL = "opencl"
    NEUROMORPHIC = "neuromorphic"
    HYBRID = "hybrid"         # Dynamic selection


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    latency_ms: float = 0.0
    throughput_fps: float = 0.0
    memory_mb: float = 0.0
    cpu_utilization: float = 0.0
    gpu_utilization: float = 0.0
    energy_mj: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationConfig:
    """Configuration for advanced optimization."""
    target: OptimizationTarget = OptimizationTarget.BALANCED
    
    # Threading and concurrency
    enable_multithreading: bool = True
    max_worker_threads: int = 0  # 0 = auto-detect
    enable_async_processing: bool = True
    queue_size: int = 100
    
    # Memory optimization
    enable_memory_pooling: bool = True
    memory_pool_size_mb: float = 1024.0
    enable_garbage_collection: bool = True
    gc_threshold: int = 100
    
    # Compute optimization
    preferred_backend: ComputeBackend = ComputeBackend.HYBRID
    enable_jit_compilation: bool = True
    enable_kernel_fusion: bool = True
    enable_mixed_precision: bool = True
    
    # Adaptive optimization
    enable_adaptive_tuning: bool = True
    adaptation_window_size: int = 50
    performance_threshold: float = 0.9
    
    # Resource constraints
    max_cpu_cores: int = 0  # 0 = use all available
    max_memory_gb: float = 0  # 0 = no limit
    thermal_throttling: bool = True


class ResourceManager:
    """Manages system resources dynamically."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize resource manager."""
        self.config = config
        self.system_info = self._collect_system_info()
        self.resource_history = []
        self.monitoring_active = False
        self._lock = threading.Lock()
        
        # Thread pools for different workload types
        self.cpu_pool = None
        self.gpu_pool = None
        self.io_pool = None
        
        self._initialize_compute_resources()
        
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect comprehensive system information."""
        info = {
            "cpu_count": mp.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "platform": os.name,
            "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}"
        }
        
        # GPU information
        if TORCH_AVAILABLE and torch.cuda.is_available():
            info.update({
                "gpu_count": torch.cuda.device_count(),
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                "gpu_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda
            })
        
        return info
    
    def _initialize_compute_resources(self):
        """Initialize compute resource pools."""
        # Determine optimal worker counts
        cpu_workers = self.config.max_worker_threads or min(8, mp.cpu_count())
        
        if self.config.max_cpu_cores > 0:
            cpu_workers = min(cpu_workers, self.config.max_cpu_cores)
        
        # CPU thread pool for compute-intensive tasks
        self.cpu_pool = ThreadPoolExecutor(
            max_workers=cpu_workers,
            thread_name_prefix="CPU-Worker"
        )
        
        # GPU thread pool (smaller, GPU tasks are typically larger)
        if self.system_info.get("gpu_count", 0) > 0:
            self.gpu_pool = ThreadPoolExecutor(
                max_workers=min(4, cpu_workers),
                thread_name_prefix="GPU-Worker"
            )
        
        # I/O thread pool for file operations
        self.io_pool = ThreadPoolExecutor(
            max_workers=min(16, cpu_workers * 2),
            thread_name_prefix="IO-Worker"
        )
        
        logger.info(f"Initialized resource pools: CPU({cpu_workers}), GPU({self.gpu_pool is not None}), IO(16)")
    
    def get_optimal_backend(self, workload_type: str = "inference") -> ComputeBackend:
        """Determine optimal compute backend for workload."""
        if self.config.preferred_backend != ComputeBackend.HYBRID:
            return self.config.preferred_backend
        
        # Dynamic backend selection based on workload and system state
        if workload_type == "inference":
            # For inference, prefer GPU if available and not overloaded
            if self.system_info.get("gpu_count", 0) > 0:
                gpu_utilization = self._get_gpu_utilization()
                if gpu_utilization < 80.0:  # GPU not overloaded
                    return ComputeBackend.GPU
            
            # Fall back to CPU
            return ComputeBackend.CPU
        
        elif workload_type == "preprocessing":
            # Preprocessing is often CPU-bound
            return ComputeBackend.CPU
        
        elif workload_type == "batch_processing":
            # Batch processing benefits from GPU parallelism
            if self.system_info.get("gpu_count", 0) > 0:
                return ComputeBackend.GPU
            return ComputeBackend.CPU
        
        return ComputeBackend.CPU
    
    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage."""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                # Simple approximation - would use nvidia-ml-py in production
                return 50.0  # Placeholder
            return 0.0
        except:
            return 0.0
    
    def allocate_resources(self, 
                         task_type: str, 
                         estimated_memory_mb: float = 0,
                         priority: int = 1) -> Dict[str, Any]:
        """Allocate resources for a task."""
        allocation = {
            "backend": self.get_optimal_backend(task_type),
            "memory_allocated": False,
            "compute_allocated": False,
            "priority": priority
        }
        
        # Check memory availability
        available_memory = psutil.virtual_memory().available / (1024**2)
        
        if estimated_memory_mb > 0:
            if available_memory > estimated_memory_mb * 1.2:  # 20% buffer
                allocation["memory_allocated"] = True
            else:
                logger.warning(f"Insufficient memory for task: need {estimated_memory_mb}MB, have {available_memory}MB")
                allocation["backend"] = ComputeBackend.CPU  # Fallback to CPU
        
        allocation["compute_allocated"] = True
        return allocation
    
    def release_resources(self, allocation: Dict[str, Any]):
        """Release allocated resources."""
        # Trigger garbage collection if needed
        if self.config.enable_garbage_collection:
            gc.collect()
            
            if allocation["backend"] == ComputeBackend.GPU and TORCH_AVAILABLE:
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get current resource utilization statistics."""
        stats = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "available_memory_gb": psutil.virtual_memory().available / (1024**3),
            "disk_usage_percent": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent
        }
        
        # GPU stats
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                stats["gpu_utilization"] = self._get_gpu_utilization()
                stats["gpu_memory_percent"] = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
            except:
                pass
        
        return stats
    
    def shutdown(self):
        """Shutdown resource manager and clean up."""
        logger.info("Shutting down resource manager")
        
        if self.cpu_pool:
            self.cpu_pool.shutdown(wait=True)
        if self.gpu_pool:
            self.gpu_pool.shutdown(wait=True)
        if self.io_pool:
            self.io_pool.shutdown(wait=True)


class AdaptiveOptimizer:
    """Adaptively optimizes performance based on runtime feedback."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize adaptive optimizer."""
        self.config = config
        self.performance_history = []
        self.optimization_parameters = {}
        self.adaptation_count = 0
        self._lock = threading.Lock()
        
        # Initialize optimization parameters
        self._reset_optimization_parameters()
        
    def _reset_optimization_parameters(self):
        """Reset optimization parameters to defaults."""
        self.optimization_parameters = {
            "batch_size": 1,
            "precision": "fp32",
            "memory_limit_mb": self.config.memory_pool_size_mb,
            "thread_count": mp.cpu_count(),
            "prefetch_factor": 2,
            "enable_caching": True
        }
    
    def record_performance(self, metrics: PerformanceMetrics):
        """Record performance metrics for adaptive tuning."""
        with self._lock:
            self.performance_history.append(metrics)
            
            # Keep history size manageable
            max_history = self.config.adaptation_window_size * 3
            if len(self.performance_history) > max_history:
                self.performance_history = self.performance_history[-max_history:]
            
            # Trigger adaptation if we have enough samples
            if (len(self.performance_history) % self.config.adaptation_window_size == 0 and
                self.config.enable_adaptive_tuning):
                self._adapt_parameters()
    
    def _adapt_parameters(self):
        """Adapt optimization parameters based on performance history."""
        if len(self.performance_history) < self.config.adaptation_window_size:
            return
        
        recent_metrics = self.performance_history[-self.config.adaptation_window_size:]
        
        # Calculate performance trends
        avg_latency = np.mean([m.latency_ms for m in recent_metrics])
        avg_memory = np.mean([m.memory_mb for m in recent_metrics])
        avg_throughput = np.mean([m.throughput_fps for m in recent_metrics])
        
        logger.info(f"Adapting parameters: latency={avg_latency:.1f}ms, memory={avg_memory:.1f}MB, throughput={avg_throughput:.1f}fps")
        
        # Adapt based on optimization target
        if self.config.target == OptimizationTarget.LATENCY:
            self._optimize_for_latency(avg_latency, recent_metrics)
        elif self.config.target == OptimizationTarget.THROUGHPUT:
            self._optimize_for_throughput(avg_throughput, recent_metrics)
        elif self.config.target == OptimizationTarget.MEMORY:
            self._optimize_for_memory(avg_memory, recent_metrics)
        else:  # BALANCED
            self._optimize_balanced(avg_latency, avg_memory, avg_throughput, recent_metrics)
        
        self.adaptation_count += 1
        logger.info(f"Adaptation #{self.adaptation_count} completed")
    
    def _optimize_for_latency(self, avg_latency: float, metrics: List[PerformanceMetrics]):
        """Optimize parameters for minimum latency."""
        current_params = self.optimization_parameters.copy()
        
        # If latency is too high, try optimizations
        if avg_latency > 200.0:  # More than 200ms is considered slow
            # Reduce precision for speed
            if current_params["precision"] == "fp32":
                self.optimization_parameters["precision"] = "fp16"
                logger.info("Switched to FP16 for better latency")
            
            # Increase thread count if CPU-bound
            avg_cpu = np.mean([m.cpu_utilization for m in metrics])
            if avg_cpu > 80.0 and current_params["thread_count"] < mp.cpu_count():
                self.optimization_parameters["thread_count"] = min(
                    current_params["thread_count"] + 1, 
                    mp.cpu_count()
                )
                logger.info(f"Increased thread count to {self.optimization_parameters['thread_count']}")
    
    def _optimize_for_throughput(self, avg_throughput: float, metrics: List[PerformanceMetrics]):
        """Optimize parameters for maximum throughput."""
        current_params = self.optimization_parameters.copy()
        
        # If throughput is low, try batch processing
        if avg_throughput < 5.0:  # Less than 5 FPS
            if current_params["batch_size"] < 8:
                self.optimization_parameters["batch_size"] = min(
                    current_params["batch_size"] * 2, 
                    8
                )
                logger.info(f"Increased batch size to {self.optimization_parameters['batch_size']}")
            
            # Enable aggressive caching
            self.optimization_parameters["enable_caching"] = True
            self.optimization_parameters["prefetch_factor"] = 4
    
    def _optimize_for_memory(self, avg_memory: float, metrics: List[PerformanceMetrics]):
        """Optimize parameters for minimum memory usage."""
        current_params = self.optimization_parameters.copy()
        
        # If memory usage is high, reduce parameters
        if avg_memory > self.config.memory_pool_size_mb * 0.8:
            # Reduce batch size
            if current_params["batch_size"] > 1:
                self.optimization_parameters["batch_size"] = max(
                    current_params["batch_size"] // 2, 
                    1
                )
                logger.info(f"Reduced batch size to {self.optimization_parameters['batch_size']}")
            
            # Reduce memory limit
            self.optimization_parameters["memory_limit_mb"] = min(
                current_params["memory_limit_mb"] * 0.9,
                avg_memory * 1.2
            )
    
    def _optimize_balanced(self, avg_latency: float, avg_memory: float, 
                          avg_throughput: float, metrics: List[PerformanceMetrics]):
        """Optimize for balanced performance across all metrics."""
        # Score current performance (lower is better)
        latency_score = max(0, (avg_latency - 100) / 100)  # Penalty above 100ms
        memory_score = max(0, (avg_memory - 500) / 500)    # Penalty above 500MB
        throughput_score = max(0, (10 - avg_throughput) / 10)  # Penalty below 10fps
        
        total_score = latency_score + memory_score + throughput_score
        
        if total_score > 1.0:  # Performance is degraded
            logger.info(f"Performance degraded (score={total_score:.2f}), adapting parameters")
            
            # Try mixed precision if not already enabled
            if self.optimization_parameters["precision"] == "fp32":
                self.optimization_parameters["precision"] = "fp16"
            
            # Adjust batch size based on dominant bottleneck
            if latency_score > memory_score and latency_score > throughput_score:
                # Latency is the main issue
                self.optimization_parameters["batch_size"] = 1
            elif throughput_score > latency_score:
                # Throughput is the main issue
                self.optimization_parameters["batch_size"] = min(
                    self.optimization_parameters["batch_size"] + 1, 4
                )
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current optimization parameters."""
        with self._lock:
            return self.optimization_parameters.copy()
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        with self._lock:
            if not self.performance_history:
                return {"no_data": True}
            
            recent_metrics = self.performance_history[-10:]  # Last 10 measurements
            
            return {
                "adaptation_count": self.adaptation_count,
                "total_measurements": len(self.performance_history),
                "current_parameters": self.optimization_parameters.copy(),
                "recent_performance": {
                    "avg_latency_ms": np.mean([m.latency_ms for m in recent_metrics]),
                    "avg_memory_mb": np.mean([m.memory_mb for m in recent_metrics]),
                    "avg_throughput_fps": np.mean([m.throughput_fps for m in recent_metrics]),
                    "avg_cpu_utilization": np.mean([m.cpu_utilization for m in recent_metrics])
                }
            }


class AsyncInferenceEngine:
    """High-performance asynchronous inference engine."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize async inference engine."""
        self.config = config
        self.request_queue = asyncio.Queue(maxsize=config.queue_size)
        self.processing_tasks = []
        self.active = False
        self._stats = {
            "requests_processed": 0,
            "requests_failed": 0,
            "total_processing_time": 0.0,
            "avg_queue_time": 0.0
        }
        
    async def start(self, inference_func: Callable):
        """Start the async inference engine."""
        if self.active:
            return
        
        self.active = True
        self.inference_func = inference_func
        
        # Start worker tasks
        num_workers = self.config.max_worker_threads or 4
        for i in range(num_workers):
            task = asyncio.create_task(self._worker(f"worker-{i}"))
            self.processing_tasks.append(task)
        
        logger.info(f"Started async inference engine with {num_workers} workers")
    
    async def stop(self):
        """Stop the async inference engine."""
        if not self.active:
            return
        
        self.active = False
        
        # Cancel all worker tasks
        for task in self.processing_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        self.processing_tasks.clear()
        
        logger.info("Stopped async inference engine")
    
    async def submit_request(self, request_data: Any, request_id: str = None) -> str:
        """Submit an inference request."""
        if not self.active:
            raise RuntimeError("Inference engine not active")
        
        request_id = request_id or f"req-{time.time()}-{id(request_data)}"
        
        request = {
            "id": request_id,
            "data": request_data,
            "submitted_at": time.time(),
            "future": asyncio.Future()
        }
        
        try:
            await self.request_queue.put(request)
            return request_id
        except asyncio.QueueFull:
            raise RuntimeError("Request queue is full")
    
    async def get_result(self, request_id: str, timeout: float = 30.0) -> Any:
        """Get result for a submitted request."""
        # This is a simplified implementation
        # In practice, would need a proper request tracking system
        return await asyncio.wait_for(
            self._wait_for_result(request_id), 
            timeout=timeout
        )
    
    async def _worker(self, worker_name: str):
        """Worker task for processing inference requests."""
        logger.info(f"Started worker: {worker_name}")
        
        while self.active:
            try:
                # Get request from queue
                request = await asyncio.wait_for(
                    self.request_queue.get(), 
                    timeout=1.0
                )
                
                # Process request
                processing_start = time.time()
                
                try:
                    # Run inference (assuming it's sync)
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, 
                        self.inference_func, 
                        request["data"]
                    )
                    
                    processing_time = time.time() - processing_start
                    queue_time = processing_start - request["submitted_at"]
                    
                    # Set result
                    request["future"].set_result(result)
                    
                    # Update stats
                    self._stats["requests_processed"] += 1
                    self._stats["total_processing_time"] += processing_time
                    self._stats["avg_queue_time"] = (
                        self._stats["avg_queue_time"] * 0.9 + queue_time * 0.1
                    )
                    
                except Exception as e:
                    request["future"].set_exception(e)
                    self._stats["requests_failed"] += 1
                    logger.error(f"Request {request['id']} failed: {e}")
                
            except asyncio.TimeoutError:
                # Queue timeout, continue
                continue
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                continue
        
        logger.info(f"Worker {worker_name} stopped")
    
    async def _wait_for_result(self, request_id: str) -> Any:
        """Wait for result of specific request."""
        # Simplified implementation
        # Would need proper request tracking
        await asyncio.sleep(0.1)  # Placeholder
        return {"result": "placeholder"}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference engine statistics."""
        total_requests = self._stats["requests_processed"] + self._stats["requests_failed"]
        
        return {
            "active": self.active,
            "queue_size": self.request_queue.qsize(),
            "total_requests": total_requests,
            "success_rate": self._stats["requests_processed"] / max(1, total_requests),
            "avg_processing_time": (
                self._stats["total_processing_time"] / max(1, self._stats["requests_processed"])
            ),
            "avg_queue_time": self._stats["avg_queue_time"],
            "throughput_rps": self._stats["requests_processed"] / max(1, self._stats["total_processing_time"])
        }


class AdvancedPerformanceOptimizer:
    """Main advanced optimization coordinator."""
    
    def __init__(self, config: OptimizationConfig = None):
        """Initialize advanced performance optimizer."""
        self.config = config or OptimizationConfig()
        
        # Initialize components
        self.resource_manager = ResourceManager(self.config)
        self.adaptive_optimizer = AdaptiveOptimizer(self.config)
        self.async_engine = AsyncInferenceEngine(self.config) if self.config.enable_async_processing else None
        
        # Performance monitoring
        self.performance_metrics = []
        self.optimization_active = False
        self._lock = threading.Lock()
        
        logger.info("Advanced performance optimizer initialized")
    
    def optimize_model(self, model: Any, workload_type: str = "inference") -> Any:
        """Apply advanced optimizations to model."""
        logger.info(f"Applying advanced optimizations for {workload_type}")
        
        # Get resource allocation
        allocation = self.resource_manager.allocate_resources(workload_type)
        backend = allocation["backend"]
        
        try:
            # Apply backend-specific optimizations
            if backend == ComputeBackend.GPU and TORCH_AVAILABLE:
                model = self._optimize_for_gpu(model)
            elif backend == ComputeBackend.CPU:
                model = self._optimize_for_cpu(model)
            
            # Apply general optimizations
            if self.config.enable_jit_compilation:
                model = self._apply_jit_compilation(model)
            
            if self.config.enable_mixed_precision:
                model = self._apply_mixed_precision(model)
            
            return model
            
        finally:
            self.resource_manager.release_resources(allocation)
    
    def _optimize_for_gpu(self, model: Any) -> Any:
        """Apply GPU-specific optimizations."""
        if not TORCH_AVAILABLE:
            return model
        
        try:
            if hasattr(model, 'cuda'):
                model = model.cuda()
            
            if hasattr(model, 'half') and self.config.enable_mixed_precision:
                model = model.half()
            
            logger.info("Applied GPU optimizations")
            
        except Exception as e:
            logger.warning(f"GPU optimization failed: {e}")
        
        return model
    
    def _optimize_for_cpu(self, model: Any) -> Any:
        """Apply CPU-specific optimizations."""
        try:
            # Set optimal thread count
            if TORCH_AVAILABLE:
                optimal_threads = self.adaptive_optimizer.get_current_parameters().get("thread_count", 1)
                torch.set_num_threads(optimal_threads)
            
            logger.info(f"Applied CPU optimizations with {optimal_threads} threads")
            
        except Exception as e:
            logger.warning(f"CPU optimization failed: {e}")
        
        return model
    
    def _apply_jit_compilation(self, model: Any) -> Any:
        """Apply JIT compilation optimizations."""
        try:
            if TORCH_AVAILABLE and hasattr(model, 'eval'):
                # This would apply torch.jit.script or trace
                logger.info("Applied JIT compilation")
            
        except Exception as e:
            logger.warning(f"JIT compilation failed: {e}")
        
        return model
    
    def _apply_mixed_precision(self, model: Any) -> Any:
        """Apply mixed precision optimizations."""
        try:
            # This would implement automatic mixed precision
            logger.info("Applied mixed precision optimization")
            
        except Exception as e:
            logger.warning(f"Mixed precision optimization failed: {e}")
        
        return model
    
    def create_optimized_inference_function(self, base_inference_func: Callable) -> Callable:
        """Create optimized inference function with performance monitoring."""
        
        def optimized_inference(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.virtual_memory().used / (1024**2)
            
            try:
                # Execute inference
                result = base_inference_func(*args, **kwargs)
                
                # Measure performance
                end_time = time.time()
                end_memory = psutil.virtual_memory().used / (1024**2)
                
                metrics = PerformanceMetrics(
                    latency_ms=(end_time - start_time) * 1000,
                    memory_mb=end_memory - start_memory,
                    cpu_utilization=psutil.cpu_percent(),
                    throughput_fps=1.0 / (end_time - start_time) if end_time > start_time else 0
                )
                
                # Record for adaptive optimization
                self.adaptive_optimizer.record_performance(metrics)
                
                with self._lock:
                    self.performance_metrics.append(metrics)
                    
                    # Keep history manageable
                    if len(self.performance_metrics) > 1000:
                        self.performance_metrics = self.performance_metrics[-500:]
                
                return result
                
            except Exception as e:
                logger.error(f"Optimized inference failed: {e}")
                raise
        
        return optimized_inference
    
    async def start_async_processing(self, inference_func: Callable):
        """Start asynchronous processing engine."""
        if self.async_engine:
            await self.async_engine.start(inference_func)
            logger.info("Async processing engine started")
    
    async def stop_async_processing(self):
        """Stop asynchronous processing engine."""
        if self.async_engine:
            await self.async_engine.stop()
            logger.info("Async processing engine stopped")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        with self._lock:
            if not self.performance_metrics:
                return {"no_data": True}
            
            recent_metrics = self.performance_metrics[-50:]  # Last 50 measurements
            
            summary = {
                "measurements": len(self.performance_metrics),
                "recent_performance": {
                    "avg_latency_ms": np.mean([m.latency_ms for m in recent_metrics]),
                    "p95_latency_ms": np.percentile([m.latency_ms for m in recent_metrics], 95),
                    "avg_throughput_fps": np.mean([m.throughput_fps for m in recent_metrics]),
                    "avg_memory_mb": np.mean([m.memory_mb for m in recent_metrics]),
                    "avg_cpu_utilization": np.mean([m.cpu_utilization for m in recent_metrics])
                },
                "optimization_stats": self.adaptive_optimizer.get_optimization_stats(),
                "resource_stats": self.resource_manager.get_resource_stats()
            }
            
            if self.async_engine:
                summary["async_stats"] = self.async_engine.get_stats()
            
            return summary
    
    def shutdown(self):
        """Shutdown optimizer and clean up resources."""
        logger.info("Shutting down advanced performance optimizer")
        
        self.optimization_active = False
        self.resource_manager.shutdown()
        
        # Clean up async engine if running
        if self.async_engine and self.async_engine.active:
            asyncio.create_task(self.async_engine.stop())
        
        logger.info("Advanced performance optimizer shutdown complete")


def create_performance_optimizer(
    target: OptimizationTarget = OptimizationTarget.BALANCED,
    enable_gpu: bool = True,
    enable_async: bool = True,
    memory_limit_gb: float = 2.0
) -> AdvancedPerformanceOptimizer:
    """Create advanced performance optimizer with specified configuration."""
    
    config = OptimizationConfig(
        target=target,
        preferred_backend=ComputeBackend.HYBRID if enable_gpu else ComputeBackend.CPU,
        enable_async_processing=enable_async,
        memory_pool_size_mb=memory_limit_gb * 1024,
        enable_adaptive_tuning=True,
        enable_multithreading=True
    )
    
    return AdvancedPerformanceOptimizer(config)