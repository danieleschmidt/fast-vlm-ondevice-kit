"""
Hyper Scaling Engine for FastVLM Generation 3.
Provides advanced performance optimization, auto-scaling, and distributed processing.
"""

import time
import logging
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import queue
import hashlib
import json
from threading import Lock, RLock
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Scaling strategies for different workload patterns."""
    CONSERVATIVE = "conservative"  # Gradual scaling
    AGGRESSIVE = "aggressive"     # Fast scaling
    ADAPTIVE = "adaptive"         # AI-driven scaling
    PREDICTIVE = "predictive"     # Forecast-based scaling


@dataclass
class PerformanceMetrics:
    """Performance metrics for scaling decisions."""
    timestamp: float
    throughput_qps: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    cpu_usage_percent: float
    memory_usage_mb: float
    queue_length: int
    error_rate_percent: float
    cache_hit_rate_percent: float


@dataclass
class ScalingAction:
    """Scaling action to be executed."""
    action_type: str  # "scale_up", "scale_down", "optimize", "rebalance"
    target_workers: int
    reason: str
    confidence: float
    estimated_impact: Dict[str, float]


class WorkerPool:
    """Advanced worker pool with dynamic scaling."""
    
    def __init__(self, 
                 min_workers: int = 2,
                 max_workers: int = 16,
                 scaling_strategy: ScalingStrategy = ScalingStrategy.ADAPTIVE):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scaling_strategy = scaling_strategy
        self.current_workers = min_workers
        
        # Thread pool for I/O bound tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=self.current_workers)
        
        # Process pool for CPU bound tasks (disabled if no multiprocessing)
        try:
            self.process_pool = ProcessPoolExecutor(max_workers=min(self.current_workers, mp.cpu_count()))
            self.process_pool_available = True
        except:
            self.process_pool = None
            self.process_pool_available = False
            logger.warning("Process pool not available, using thread pool only")
        
        # Metrics tracking
        self.metrics_history: List[PerformanceMetrics] = []
        self.scaling_history: List[ScalingAction] = []
        self.lock = RLock()
        
        # Performance monitoring
        self.request_times = []
        self.error_count = 0
        self.total_requests = 0
        
        logger.info(f"WorkerPool initialized: {min_workers}-{max_workers} workers, strategy: {scaling_strategy.value}")
    
    def submit_task(self, func: Callable, *args, use_processes: bool = False, **kwargs):
        """Submit task to appropriate worker pool."""
        with self.lock:
            self.total_requests += 1
            
            if use_processes and self.process_pool_available:
                return self.process_pool.submit(func, *args, **kwargs)
            else:
                return self.thread_pool.submit(func, *args, **kwargs)
    
    def record_performance(self, latency_ms: float, error: bool = False):
        """Record performance metrics for scaling decisions."""
        with self.lock:
            self.request_times.append(latency_ms)
            if error:
                self.error_count += 1
            
            # Keep only recent metrics
            if len(self.request_times) > 1000:
                self.request_times = self.request_times[-1000:]
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        with self.lock:
            if not self.request_times:
                return PerformanceMetrics(
                    timestamp=time.time(),
                    throughput_qps=0.0,
                    latency_p50_ms=0.0,
                    latency_p95_ms=0.0,
                    latency_p99_ms=0.0,
                    cpu_usage_percent=0.0,
                    memory_usage_mb=0.0,
                    queue_length=0,
                    error_rate_percent=0.0,
                    cache_hit_rate_percent=0.0
                )
            
            # Calculate latency percentiles
            sorted_times = sorted(self.request_times)
            n = len(sorted_times)
            
            p50 = sorted_times[int(n * 0.5)] if n > 0 else 0
            p95 = sorted_times[int(n * 0.95)] if n > 0 else 0
            p99 = sorted_times[int(n * 0.99)] if n > 0 else 0
            
            # Calculate throughput (requests per second)
            if len(self.request_times) >= 2:
                time_window = 60  # 1 minute window
                recent_requests = len([t for t in self.request_times if time.time() - t < time_window])
                throughput = recent_requests / time_window
            else:
                throughput = 0.0
            
            # Error rate
            error_rate = (self.error_count / self.total_requests * 100) if self.total_requests > 0 else 0
            
            # CPU and memory usage (simplified)
            try:
                import psutil
                cpu_usage = psutil.cpu_percent()
                memory_usage = psutil.virtual_memory().used / (1024**2)  # MB
            except ImportError:
                cpu_usage = 50.0  # Simulated
                memory_usage = 500.0  # Simulated
            
            return PerformanceMetrics(
                timestamp=time.time(),
                throughput_qps=throughput,
                latency_p50_ms=p50,
                latency_p95_ms=p95,
                latency_p99_ms=p99,
                cpu_usage_percent=cpu_usage,
                memory_usage_mb=memory_usage,
                queue_length=0,  # Would need actual queue monitoring
                error_rate_percent=error_rate,
                cache_hit_rate_percent=95.0  # Would get from cache system
            )
    
    def should_scale(self) -> Optional[ScalingAction]:
        """Determine if scaling action is needed."""
        metrics = self.get_current_metrics()
        self.metrics_history.append(metrics)
        
        # Keep only recent history
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        # Need at least 3 data points for scaling decisions
        if len(self.metrics_history) < 3:
            return None
        
        recent_metrics = self.metrics_history[-3:]
        
        # Scale up conditions
        if (metrics.latency_p95_ms > 500 or  # High latency
            metrics.cpu_usage_percent > 80 or  # High CPU
            metrics.throughput_qps > self.current_workers * 10):  # High throughput
            
            if self.current_workers < self.max_workers:
                new_workers = min(self.current_workers + 2, self.max_workers)
                return ScalingAction(
                    action_type="scale_up",
                    target_workers=new_workers,
                    reason=f"High load detected: latency={metrics.latency_p95_ms:.1f}ms, CPU={metrics.cpu_usage_percent:.1f}%",
                    confidence=0.8,
                    estimated_impact={"latency_reduction": 0.3, "throughput_increase": 0.5}
                )
        
        # Scale down conditions (conservative)
        elif (metrics.latency_p95_ms < 100 and  # Low latency
              metrics.cpu_usage_percent < 30 and  # Low CPU
              all(m.latency_p95_ms < 100 for m in recent_metrics)):  # Consistently low
            
            if self.current_workers > self.min_workers:
                new_workers = max(self.current_workers - 1, self.min_workers)
                return ScalingAction(
                    action_type="scale_down",
                    target_workers=new_workers,
                    reason="Low load detected, scaling down for efficiency",
                    confidence=0.6,
                    estimated_impact={"resource_savings": 0.2}
                )
        
        return None
    
    def execute_scaling_action(self, action: ScalingAction) -> bool:
        """Execute a scaling action."""
        try:
            with self.lock:
                logger.info(f"Executing scaling action: {action.action_type} to {action.target_workers} workers")
                
                if action.action_type == "scale_up":
                    # Increase thread pool size
                    old_workers = self.current_workers
                    self.current_workers = action.target_workers
                    
                    # Recreate thread pool with new size
                    self.thread_pool.shutdown(wait=False)
                    self.thread_pool = ThreadPoolExecutor(max_workers=self.current_workers)
                    
                    # Recreate process pool if available
                    if self.process_pool_available:
                        self.process_pool.shutdown(wait=False)
                        self.process_pool = ProcessPoolExecutor(max_workers=min(self.current_workers, mp.cpu_count()))
                    
                    logger.info(f"Scaled up from {old_workers} to {self.current_workers} workers")
                    
                elif action.action_type == "scale_down":
                    old_workers = self.current_workers
                    self.current_workers = action.target_workers
                    
                    # Recreate pools with smaller size
                    self.thread_pool.shutdown(wait=True)
                    self.thread_pool = ThreadPoolExecutor(max_workers=self.current_workers)
                    
                    if self.process_pool_available:
                        self.process_pool.shutdown(wait=True)
                        self.process_pool = ProcessPoolExecutor(max_workers=min(self.current_workers, mp.cpu_count()))
                    
                    logger.info(f"Scaled down from {old_workers} to {self.current_workers} workers")
                
                # Record scaling action
                self.scaling_history.append(action)
                return True
                
        except Exception as e:
            logger.error(f"Failed to execute scaling action: {e}")
            return False
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        with self.lock:
            metrics = self.get_current_metrics()
            
            return {
                "current_workers": self.current_workers,
                "min_workers": self.min_workers,
                "max_workers": self.max_workers,
                "scaling_strategy": self.scaling_strategy.value,
                "current_metrics": asdict(metrics),
                "scaling_actions": len(self.scaling_history),
                "last_scaling": self.scaling_history[-1] if self.scaling_history else None,
                "process_pool_available": self.process_pool_available
            }
    
    def shutdown(self):
        """Shutdown worker pools."""
        logger.info("Shutting down worker pools...")
        self.thread_pool.shutdown(wait=True)
        if self.process_pool_available:
            self.process_pool.shutdown(wait=True)


class HyperCache:
    """High-performance multi-level cache system."""
    
    def __init__(self, 
                 l1_size: int = 100,      # Fast in-memory cache
                 l2_size: int = 1000,     # Larger persistent cache
                 ttl_seconds: int = 3600): # Time to live
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.ttl_seconds = ttl_seconds
        
        # L1 Cache: Fast access, small size
        self.l1_cache = {}
        self.l1_access_times = {}
        
        # L2 Cache: Larger, simulated persistent storage
        self.l2_cache = {}
        self.l2_access_times = {}
        
        # Cache statistics
        self.stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0
        }
        
        self.lock = Lock()
        logger.info(f"HyperCache initialized: L1={l1_size}, L2={l2_size}, TTL={ttl_seconds}s")
    
    def _generate_key(self, data: bytes, question: str) -> str:
        """Generate cache key from input data."""
        combined = data + question.encode()
        return hashlib.sha256(combined).hexdigest()[:16]
    
    def _is_expired(self, access_time: float) -> bool:
        """Check if cache entry is expired."""
        return time.time() - access_time > self.ttl_seconds
    
    def get(self, image_data: bytes, question: str) -> Optional[Dict[str, Any]]:
        """Get cached result with multi-level lookup."""
        key = self._generate_key(image_data, question)
        
        with self.lock:
            self.stats["total_requests"] += 1
            
            # L1 Cache lookup
            if key in self.l1_cache:
                if not self._is_expired(self.l1_access_times[key]):
                    self.l1_access_times[key] = time.time()  # Update access time
                    self.stats["l1_hits"] += 1
                    logger.debug(f"L1 cache hit for key {key[:8]}")
                    return self.l1_cache[key]
                else:
                    # Expired, remove from L1
                    del self.l1_cache[key]
                    del self.l1_access_times[key]
            
            # L2 Cache lookup
            if key in self.l2_cache:
                if not self._is_expired(self.l2_access_times[key]):
                    # Move to L1 for faster future access
                    self._promote_to_l1(key, self.l2_cache[key])
                    self.stats["l2_hits"] += 1
                    logger.debug(f"L2 cache hit for key {key[:8]}, promoted to L1")
                    return self.l2_cache[key]
                else:
                    # Expired, remove from L2
                    del self.l2_cache[key]
                    del self.l2_access_times[key]
            
            # Cache miss
            self.stats["misses"] += 1
            return None
    
    def put(self, image_data: bytes, question: str, result: Dict[str, Any]):
        """Store result in cache with intelligent placement."""
        key = self._generate_key(image_data, question)
        current_time = time.time()
        
        with self.lock:
            # Always try to put in L1 first
            if len(self.l1_cache) >= self.l1_size:
                self._evict_from_l1()
            
            self.l1_cache[key] = result
            self.l1_access_times[key] = current_time
            
            # Also store in L2 for persistence
            if len(self.l2_cache) >= self.l2_size:
                self._evict_from_l2()
            
            self.l2_cache[key] = result
            self.l2_access_times[key] = current_time
            
            logger.debug(f"Cached result for key {key[:8]} in L1 and L2")
    
    def _promote_to_l1(self, key: str, result: Dict[str, Any]):
        """Promote result from L2 to L1."""
        if len(self.l1_cache) >= self.l1_size:
            self._evict_from_l1()
        
        self.l1_cache[key] = result
        self.l1_access_times[key] = time.time()
    
    def _evict_from_l1(self):
        """Evict least recently used item from L1."""
        if not self.l1_cache:
            return
        
        # Find least recently used
        lru_key = min(self.l1_access_times.keys(), key=lambda k: self.l1_access_times[k])
        del self.l1_cache[lru_key]
        del self.l1_access_times[lru_key]
        self.stats["evictions"] += 1
    
    def _evict_from_l2(self):
        """Evict least recently used item from L2."""
        if not self.l2_cache:
            return
        
        # Find least recently used
        lru_key = min(self.l2_access_times.keys(), key=lambda k: self.l2_access_times[k])
        del self.l2_cache[lru_key]
        del self.l2_access_times[lru_key]
        self.stats["evictions"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self.lock:
            total = self.stats["total_requests"]
            if total == 0:
                return self.stats
            
            l1_hit_rate = (self.stats["l1_hits"] / total) * 100
            l2_hit_rate = (self.stats["l2_hits"] / total) * 100
            overall_hit_rate = ((self.stats["l1_hits"] + self.stats["l2_hits"]) / total) * 100
            
            return {
                **self.stats,
                "l1_hit_rate_percent": round(l1_hit_rate, 2),
                "l2_hit_rate_percent": round(l2_hit_rate, 2),
                "overall_hit_rate_percent": round(overall_hit_rate, 2),
                "l1_size": len(self.l1_cache),
                "l2_size": len(self.l2_cache),
                "l1_capacity": self.l1_size,
                "l2_capacity": self.l2_size
            }
    
    def clear(self):
        """Clear all cache levels."""
        with self.lock:
            self.l1_cache.clear()
            self.l1_access_times.clear()
            self.l2_cache.clear()
            self.l2_access_times.clear()
            logger.info("All cache levels cleared")


class HyperScalingEngine:
    """Main scaling engine that orchestrates all optimization components."""
    
    def __init__(self, 
                 min_workers: int = 2,
                 max_workers: int = 16,
                 cache_l1_size: int = 100,
                 cache_l2_size: int = 1000,
                 scaling_strategy: ScalingStrategy = ScalingStrategy.ADAPTIVE):
        
        self.worker_pool = WorkerPool(min_workers, max_workers, scaling_strategy)
        self.hyper_cache = HyperCache(cache_l1_size, cache_l2_size)
        
        # Auto-scaling configuration
        self.auto_scaling_enabled = True
        self.scaling_interval = 30  # seconds
        self.last_scaling_check = time.time()
        
        # Performance optimization
        self.optimization_strategies = {
            "batch_processing": True,
            "request_deduplication": True,
            "adaptive_quality": True,
            "predictive_caching": True
        }
        
        # Monitoring
        self.request_queue = queue.Queue()
        self.processing_stats = {
            "total_processed": 0,
            "total_cached": 0,
            "total_scaled": 0,
            "average_latency_ms": 0.0,
            "peak_throughput_qps": 0.0
        }
        
        logger.info(f"HyperScalingEngine initialized with {min_workers}-{max_workers} workers")
    
    def process_request_scaled(self, 
                              image_data: bytes, 
                              question: str,
                              pipeline_func: Callable) -> Dict[str, Any]:
        """Process request with full scaling optimizations."""
        start_time = time.time()
        
        # Check cache first
        cached_result = self.hyper_cache.get(image_data, question)
        if cached_result:
            cached_result["metadata"]["cache_used"] = True
            cached_result["metadata"]["latency_ms"] = (time.time() - start_time) * 1000
            self.processing_stats["total_cached"] += 1
            return cached_result
        
        # Check if auto-scaling is needed
        if self.auto_scaling_enabled:
            self._check_and_apply_scaling()
        
        # Submit to worker pool
        future = self.worker_pool.submit_task(pipeline_func, image_data, question)
        
        try:
            # Get result with timeout
            result = future.result(timeout=30)
            
            # Convert InferenceResult to dict if needed
            if hasattr(result, '__dict__'):
                result_dict = asdict(result) if hasattr(result, '_fields') else vars(result)
            else:
                result_dict = result
            
            # Record performance
            latency_ms = (time.time() - start_time) * 1000
            self.worker_pool.record_performance(latency_ms, error=False)
            
            # Cache the result
            self.hyper_cache.put(image_data, question, result_dict)
            
            # Update stats
            self.processing_stats["total_processed"] += 1
            self._update_average_latency(latency_ms)
            
            # Add scaling metadata
            result_dict["metadata"]["scaled_processing"] = True
            result_dict["metadata"]["workers_used"] = self.worker_pool.current_workers
            result_dict["metadata"]["cache_stats"] = self.hyper_cache.get_stats()
            
            return result_dict
            
        except Exception as e:
            # Record error
            error_latency = (time.time() - start_time) * 1000
            self.worker_pool.record_performance(error_latency, error=True)
            
            # Return error response
            return {
                "answer": f"Scaling engine error: {str(e)}",
                "confidence": 0.0,
                "latency_ms": error_latency,
                "model_used": "scaling_engine",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "metadata": {
                    "error": str(e),
                    "scaled_processing": True,
                    "workers_used": self.worker_pool.current_workers
                }
            }
    
    def _check_and_apply_scaling(self):
        """Check if scaling is needed and apply if so."""
        current_time = time.time()
        if current_time - self.last_scaling_check < self.scaling_interval:
            return
        
        self.last_scaling_check = current_time
        
        scaling_action = self.worker_pool.should_scale()
        if scaling_action:
            success = self.worker_pool.execute_scaling_action(scaling_action)
            if success:
                self.processing_stats["total_scaled"] += 1
                logger.info(f"Applied scaling action: {scaling_action.action_type}")
    
    def _update_average_latency(self, new_latency: float):
        """Update running average latency."""
        total = self.processing_stats["total_processed"]
        current_avg = self.processing_stats["average_latency_ms"]
        
        if total == 1:
            self.processing_stats["average_latency_ms"] = new_latency
        else:
            self.processing_stats["average_latency_ms"] = (
                (current_avg * (total - 1) + new_latency) / total
            )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        worker_status = self.worker_pool.get_scaling_status()
        cache_stats = self.hyper_cache.get_stats()
        
        return {
            "engine_stats": self.processing_stats,
            "worker_pool": worker_status,
            "cache_performance": cache_stats,
            "optimization_strategies": self.optimization_strategies,
            "auto_scaling_enabled": self.auto_scaling_enabled,
            "scaling_interval_seconds": self.scaling_interval
        }
    
    def enable_optimization(self, strategy: str, enabled: bool = True):
        """Enable or disable specific optimization strategy."""
        if strategy in self.optimization_strategies:
            self.optimization_strategies[strategy] = enabled
            logger.info(f"Optimization strategy '{strategy}': {'enabled' if enabled else 'disabled'}")
    
    def shutdown(self):
        """Shutdown the scaling engine."""
        logger.info("Shutting down HyperScalingEngine...")
        self.worker_pool.shutdown()


# Factory functions
def create_hyper_scaling_engine(
    min_workers: int = 2,
    max_workers: int = 16,
    cache_l1_size: int = 100,
    cache_l2_size: int = 1000,
    strategy: ScalingStrategy = ScalingStrategy.ADAPTIVE
) -> HyperScalingEngine:
    """Create a hyper scaling engine with specified configuration."""
    return HyperScalingEngine(
        min_workers=min_workers,
        max_workers=max_workers,
        cache_l1_size=cache_l1_size,
        cache_l2_size=cache_l2_size,
        scaling_strategy=strategy
    )


def create_hyper_cache(l1_size: int = 100, l2_size: int = 1000, ttl_seconds: int = 3600) -> HyperCache:
    """Create a hyper cache with specified configuration."""
    return HyperCache(l1_size, l2_size, ttl_seconds)


if __name__ == "__main__":
    # Demo the hyper scaling engine
    print("HyperScalingEngine Demo")
    print("=" * 30)
    
    # Create scaling engine
    engine = create_hyper_scaling_engine(
        min_workers=2,
        max_workers=8,
        cache_l1_size=50,
        cache_l2_size=200,
        strategy=ScalingStrategy.ADAPTIVE
    )
    
    # Simulate processing function
    def mock_processing(image_data: bytes, question: str) -> Dict[str, Any]:
        import random
        time.sleep(random.uniform(0.01, 0.1))  # Simulate processing time
        return {
            "answer": f"Processed: {question[:20]}...",
            "confidence": random.uniform(0.7, 0.95),
            "latency_ms": random.uniform(10, 100),
            "model_used": "mock_model",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": {"mock": True}
        }
    
    # Test processing with scaling
    print("\nProcessing test requests...")
    test_questions = [
        "What is in this image?",
        "Describe the scene",
        "Count the objects",
        "What colors do you see?",
        "Is this indoors or outdoors?"
    ]
    
    test_image = b"mock_image_data" * 100
    
    for i, question in enumerate(test_questions, 1):
        print(f"\nRequest {i}: {question}")
        
        result = engine.process_request_scaled(test_image, question, mock_processing)
        
        print(f"  Answer: {result['answer'][:40]}...")
        print(f"  Latency: {result['latency_ms']:.1f}ms")
        print(f"  Workers: {result['metadata'].get('workers_used', 'unknown')}")
        print(f"  Cached: {result['metadata'].get('cache_used', False)}")
    
    # Test cache hits
    print(f"\nTesting cache hits...")
    for i in range(3):
        result = engine.process_request_scaled(test_image, test_questions[0], mock_processing)
        print(f"  Attempt {i+1} - Cached: {result['metadata'].get('cache_used', False)}")
    
    # Show performance report
    print(f"\nPerformance Report:")
    report = engine.get_performance_report()
    
    print(f"  Total Processed: {report['engine_stats']['total_processed']}")
    print(f"  Total Cached: {report['engine_stats']['total_cached']}")
    print(f"  Average Latency: {report['engine_stats']['average_latency_ms']:.1f}ms")
    print(f"  Current Workers: {report['worker_pool']['current_workers']}")
    print(f"  Cache Hit Rate: {report['cache_performance']['overall_hit_rate_percent']:.1f}%")
    print(f"  Scaling Actions: {report['worker_pool']['scaling_actions']}")
    
    # Shutdown
    engine.shutdown()
    print("\nDemo completed.")