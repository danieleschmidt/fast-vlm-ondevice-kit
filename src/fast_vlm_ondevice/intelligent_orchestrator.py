"""
Intelligent orchestrator for FastVLM on-device deployment.

Coordinates all system components with intelligent decision-making,
resource allocation, and adaptive optimization for optimal mobile performance.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from pathlib import Path
import weakref
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager

from .core_pipeline import FastVLMCorePipeline as FastVLMPipeline, InferenceConfig as PipelineConfig
# Define PipelineMetrics locally if not in core_pipeline
from dataclasses import dataclass
@dataclass 
class PipelineMetrics:
    inference_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    accuracy_score: float = 0.0
from .mobile_optimizer import MobileOptimizer, MobileOptimizationConfig, OptimizationLevel
from .reliability_engine import ReliabilityEngine, HealthStatus
from .monitoring import MetricsCollector, PerformanceProfiler

logger = logging.getLogger(__name__)


class OrchestratorMode(Enum):
    """Orchestrator operation modes."""
    INITIALIZATION = "initialization"
    ACTIVE = "active"
    OPTIMIZATION = "optimization"
    RECOVERY = "recovery"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"


class ResourceAllocation(Enum):
    """Resource allocation strategies."""
    CONSERVATIVE = "conservative"  # Prioritize stability
    BALANCED = "balanced"         # Balance performance and stability
    AGGRESSIVE = "aggressive"     # Prioritize performance
    ADAPTIVE = "adaptive"         # Dynamic based on conditions


@dataclass
class OrchestratorConfig:
    """Configuration for the intelligent orchestrator."""
    # Core settings
    model_path: str = "models/fastvlm.mlpackage"
    max_concurrent_requests: int = 4
    request_timeout_seconds: float = 30.0
    
    # Resource management
    resource_allocation: ResourceAllocation = ResourceAllocation.ADAPTIVE
    enable_request_batching: bool = True
    batch_timeout_ms: float = 50.0
    max_batch_size: int = 8
    
    # Optimization settings
    enable_intelligent_caching: bool = True
    cache_prediction_window: int = 10
    enable_preemptive_optimization: bool = True
    
    # Reliability settings
    enable_health_monitoring: bool = True
    health_check_interval: float = 30.0
    enable_self_healing: bool = True
    max_recovery_attempts: int = 3
    
    # Performance settings
    target_latency_p95_ms: float = 300.0
    target_throughput_fps: float = 4.0
    memory_limit_mb: float = 1500.0
    
    # Mobile optimization
    mobile_optimization: MobileOptimizationConfig = field(default_factory=MobileOptimizationConfig)


@dataclass
class RequestContext:
    """Context for processing requests."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    priority: int = 1  # 1=highest, 5=lowest
    timeout_seconds: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Performance tracking
    queue_time_ms: Optional[float] = None
    processing_time_ms: Optional[float] = None
    total_time_ms: Optional[float] = None
    
    # Resource usage
    memory_peak_mb: Optional[float] = None
    cpu_time_ms: Optional[float] = None


@dataclass
class SystemState:
    """Current system state."""
    mode: OrchestratorMode = OrchestratorMode.INITIALIZATION
    health_status: HealthStatus = HealthStatus.HEALTHY
    active_requests: int = 0
    queued_requests: int = 0
    
    # Performance metrics
    current_latency_p95_ms: float = 0.0
    current_throughput_fps: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Optimization state
    last_optimization: float = 0.0
    optimization_in_progress: bool = False
    adaptive_parameters: Dict[str, Any] = field(default_factory=dict)


class IntelligentRequestScheduler:
    """Intelligent request scheduling and batching."""
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.request_queue = asyncio.PriorityQueue()
        self.active_requests = {}
        self.completed_requests = []
        self.batch_buffer = []
        self.last_batch_time = 0.0
        self._lock = asyncio.Lock()
        
    async def submit_request(self, image_data: Any, question: str, 
                           context: RequestContext = None) -> str:
        """Submit a request for processing."""
        if context is None:
            context = RequestContext()
        
        # Add to queue with priority
        await self.request_queue.put((context.priority, context.timestamp, {
            "image_data": image_data,
            "question": question,
            "context": context
        }))
        
        logger.debug(f"Request {context.request_id[:8]} queued with priority {context.priority}")
        return context.request_id
    
    async def get_next_batch(self) -> List[Dict[str, Any]]:
        """Get next batch of requests for processing."""
        async with self._lock:
            batch = []
            current_time = time.time()
            
            # Check if we should process accumulated batch
            should_process_batch = (
                len(self.batch_buffer) >= self.config.max_batch_size or
                (self.batch_buffer and 
                 current_time - self.last_batch_time > self.config.batch_timeout_ms / 1000)
            )
            
            if should_process_batch:
                batch = self.batch_buffer.copy()
                self.batch_buffer.clear()
                self.last_batch_time = current_time
            
            # Try to add more requests to batch if enabled
            if self.config.enable_request_batching and len(batch) < self.config.max_batch_size:
                while (len(batch) < self.config.max_batch_size and 
                       not self.request_queue.empty()):
                    try:
                        priority, timestamp, request_data = await asyncio.wait_for(
                            self.request_queue.get(), timeout=0.01
                        )
                        
                        request_data["context"].queue_time_ms = (current_time - timestamp) * 1000
                        batch.append(request_data)
                        
                    except asyncio.TimeoutError:
                        break
            
            return batch
    
    async def mark_request_completed(self, request_id: str, result: Dict[str, Any]):
        """Mark a request as completed."""
        async with self._lock:
            if request_id in self.active_requests:
                context = self.active_requests.pop(request_id)
                context.total_time_ms = (time.time() - context.timestamp) * 1000
                
                self.completed_requests.append({
                    "context": context,
                    "result": result
                })
                
                # Keep only recent completed requests
                if len(self.completed_requests) > 1000:
                    self.completed_requests = self.completed_requests[-500:]


class IntelligentCacheManager:
    """Intelligent caching with prediction and optimization."""
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.feature_cache = {}
        self.prediction_cache = {}
        self.access_patterns = []
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "predictions": 0
        }
        
    async def get_cached_features(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached features if available."""
        if cache_key in self.feature_cache:
            self.cache_stats["hits"] += 1
            self.access_patterns.append({
                "key": cache_key,
                "timestamp": time.time(),
                "type": "hit"
            })
            return self.feature_cache[cache_key]
        
        self.cache_stats["misses"] += 1
        return None
    
    async def cache_features(self, cache_key: str, features: Dict[str, Any]):
        """Cache computed features."""
        self.feature_cache[cache_key] = {
            "data": features,
            "timestamp": time.time(),
            "access_count": 1
        }
        
        self.access_patterns.append({
            "key": cache_key,
            "timestamp": time.time(),
            "type": "store"
        })
        
        # Intelligent cache eviction
        await self._manage_cache_size()
    
    async def _manage_cache_size(self):
        """Manage cache size with intelligent eviction."""
        max_cache_size = 100  # Maximum number of cached items
        
        if len(self.feature_cache) > max_cache_size:
            # Evict based on access patterns and predictions
            candidates = []
            current_time = time.time()
            
            for key, data in self.feature_cache.items():
                score = self._calculate_eviction_score(key, data, current_time)
                candidates.append((score, key))
            
            # Sort by eviction score (higher = more likely to evict)
            candidates.sort(reverse=True)
            
            # Evict top candidates
            evict_count = len(self.feature_cache) - max_cache_size
            for _, key in candidates[:evict_count]:
                del self.feature_cache[key]
                self.cache_stats["evictions"] += 1
    
    def _calculate_eviction_score(self, key: str, data: Dict[str, Any], current_time: float) -> float:
        """Calculate eviction score (higher = more likely to evict)."""
        age_hours = (current_time - data["timestamp"]) / 3600
        access_count = data.get("access_count", 1)
        
        # Base score on age and inverse of access frequency
        score = age_hours / max(access_count, 1)
        
        # Add prediction component
        if self.config.enable_intelligent_caching:
            predicted_access = self._predict_future_access(key)
            score *= (1.0 - predicted_access)
        
        return score
    
    def _predict_future_access(self, key: str) -> float:
        """Predict likelihood of future access."""
        # Simple prediction based on recent access patterns
        recent_accesses = [
            p for p in self.access_patterns[-self.config.cache_prediction_window:]
            if p["key"] == key and p["timestamp"] > time.time() - 3600
        ]
        
        if not recent_accesses:
            return 0.0
        
        # Higher access frequency = higher prediction
        access_rate = len(recent_accesses) / min(3600, time.time() - recent_accesses[0]["timestamp"])
        return min(1.0, access_rate * 10)  # Normalize to 0-1


class PerformanceOptimizer:
    """Intelligent performance optimization."""
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.optimization_history = []
        self.performance_baseline = None
        self.current_optimizations = {}
        self.last_optimization_time = 0.0
        
    async def should_optimize(self, system_state: SystemState) -> bool:
        """Determine if optimization should be triggered."""
        if system_state.optimization_in_progress:
            return False
        
        current_time = time.time()
        
        # Time-based optimization
        time_since_last = current_time - self.last_optimization_time
        if time_since_last < 300:  # Minimum 5 minutes between optimizations
            return False
        
        # Performance-based triggers
        performance_triggers = [
            system_state.current_latency_p95_ms > self.config.target_latency_p95_ms * 1.2,
            system_state.current_throughput_fps < self.config.target_throughput_fps * 0.8,
            system_state.memory_usage_mb > self.config.memory_limit_mb * 0.9
        ]
        
        return any(performance_triggers)
    
    async def optimize_system(self, system_state: SystemState) -> Dict[str, Any]:
        """Perform intelligent system optimization."""
        logger.info("Starting intelligent system optimization")
        optimization_start = time.time()
        
        optimizations_applied = {}
        
        try:
            # Memory optimization
            if system_state.memory_usage_mb > self.config.memory_limit_mb * 0.8:
                memory_opts = await self._optimize_memory_usage(system_state)
                optimizations_applied.update(memory_opts)
            
            # Latency optimization
            if system_state.current_latency_p95_ms > self.config.target_latency_p95_ms:
                latency_opts = await self._optimize_latency(system_state)
                optimizations_applied.update(latency_opts)
            
            # Throughput optimization
            if system_state.current_throughput_fps < self.config.target_throughput_fps:
                throughput_opts = await self._optimize_throughput(system_state)
                optimizations_applied.update(throughput_opts)
            
            # Apply adaptive parameters
            adaptive_opts = await self._optimize_adaptive_parameters(system_state)
            optimizations_applied.update(adaptive_opts)
            
            self.last_optimization_time = time.time()
            optimization_duration = (self.last_optimization_time - optimization_start) * 1000
            
            optimization_result = {
                "optimizations_applied": optimizations_applied,
                "optimization_duration_ms": optimization_duration,
                "timestamp": self.last_optimization_time,
                "system_state_before": {
                    "latency_p95_ms": system_state.current_latency_p95_ms,
                    "throughput_fps": system_state.current_throughput_fps,
                    "memory_usage_mb": system_state.memory_usage_mb
                }
            }
            
            self.optimization_history.append(optimization_result)
            logger.info(f"System optimization completed in {optimization_duration:.1f}ms")
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"System optimization failed: {e}")
            return {"error": str(e), "optimizations_applied": {}}
    
    async def _optimize_memory_usage(self, system_state: SystemState) -> Dict[str, Any]:
        """Optimize memory usage."""
        optimizations = {}
        
        # Cache optimization
        optimizations["cache_cleanup"] = True
        optimizations["aggressive_gc"] = True
        
        # Batch size reduction
        if system_state.memory_usage_mb > self.config.memory_limit_mb * 0.9:
            optimizations["reduce_batch_size"] = True
            optimizations["new_max_batch_size"] = max(1, self.config.max_batch_size - 2)
        
        return optimizations
    
    async def _optimize_latency(self, system_state: SystemState) -> Dict[str, Any]:
        """Optimize for lower latency."""
        optimizations = {}
        
        # Reduce quality for speed
        if system_state.current_latency_p95_ms > self.config.target_latency_p95_ms * 1.5:
            optimizations["reduce_quality"] = True
            optimizations["new_image_size"] = (224, 224)  # Smaller input size
        
        # Enable more aggressive caching
        optimizations["increase_cache_size"] = True
        optimizations["enable_prefetch"] = True
        
        return optimizations
    
    async def _optimize_throughput(self, system_state: SystemState) -> Dict[str, Any]:
        """Optimize for higher throughput."""
        optimizations = {}
        
        # Increase batch size if memory allows
        if system_state.memory_usage_mb < self.config.memory_limit_mb * 0.7:
            optimizations["increase_batch_size"] = True
            optimizations["new_max_batch_size"] = min(16, self.config.max_batch_size + 2)
        
        # Reduce batch timeout for faster processing
        optimizations["reduce_batch_timeout"] = True
        optimizations["new_batch_timeout_ms"] = max(20, self.config.batch_timeout_ms * 0.8)
        
        return optimizations
    
    async def _optimize_adaptive_parameters(self, system_state: SystemState) -> Dict[str, Any]:
        """Optimize adaptive parameters based on current conditions."""
        optimizations = {}
        
        # Adjust based on request patterns
        if system_state.active_requests > self.config.max_concurrent_requests * 0.8:
            optimizations["increase_concurrency"] = True
            optimizations["new_max_concurrent"] = min(8, self.config.max_concurrent_requests + 1)
        
        # Adjust timeouts based on performance
        if system_state.current_latency_p95_ms < self.config.target_latency_p95_ms * 0.7:
            optimizations["reduce_timeout"] = True
            optimizations["new_timeout_seconds"] = max(10, self.config.request_timeout_seconds * 0.9)
        
        return optimizations


class IntelligentOrchestrator:
    """Main orchestrator coordinating all FastVLM components."""
    
    def __init__(self, config: OrchestratorConfig = None):
        self.config = config or OrchestratorConfig()
        self.system_state = SystemState()
        
        # Initialize components
        self._initialize_components()
        
        # Orchestrator state
        self.is_running = False
        self.request_counter = 0
        self.start_time = None
        self._shutdown_event = asyncio.Event()
        
    def _initialize_components(self):
        """Initialize orchestrator components."""
        logger.info("Initializing intelligent orchestrator components")
        
        # Core components
        pipeline_config = PipelineConfig(
            model_path=self.config.model_path,
            target_latency_ms=self.config.target_latency_p95_ms,
            memory_limit_mb=self.config.memory_limit_mb
        )
        self.pipeline = FastVLMPipeline(pipeline_config)
        
        # Mobile optimization
        self.mobile_optimizer = MobileOptimizer(self.config.mobile_optimization)
        
        # Reliability engine
        self.reliability_engine = ReliabilityEngine()
        
        # Intelligent subsystems
        self.request_scheduler = IntelligentRequestScheduler(self.config)
        self.cache_manager = IntelligentCacheManager(self.config)
        self.performance_optimizer = PerformanceOptimizer(self.config)
        
        # Monitoring
        self.metrics_collector = MetricsCollector()
        self.performance_profiler = PerformanceProfiler(
            self.metrics_collector, 
            "orchestrator"
        )
        
    async def start(self):
        """Start the orchestrator."""
        if self.is_running:
            return
        
        logger.info("Starting FastVLM intelligent orchestrator")
        self.start_time = time.time()
        self.is_running = True
        self.system_state.mode = OrchestratorMode.ACTIVE
        
        # Initialize reliability engine
        self.reliability_engine.initialize()
        
        # Start background tasks
        background_tasks = [
            asyncio.create_task(self._orchestration_loop()),
            asyncio.create_task(self._optimization_loop()),
            asyncio.create_task(self._monitoring_loop())
        ]
        
        logger.info("Orchestrator started successfully")
        
        # Wait for shutdown
        await self._shutdown_event.wait()
        
        # Cancel background tasks
        for task in background_tasks:
            task.cancel()
        
        await asyncio.gather(*background_tasks, return_exceptions=True)
    
    async def stop(self):
        """Stop the orchestrator."""
        if not self.is_running:
            return
        
        logger.info("Stopping orchestrator")
        self.is_running = False
        self.system_state.mode = OrchestratorMode.SHUTDOWN
        
        # Shutdown components
        self.reliability_engine.shutdown()
        self.mobile_optimizer.cleanup()
        
        # Signal shutdown
        self._shutdown_event.set()
        
    async def process_request(self, image_data: Any, question: str, 
                            context: RequestContext = None) -> Dict[str, Any]:
        """Process a request through the intelligent orchestrator."""
        if not self.is_running:
            raise RuntimeError("Orchestrator not running")
        
        if context is None:
            context = RequestContext()
        
        self.request_counter += 1
        logger.info(f"Processing request {self.request_counter} (ID: {context.request_id[:8]})")
        
        # Submit to scheduler
        request_id = await self.request_scheduler.submit_request(image_data, question, context)
        
        # Process through pipeline with reliability
        try:
            with self.reliability_engine.reliability_context("orchestrator"):
                result = await self.pipeline.process(image_data, question)
                
            # Mark as completed
            await self.request_scheduler.mark_request_completed(request_id, result)
            
            # Update system metrics
            self._update_system_metrics(context, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Request processing failed: {e}")
            error_result = {
                "error": str(e),
                "request_id": request_id,
                "answer": "I apologize, but I encountered an error processing your request."
            }
            
            await self.request_scheduler.mark_request_completed(request_id, error_result)
            return error_result
    
    async def _orchestration_loop(self):
        """Main orchestration loop."""
        logger.info("Starting orchestration loop")
        
        while self.is_running:
            try:
                await self._orchestration_cycle()
                await asyncio.sleep(1)  # Run every second
            except Exception as e:
                logger.error(f"Orchestration cycle error: {e}")
                await asyncio.sleep(5)  # Wait longer on error
    
    async def _orchestration_cycle(self):
        """Single orchestration cycle."""
        # Update system state
        self._update_system_state()
        
        # Process batch requests if available
        batch = await self.request_scheduler.get_next_batch()
        if batch:
            logger.debug(f"Processing batch of {len(batch)} requests")
            # Batch processing would be implemented here
    
    async def _optimization_loop(self):
        """Optimization loop."""
        logger.info("Starting optimization loop")
        
        while self.is_running:
            try:
                if await self.performance_optimizer.should_optimize(self.system_state):
                    self.system_state.optimization_in_progress = True
                    
                    optimization_result = await self.performance_optimizer.optimize_system(
                        self.system_state
                    )
                    
                    logger.info(f"Applied optimizations: {list(optimization_result.get('optimizations_applied', {}).keys())}")
                    
                    self.system_state.optimization_in_progress = False
                    self.system_state.last_optimization = time.time()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                self.system_state.optimization_in_progress = False
                await asyncio.sleep(60)
    
    async def _monitoring_loop(self):
        """Monitoring loop."""
        logger.info("Starting monitoring loop")
        
        while self.is_running:
            try:
                # Collect metrics
                metrics = self._collect_system_metrics()
                
                # Update system health
                health_report = self.reliability_engine.get_reliability_report()
                self.system_state.health_status = HealthStatus(
                    health_report["system_health"]["overall_status"]
                )
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(30)
    
    def _update_system_state(self):
        """Update current system state."""
        # This would update with real metrics in production
        self.system_state.active_requests = len(self.request_scheduler.active_requests)
        self.system_state.queued_requests = self.request_scheduler.request_queue.qsize()
    
    def _update_system_metrics(self, context: RequestContext, result: Dict[str, Any]):
        """Update system metrics with request results."""
        if "metrics" in result:
            metrics = result["metrics"]
            self.system_state.current_latency_p95_ms = metrics.get("total_latency_ms", 0)
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics."""
        uptime_seconds = time.time() - (self.start_time or time.time())
        
        return {
            "uptime_seconds": uptime_seconds,
            "total_requests": self.request_counter,
            "requests_per_second": self.request_counter / max(1, uptime_seconds),
            "system_state": {
                "mode": self.system_state.mode.value,
                "health_status": self.system_state.health_status.value,
                "active_requests": self.system_state.active_requests,
                "queued_requests": self.system_state.queued_requests
            },
            "cache_stats": self.cache_manager.cache_stats,
            "optimization_history": len(self.performance_optimizer.optimization_history)
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "orchestrator": {
                "is_running": self.is_running,
                "mode": self.system_state.mode.value,
                "health_status": self.system_state.health_status.value,
                "uptime_seconds": time.time() - (self.start_time or time.time()),
                "total_requests": self.request_counter
            },
            "components": {
                "pipeline_initialized": self.pipeline.is_initialized,
                "reliability_engine": self.reliability_engine.is_initialized,
                "mobile_optimizer": "active",
                "cache_manager": "active"
            },
            "performance": {
                "current_latency_p95_ms": self.system_state.current_latency_p95_ms,
                "current_throughput_fps": self.system_state.current_throughput_fps,
                "memory_usage_mb": self.system_state.memory_usage_mb,
                "active_requests": self.system_state.active_requests,
                "queued_requests": self.system_state.queued_requests
            },
            "optimization": {
                "last_optimization": self.system_state.last_optimization,
                "optimization_in_progress": self.system_state.optimization_in_progress,
                "optimization_count": len(self.performance_optimizer.optimization_history)
            }
        }