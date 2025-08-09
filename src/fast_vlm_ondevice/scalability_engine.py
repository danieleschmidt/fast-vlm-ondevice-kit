"""
Scalability engine for FastVLM on-device deployment.

Implements advanced scaling capabilities including dynamic load balancing,
resource pooling, auto-scaling, and distributed processing optimization.
"""

import asyncio
import logging
import time
import threading
import multiprocessing as mp
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import weakref
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import asynccontextmanager
import queue
import heapq

logger = logging.getLogger(__name__)


class ScalingMode(Enum):
    """Scaling operation modes."""
    SINGLE_INSTANCE = "single_instance"
    MULTI_THREADED = "multi_threaded"
    MULTI_PROCESS = "multi_process"
    HYBRID = "hybrid"
    DISTRIBUTED = "distributed"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_RESPONSE_TIME = "weighted_response_time"
    RESOURCE_AWARE = "resource_aware"
    INTELLIGENT = "intelligent"


class ResourceType(Enum):
    """Resource types for pooling."""
    CPU_CORE = "cpu_core"
    GPU_COMPUTE = "gpu_compute"
    NEURAL_ENGINE = "neural_engine"
    MEMORY_BLOCK = "memory_block"
    CACHE_PARTITION = "cache_partition"


@dataclass
class ScalabilityConfig:
    """Configuration for scalability engine."""
    # Scaling settings
    scaling_mode: ScalingMode = ScalingMode.HYBRID
    min_workers: int = 1
    max_workers: int = 8
    target_cpu_utilization: float = 0.75
    scale_up_threshold: float = 0.85
    scale_down_threshold: float = 0.30
    
    # Load balancing
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.INTELLIGENT
    health_check_interval: float = 10.0
    redistribute_threshold: float = 0.20
    
    # Resource pooling
    enable_resource_pooling: bool = True
    pool_size_cpu: int = 4
    pool_size_memory_mb: int = 2000
    pool_warmup_time: float = 5.0
    
    # Performance optimization
    enable_request_coalescing: bool = True
    coalescing_window_ms: float = 25.0
    enable_predictive_scaling: bool = True
    prediction_window_minutes: int = 5
    
    # Quality of Service
    priority_queue_enabled: bool = True
    max_queue_size: int = 1000
    request_timeout_seconds: float = 30.0
    circuit_breaker_enabled: bool = True


@dataclass
class WorkerNode:
    """Represents a worker node in the scaling system."""
    worker_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    worker_type: str = "thread"  # thread, process, distributed
    status: str = "idle"  # idle, busy, overloaded, failed
    
    # Performance metrics
    current_load: float = 0.0
    request_count: int = 0
    average_response_time_ms: float = 0.0
    error_rate: float = 0.0
    
    # Resource usage
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    last_health_check: float = field(default_factory=time.time)
    
    # Capacity
    max_concurrent_requests: int = 4
    current_requests: int = 0
    
    # Worker-specific data
    worker_instance: Optional[Any] = None
    creation_time: float = field(default_factory=time.time)


@dataclass
class ResourcePool:
    """Resource pool for efficient resource management."""
    resource_type: ResourceType
    pool_size: int
    available_resources: List[Any] = field(default_factory=list)
    allocated_resources: Dict[str, Any] = field(default_factory=dict)
    allocation_stats: Dict[str, int] = field(default_factory=dict)
    creation_time: float = field(default_factory=time.time)


class IntelligentLoadBalancer:
    """Intelligent load balancer with multiple strategies."""
    
    def __init__(self, config: ScalabilityConfig):
        self.config = config
        self.workers = {}
        self.strategy = config.load_balancing_strategy
        self.round_robin_index = 0
        self.performance_history = {}
        self._lock = threading.Lock()
        
    def register_worker(self, worker: WorkerNode):
        """Register a new worker."""
        with self._lock:
            self.workers[worker.worker_id] = worker
            self.performance_history[worker.worker_id] = []
            
        logger.info(f"Registered worker {worker.worker_id[:8]} ({worker.worker_type})")
    
    def unregister_worker(self, worker_id: str):
        """Unregister a worker."""
        with self._lock:
            if worker_id in self.workers:
                del self.workers[worker_id]
                if worker_id in self.performance_history:
                    del self.performance_history[worker_id]
                    
        logger.info(f"Unregistered worker {worker_id[:8]}")
    
    def select_worker(self, request_context: Dict[str, Any] = None) -> Optional[WorkerNode]:
        """Select the best worker for a request."""
        with self._lock:
            available_workers = [
                worker for worker in self.workers.values()
                if worker.status in ["idle", "busy"] and 
                   worker.current_requests < worker.max_concurrent_requests
            ]
            
            if not available_workers:
                return None
            
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._round_robin_selection(available_workers)
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._least_connections_selection(available_workers)
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_RESPONSE_TIME:
                return self._weighted_response_time_selection(available_workers)
            elif self.strategy == LoadBalancingStrategy.RESOURCE_AWARE:
                return self._resource_aware_selection(available_workers)
            else:  # INTELLIGENT
                return self._intelligent_selection(available_workers, request_context)
    
    def _round_robin_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Round-robin worker selection."""
        worker = workers[self.round_robin_index % len(workers)]
        self.round_robin_index += 1
        return worker
    
    def _least_connections_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker with least connections."""
        return min(workers, key=lambda w: w.current_requests)
    
    def _weighted_response_time_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker with best response time."""
        # Weight by inverse of response time
        weights = []
        for worker in workers:
            response_time = max(1.0, worker.average_response_time_ms)
            weight = 1.0 / response_time
            weights.append((weight, worker))
        
        # Select highest weight
        return max(weights, key=lambda x: x[0])[1]
    
    def _resource_aware_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker based on resource availability."""
        # Score based on available resources
        scores = []
        for worker in workers:
            cpu_score = 1.0 - worker.cpu_usage
            memory_score = 1.0 - (worker.memory_usage_mb / 1000.0)  # Normalize to GB
            load_score = 1.0 - worker.current_load
            
            total_score = (cpu_score + memory_score + load_score) / 3.0
            scores.append((total_score, worker))
        
        return max(scores, key=lambda x: x[0])[1]
    
    def _intelligent_selection(self, workers: List[WorkerNode], 
                             request_context: Dict[str, Any] = None) -> WorkerNode:
        """Intelligent worker selection using ML-based approach."""
        # Combine multiple factors with intelligent weighting
        scores = []
        current_time = time.time()
        
        for worker in workers:
            # Base metrics
            load_score = 1.0 - worker.current_load
            response_score = 1.0 / max(1.0, worker.average_response_time_ms / 100)
            error_score = 1.0 - worker.error_rate
            
            # Resource utilization
            resource_score = (1.0 - worker.cpu_usage) * 0.5 + \
                           (1.0 - min(1.0, worker.memory_usage_mb / 1000)) * 0.5
            
            # Historical performance
            history_score = self._calculate_history_score(worker.worker_id, current_time)
            
            # Request type affinity (if context provided)
            affinity_score = self._calculate_affinity_score(worker, request_context)
            
            # Weighted combination
            total_score = (
                load_score * 0.25 +
                response_score * 0.20 +
                error_score * 0.15 +
                resource_score * 0.20 +
                history_score * 0.10 +
                affinity_score * 0.10
            )
            
            scores.append((total_score, worker))
        
        return max(scores, key=lambda x: x[0])[1]
    
    def _calculate_history_score(self, worker_id: str, current_time: float) -> float:
        """Calculate historical performance score."""
        history = self.performance_history.get(worker_id, [])
        if not history:
            return 0.5  # Neutral score for new workers
        
        # Consider recent performance (last 10 minutes)
        recent_history = [
            h for h in history 
            if current_time - h.get("timestamp", 0) < 600
        ]
        
        if not recent_history:
            return 0.5
        
        # Average success rate
        success_rates = [h.get("success_rate", 1.0) for h in recent_history]
        return sum(success_rates) / len(success_rates)
    
    def _calculate_affinity_score(self, worker: WorkerNode, 
                                request_context: Dict[str, Any] = None) -> float:
        """Calculate request-worker affinity score."""
        if not request_context:
            return 0.5
        
        # Simple affinity based on request type
        request_type = request_context.get("type", "general")
        
        # GPU workers better for image processing
        if request_type == "image_heavy" and "gpu" in worker.worker_type:
            return 0.8
        # CPU workers better for text processing
        elif request_type == "text_heavy" and "cpu" in worker.worker_type:
            return 0.8
        
        return 0.5  # Neutral
    
    def update_worker_metrics(self, worker_id: str, metrics: Dict[str, Any]):
        """Update worker performance metrics."""
        with self._lock:
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                
                # Update current metrics
                worker.current_load = metrics.get("current_load", worker.current_load)
                worker.average_response_time_ms = metrics.get("response_time_ms", worker.average_response_time_ms)
                worker.error_rate = metrics.get("error_rate", worker.error_rate)
                worker.cpu_usage = metrics.get("cpu_usage", worker.cpu_usage)
                worker.memory_usage_mb = metrics.get("memory_usage_mb", worker.memory_usage_mb)
                worker.current_requests = metrics.get("current_requests", worker.current_requests)
                worker.last_health_check = time.time()
                
                # Add to performance history
                history_entry = {
                    "timestamp": time.time(),
                    "response_time_ms": worker.average_response_time_ms,
                    "error_rate": worker.error_rate,
                    "success_rate": 1.0 - worker.error_rate,
                    "cpu_usage": worker.cpu_usage,
                    "memory_usage_mb": worker.memory_usage_mb
                }
                
                self.performance_history[worker_id].append(history_entry)
                
                # Keep only recent history
                cutoff_time = time.time() - 3600  # Keep last hour
                self.performance_history[worker_id] = [
                    h for h in self.performance_history[worker_id] 
                    if h["timestamp"] > cutoff_time
                ]


class AutoScaler:
    """Automatic scaling system for worker management."""
    
    def __init__(self, config: ScalabilityConfig, load_balancer: IntelligentLoadBalancer):
        self.config = config
        self.load_balancer = load_balancer
        self.scaling_history = []
        self.last_scale_action = 0.0
        self.prediction_model = None
        self.monitoring_active = False
        
    def start_monitoring(self):
        """Start auto-scaling monitoring."""
        if self.monitoring_active:
            return
            
        logger.info("Starting auto-scaling monitoring")
        self.monitoring_active = True
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop auto-scaling monitoring."""
        logger.info("Stopping auto-scaling monitoring")
        self.monitoring_active = False
        
    def _monitoring_loop(self):
        """Auto-scaling monitoring loop."""
        while self.monitoring_active:
            try:
                self._evaluate_scaling_decision()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Auto-scaling monitoring error: {e}")
                time.sleep(60)
    
    def _evaluate_scaling_decision(self):
        """Evaluate if scaling action is needed."""
        current_time = time.time()
        
        # Prevent rapid scaling changes
        if current_time - self.last_scale_action < 300:  # 5 minutes cooldown
            return
        
        metrics = self._collect_system_metrics()
        
        # Scale up conditions
        if self._should_scale_up(metrics):
            self._scale_up()
        # Scale down conditions
        elif self._should_scale_down(metrics):
            self._scale_down()
    
    def _should_scale_up(self, metrics: Dict[str, Any]) -> bool:
        """Determine if scale up is needed."""
        conditions = [
            metrics["average_cpu_utilization"] > self.config.scale_up_threshold,
            metrics["queue_size"] > self.config.max_queue_size * 0.8,
            metrics["average_response_time_ms"] > 500,  # High latency
            metrics["active_workers"] < self.config.max_workers
        ]
        
        # Need at least 2 conditions for scale up
        return sum(conditions) >= 2
    
    def _should_scale_down(self, metrics: Dict[str, Any]) -> bool:
        """Determine if scale down is needed."""
        conditions = [
            metrics["average_cpu_utilization"] < self.config.scale_down_threshold,
            metrics["queue_size"] == 0,
            metrics["average_response_time_ms"] < 200,  # Low latency
            metrics["active_workers"] > self.config.min_workers
        ]
        
        # Need all conditions for scale down
        return all(conditions)
    
    def _scale_up(self):
        """Scale up by adding workers."""
        current_workers = len(self.load_balancer.workers)
        
        if current_workers >= self.config.max_workers:
            return
        
        logger.info("Scaling up - adding worker")
        
        # Create new worker
        new_worker = WorkerNode(
            worker_type="thread",
            status="idle",
            max_concurrent_requests=4
        )
        
        # Initialize worker (placeholder)
        new_worker.worker_instance = self._create_worker_instance()
        
        # Register with load balancer
        self.load_balancer.register_worker(new_worker)
        
        # Record scaling action
        self._record_scaling_action("scale_up", current_workers + 1)
        
    def _scale_down(self):
        """Scale down by removing workers."""
        current_workers = len(self.load_balancer.workers)
        
        if current_workers <= self.config.min_workers:
            return
        
        # Find least utilized worker
        least_utilized = min(
            self.load_balancer.workers.values(),
            key=lambda w: w.current_requests
        )
        
        if least_utilized.current_requests == 0:  # Only scale down idle workers
            logger.info(f"Scaling down - removing worker {least_utilized.worker_id[:8]}")
            
            # Gracefully shutdown worker
            self._shutdown_worker(least_utilized)
            
            # Unregister from load balancer
            self.load_balancer.unregister_worker(least_utilized.worker_id)
            
            # Record scaling action
            self._record_scaling_action("scale_down", current_workers - 1)
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system metrics for scaling decisions."""
        workers = list(self.load_balancer.workers.values())
        
        if not workers:
            return {
                "active_workers": 0,
                "average_cpu_utilization": 0.0,
                "average_response_time_ms": 0.0,
                "queue_size": 0,
                "total_requests": 0
            }
        
        # Calculate averages
        total_cpu = sum(w.cpu_usage for w in workers)
        total_response_time = sum(w.average_response_time_ms for w in workers)
        total_requests = sum(w.current_requests for w in workers)
        
        return {
            "active_workers": len(workers),
            "average_cpu_utilization": total_cpu / len(workers),
            "average_response_time_ms": total_response_time / len(workers),
            "queue_size": 0,  # Would be actual queue size in implementation
            "total_requests": total_requests
        }
    
    def _create_worker_instance(self) -> Any:
        """Create new worker instance."""
        # Placeholder for actual worker creation
        return {"type": "mock_worker", "created_at": time.time()}
    
    def _shutdown_worker(self, worker: WorkerNode):
        """Gracefully shutdown a worker."""
        # Placeholder for graceful shutdown
        if worker.worker_instance:
            logger.info(f"Shutting down worker {worker.worker_id[:8]}")
    
    def _record_scaling_action(self, action: str, new_count: int):
        """Record scaling action in history."""
        self.last_scale_action = time.time()
        
        scaling_event = {
            "timestamp": self.last_scale_action,
            "action": action,
            "worker_count": new_count,
            "trigger": "auto_scaling"
        }
        
        self.scaling_history.append(scaling_event)
        
        # Keep only recent history
        if len(self.scaling_history) > 100:
            self.scaling_history = self.scaling_history[-50:]
        
        logger.info(f"Scaling action recorded: {action} -> {new_count} workers")


class ResourcePoolManager:
    """Manages resource pools for efficient utilization."""
    
    def __init__(self, config: ScalabilityConfig):
        self.config = config
        self.resource_pools = {}
        self.allocation_stats = {}
        self._lock = threading.Lock()
        
    def initialize_pools(self):
        """Initialize resource pools."""
        logger.info("Initializing resource pools")
        
        # CPU core pool
        if self.config.pool_size_cpu > 0:
            self.create_resource_pool(
                ResourceType.CPU_CORE,
                self.config.pool_size_cpu
            )
        
        # Memory pool
        if self.config.pool_size_memory_mb > 0:
            self.create_resource_pool(
                ResourceType.MEMORY_BLOCK,
                self.config.pool_size_memory_mb // 100  # 100MB blocks
            )
        
        # Cache partition pool
        self.create_resource_pool(
            ResourceType.CACHE_PARTITION,
            8  # 8 cache partitions
        )
        
        logger.info(f"Initialized {len(self.resource_pools)} resource pools")
    
    def create_resource_pool(self, resource_type: ResourceType, pool_size: int):
        """Create a resource pool."""
        with self._lock:
            pool = ResourcePool(resource_type=resource_type, pool_size=pool_size)
            
            # Initialize resources
            for i in range(pool_size):
                resource = self._create_resource(resource_type, i)
                pool.available_resources.append(resource)
            
            self.resource_pools[resource_type] = pool
            self.allocation_stats[resource_type] = {
                "total_allocations": 0,
                "current_allocations": 0,
                "allocation_failures": 0
            }
        
        logger.info(f"Created {resource_type.value} pool with {pool_size} resources")
    
    def allocate_resource(self, resource_type: ResourceType, 
                         requester_id: str) -> Optional[Any]:
        """Allocate a resource from the pool."""
        with self._lock:
            if resource_type not in self.resource_pools:
                return None
            
            pool = self.resource_pools[resource_type]
            stats = self.allocation_stats[resource_type]
            
            if not pool.available_resources:
                stats["allocation_failures"] += 1
                return None
            
            # Allocate resource
            resource = pool.available_resources.pop(0)
            pool.allocated_resources[requester_id] = resource
            
            # Update stats
            stats["total_allocations"] += 1
            stats["current_allocations"] += 1
            
            logger.debug(f"Allocated {resource_type.value} to {requester_id[:8]}")
            return resource
    
    def release_resource(self, resource_type: ResourceType, requester_id: str):
        """Release a resource back to the pool."""
        with self._lock:
            if resource_type not in self.resource_pools:
                return
            
            pool = self.resource_pools[resource_type]
            stats = self.allocation_stats[resource_type]
            
            if requester_id in pool.allocated_resources:
                resource = pool.allocated_resources.pop(requester_id)
                pool.available_resources.append(resource)
                
                stats["current_allocations"] -= 1
                
                logger.debug(f"Released {resource_type.value} from {requester_id[:8]}")
    
    def _create_resource(self, resource_type: ResourceType, resource_id: int) -> Dict[str, Any]:
        """Create a resource instance."""
        return {
            "type": resource_type.value,
            "id": resource_id,
            "created_at": time.time(),
            "allocated_at": None,
            "usage_count": 0
        }
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get resource pool statistics."""
        with self._lock:
            stats = {}
            
            for resource_type, pool in self.resource_pools.items():
                stats[resource_type.value] = {
                    "total_resources": pool.pool_size,
                    "available_resources": len(pool.available_resources),
                    "allocated_resources": len(pool.allocated_resources),
                    "utilization": len(pool.allocated_resources) / pool.pool_size,
                    "allocation_stats": self.allocation_stats[resource_type]
                }
            
            return stats


class ScalabilityEngine:
    """Main scalability engine coordinating all scaling features."""
    
    def __init__(self, config: ScalabilityConfig = None):
        self.config = config or ScalabilityConfig()
        self.is_initialized = False
        
        # Initialize components
        self.load_balancer = IntelligentLoadBalancer(self.config)
        self.auto_scaler = AutoScaler(self.config, self.load_balancer)
        self.resource_pool_manager = ResourcePoolManager(self.config)
        
        # Scaling metrics
        self.scaling_metrics = {
            "initialization_time": 0.0,
            "total_requests_processed": 0,
            "average_response_time_ms": 0.0,
            "peak_throughput_fps": 0.0,
            "scaling_events": 0
        }
        
    def initialize(self):
        """Initialize the scalability engine."""
        if self.is_initialized:
            return
        
        logger.info("Initializing scalability engine")
        start_time = time.time()
        
        # Initialize resource pools
        if self.config.enable_resource_pooling:
            self.resource_pool_manager.initialize_pools()
        
        # Initialize workers
        self._initialize_initial_workers()
        
        # Start auto-scaling if enabled
        if self.config.enable_predictive_scaling:
            self.auto_scaler.start_monitoring()
        
        self.scaling_metrics["initialization_time"] = (time.time() - start_time) * 1000
        self.is_initialized = True
        
        logger.info(f"Scalability engine initialized in {self.scaling_metrics['initialization_time']:.1f}ms")
    
    def _initialize_initial_workers(self):
        """Initialize the minimum number of workers."""
        logger.info(f"Initializing {self.config.min_workers} initial workers")
        
        for i in range(self.config.min_workers):
            worker = WorkerNode(
                worker_type="thread",
                status="idle",
                max_concurrent_requests=4
            )
            
            # Initialize worker instance
            worker.worker_instance = self._create_worker_instance()
            
            # Register with load balancer
            self.load_balancer.register_worker(worker)
    
    def _create_worker_instance(self) -> Any:
        """Create a new worker instance."""
        # This would create actual worker instances in production
        return {
            "type": "fastvlm_worker",
            "created_at": time.time(),
            "status": "ready"
        }
    
    async def process_request(self, image_data: Any, question: str, 
                            context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process request through the scalable system."""
        if not self.is_initialized:
            raise RuntimeError("Scalability engine not initialized")
        
        request_start = time.time()
        
        # Select optimal worker
        worker = self.load_balancer.select_worker(context)
        if not worker:
            return {
                "error": "No available workers",
                "status": "overloaded"
            }
        
        try:
            # Allocate resources if needed
            allocated_resources = {}
            if self.config.enable_resource_pooling:
                allocated_resources = await self._allocate_request_resources(worker.worker_id)
            
            # Update worker load
            worker.current_requests += 1
            worker.status = "busy"
            
            # Process request (placeholder)
            result = await self._process_with_worker(worker, image_data, question, context)
            
            # Update metrics
            processing_time = (time.time() - request_start) * 1000
            self._update_request_metrics(worker.worker_id, processing_time, True)
            
            result["processing_time_ms"] = processing_time
            result["worker_id"] = worker.worker_id[:8]
            
            return result
            
        except Exception as e:
            logger.error(f"Request processing failed on worker {worker.worker_id[:8]}: {e}")
            processing_time = (time.time() - request_start) * 1000
            self._update_request_metrics(worker.worker_id, processing_time, False)
            
            return {
                "error": str(e),
                "processing_time_ms": processing_time,
                "worker_id": worker.worker_id[:8]
            }
        
        finally:
            # Release resources
            if allocated_resources:
                await self._release_request_resources(worker.worker_id, allocated_resources)
            
            # Update worker status
            worker.current_requests = max(0, worker.current_requests - 1)
            if worker.current_requests == 0:
                worker.status = "idle"
    
    async def _allocate_request_resources(self, worker_id: str) -> Dict[ResourceType, Any]:
        """Allocate resources for request processing."""
        allocated = {}
        
        # Allocate CPU core
        cpu_resource = self.resource_pool_manager.allocate_resource(
            ResourceType.CPU_CORE, worker_id
        )
        if cpu_resource:
            allocated[ResourceType.CPU_CORE] = cpu_resource
        
        # Allocate memory block
        memory_resource = self.resource_pool_manager.allocate_resource(
            ResourceType.MEMORY_BLOCK, worker_id
        )
        if memory_resource:
            allocated[ResourceType.MEMORY_BLOCK] = memory_resource
        
        return allocated
    
    async def _release_request_resources(self, worker_id: str, 
                                       allocated_resources: Dict[ResourceType, Any]):
        """Release allocated resources."""
        for resource_type in allocated_resources:
            self.resource_pool_manager.release_resource(resource_type, worker_id)
    
    async def _process_with_worker(self, worker: WorkerNode, image_data: Any, 
                                 question: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process request with specific worker."""
        # Simulate processing with worker
        await asyncio.sleep(0.2)  # 200ms simulated processing
        
        return {
            "answer": f"Processed by worker {worker.worker_id[:8]}",
            "confidence": 0.95,
            "worker_type": worker.worker_type,
            "status": "success"
        }
    
    def _update_request_metrics(self, worker_id: str, processing_time_ms: float, success: bool):
        """Update request processing metrics."""
        self.scaling_metrics["total_requests_processed"] += 1
        
        # Update worker metrics
        metrics = {
            "response_time_ms": processing_time_ms,
            "error_rate": 0.0 if success else 1.0,
            "current_load": 0.5  # Placeholder
        }
        
        self.load_balancer.update_worker_metrics(worker_id, metrics)
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get comprehensive scaling status."""
        return {
            "engine_status": {
                "initialized": self.is_initialized,
                "scaling_mode": self.config.scaling_mode.value,
                "workers_active": len(self.load_balancer.workers)
            },
            "load_balancing": {
                "strategy": self.config.load_balancing_strategy.value,
                "worker_distribution": self._get_worker_distribution()
            },
            "auto_scaling": {
                "monitoring_active": self.auto_scaler.monitoring_active,
                "scaling_events": len(self.auto_scaler.scaling_history),
                "min_workers": self.config.min_workers,
                "max_workers": self.config.max_workers
            },
            "resource_pools": self.resource_pool_manager.get_pool_statistics() if self.config.enable_resource_pooling else {},
            "performance_metrics": self.scaling_metrics
        }
    
    def _get_worker_distribution(self) -> Dict[str, Any]:
        """Get worker load distribution."""
        workers = list(self.load_balancer.workers.values())
        
        if not workers:
            return {"total_workers": 0}
        
        return {
            "total_workers": len(workers),
            "idle_workers": len([w for w in workers if w.status == "idle"]),
            "busy_workers": len([w for w in workers if w.status == "busy"]),
            "overloaded_workers": len([w for w in workers if w.status == "overloaded"]),
            "average_load": sum(w.current_load for w in workers) / len(workers),
            "average_response_time_ms": sum(w.average_response_time_ms for w in workers) / len(workers)
        }
    
    def shutdown(self):
        """Shutdown the scalability engine."""
        logger.info("Shutting down scalability engine")
        
        # Stop auto-scaling
        self.auto_scaler.stop_monitoring()
        
        # Shutdown all workers
        for worker_id in list(self.load_balancer.workers.keys()):
            worker = self.load_balancer.workers[worker_id]
            self.auto_scaler._shutdown_worker(worker)
            self.load_balancer.unregister_worker(worker_id)
        
        self.is_initialized = False
        logger.info("Scalability engine shutdown complete")