"""
High-Performance Distributed Computing Engine for FastVLM.

Provides advanced concurrency, auto-scaling, load balancing, and distributed inference
capabilities for production-scale deployment.
"""

import asyncio
import concurrent.futures
import threading
import multiprocessing
import queue
import time
import logging
import psutil
import uuid
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import weakref
from collections import defaultdict, deque
import heapq
import socket
import hashlib

logger = logging.getLogger(__name__)


class ComputeStrategy(Enum):
    """Compute distribution strategies."""
    SINGLE_THREADED = "single_threaded"
    MULTI_THREADED = "multi_threaded"
    MULTI_PROCESS = "multi_process"
    ASYNC_CONCURRENT = "async_concurrent"
    DISTRIBUTED = "distributed"
    HYBRID = "hybrid"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_RESPONSE_TIME = "weighted_response_time"
    RESOURCE_BASED = "resource_based"
    ADAPTIVE = "adaptive"


class ScalingPolicy(Enum):
    """Auto-scaling policies."""
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    QUEUE_DEPTH = "queue_depth"
    RESPONSE_TIME = "response_time"
    PREDICTIVE = "predictive"
    HYBRID_METRICS = "hybrid_metrics"


@dataclass
class WorkerNode:
    """Worker node configuration."""
    node_id: str
    host: str
    port: int
    capabilities: Dict[str, Any]
    current_load: float
    max_capacity: int
    active_tasks: int
    last_heartbeat: float
    health_status: str
    performance_metrics: Dict[str, float]


@dataclass
class TaskRequest:
    """Distributed task request."""
    task_id: str
    task_type: str
    priority: int
    payload: Dict[str, Any]
    timestamp: float
    timeout_ms: int
    required_capabilities: List[str]
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    throughput_rps: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    cpu_utilization: float
    memory_utilization: float
    gpu_utilization: float
    queue_depth: int
    error_rate: float
    cache_hit_rate: float


@dataclass
class ScalingConfig:
    """Auto-scaling configuration."""
    min_workers: int = 1
    max_workers: int = 10
    target_cpu_utilization: float = 0.7
    target_memory_utilization: float = 0.8
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    cooldown_period_seconds: int = 300
    predictive_scaling: bool = True


class IntelligentTaskScheduler:
    """Intelligent task scheduling with priority and resource awareness."""
    
    def __init__(self, max_queue_size: int = 10000):
        self.task_queue = []  # Priority heap
        self.pending_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        self.task_metrics = defaultdict(list)
        self.lock = threading.Lock()
        self.max_queue_size = max_queue_size
        
        logger.info(f"Task scheduler initialized with max queue size: {max_queue_size}")
    
    def submit_task(self, task: TaskRequest) -> bool:
        """Submit task to scheduler."""
        with self.lock:
            if len(self.task_queue) >= self.max_queue_size:
                logger.warning("Task queue full, rejecting task")
                return False
            
            # Use negative priority for max-heap behavior
            heapq.heappush(self.task_queue, (-task.priority, task.timestamp, task))
            self.pending_tasks[task.task_id] = task
            
            logger.debug(f"Task submitted: {task.task_id} (priority: {task.priority})")
            return True
    
    def get_next_task(self, worker_capabilities: List[str] = None) -> Optional[TaskRequest]:
        """Get next task matching worker capabilities."""
        with self.lock:
            if not self.task_queue:
                return None
            
            # Find first task that matches capabilities
            for i, (neg_priority, timestamp, task) in enumerate(self.task_queue):
                if worker_capabilities is None or self._task_matches_capabilities(task, worker_capabilities):
                    # Remove task from queue
                    del self.task_queue[i]
                    heapq.heapify(self.task_queue)
                    del self.pending_tasks[task.task_id]
                    return task
            
            return None
    
    def mark_task_completed(self, task_id: str, result: Dict[str, Any], duration_ms: float):
        """Mark task as completed."""
        with self.lock:
            if task_id in self.pending_tasks:
                task = self.pending_tasks.pop(task_id)
                self.completed_tasks[task_id] = {
                    "task": task,
                    "result": result,
                    "duration_ms": duration_ms,
                    "completed_at": time.time()
                }
                
                # Update metrics
                self.task_metrics[task.task_type].append(duration_ms)
                logger.debug(f"Task completed: {task_id} ({duration_ms:.1f}ms)")
    
    def mark_task_failed(self, task_id: str, error: str):
        """Mark task as failed."""
        with self.lock:
            if task_id in self.pending_tasks:
                task = self.pending_tasks.pop(task_id)
                
                # Retry if possible
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    heapq.heappush(self.task_queue, (-task.priority, time.time(), task))
                    self.pending_tasks[task.task_id] = task
                    logger.info(f"Task retry {task.retry_count}/{task.max_retries}: {task_id}")
                else:
                    # Max retries exceeded
                    self.failed_tasks[task_id] = {
                        "task": task,
                        "error": error,
                        "failed_at": time.time()
                    }
                    logger.error(f"Task failed permanently: {task_id} - {error}")
    
    def _task_matches_capabilities(self, task: TaskRequest, worker_capabilities: List[str]) -> bool:
        """Check if task requirements match worker capabilities."""
        if not task.required_capabilities:
            return True
        
        return all(cap in worker_capabilities for cap in task.required_capabilities)
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self.lock:
            return {
                "pending_tasks": len(self.task_queue),
                "completed_tasks": len(self.completed_tasks),
                "failed_tasks": len(self.failed_tasks),
                "task_types": list(self.task_metrics.keys()),
                "average_durations": {
                    task_type: sum(durations) / len(durations) if durations else 0
                    for task_type, durations in self.task_metrics.items()
                }
            }


class AdaptiveLoadBalancer:
    """Adaptive load balancer with multiple strategies."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE):
        self.strategy = strategy
        self.worker_nodes = {}
        self.connection_counts = defaultdict(int)
        self.response_times = defaultdict(deque)
        self.weights = defaultdict(float)
        self.round_robin_index = 0
        self.lock = threading.Lock()
        
        # Strategy weights for adaptive balancing
        self.strategy_weights = {
            LoadBalancingStrategy.ROUND_ROBIN: 0.2,
            LoadBalancingStrategy.LEAST_CONNECTIONS: 0.3,
            LoadBalancingStrategy.WEIGHTED_RESPONSE_TIME: 0.3,
            LoadBalancingStrategy.RESOURCE_BASED: 0.2
        }
        
        logger.info(f"Load balancer initialized with strategy: {strategy.value}")
    
    def register_worker(self, worker: WorkerNode):
        """Register new worker node."""
        with self.lock:
            self.worker_nodes[worker.node_id] = worker
            self.weights[worker.node_id] = 1.0
            logger.info(f"Worker registered: {worker.node_id} at {worker.host}:{worker.port}")
    
    def unregister_worker(self, node_id: str):
        """Unregister worker node."""
        with self.lock:
            if node_id in self.worker_nodes:
                del self.worker_nodes[node_id]
                del self.weights[node_id]
                self.connection_counts.pop(node_id, None)
                self.response_times.pop(node_id, None)
                logger.info(f"Worker unregistered: {node_id}")
    
    def select_worker(self, task: TaskRequest) -> Optional[WorkerNode]:
        """Select best worker for task based on strategy."""
        with self.lock:
            available_workers = [
                worker for worker in self.worker_nodes.values()
                if worker.health_status == "healthy" and 
                self._worker_can_handle_task(worker, task)
            ]
            
            if not available_workers:
                return None
            
            if self.strategy == LoadBalancingStrategy.ADAPTIVE:
                return self._adaptive_selection(available_workers, task)
            elif self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._round_robin_selection(available_workers)
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._least_connections_selection(available_workers)
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_RESPONSE_TIME:
                return self._weighted_response_time_selection(available_workers)
            elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
                return self._resource_based_selection(available_workers)
            else:
                return available_workers[0]  # Fallback
    
    def record_response_time(self, node_id: str, response_time_ms: float):
        """Record response time for worker node."""
        with self.lock:
            times = self.response_times[node_id]
            times.append(response_time_ms)
            
            # Keep only recent times (last 100)
            if len(times) > 100:
                times.popleft()
            
            # Update weight based on performance
            if times:
                avg_time = sum(times) / len(times)
                # Lower response time = higher weight
                self.weights[node_id] = max(0.1, 1000 / (avg_time + 100))
    
    def increment_connections(self, node_id: str):
        """Increment connection count for worker."""
        with self.lock:
            self.connection_counts[node_id] += 1
    
    def decrement_connections(self, node_id: str):
        """Decrement connection count for worker."""
        with self.lock:
            if self.connection_counts[node_id] > 0:
                self.connection_counts[node_id] -= 1
    
    def _worker_can_handle_task(self, worker: WorkerNode, task: TaskRequest) -> bool:
        """Check if worker can handle the task."""
        # Check capacity
        if worker.active_tasks >= worker.max_capacity:
            return False
        
        # Check capabilities
        if task.required_capabilities:
            worker_caps = worker.capabilities.get("supported_operations", [])
            if not all(cap in worker_caps for cap in task.required_capabilities):
                return False
        
        return True
    
    def _adaptive_selection(self, workers: List[WorkerNode], task: TaskRequest) -> WorkerNode:
        """Adaptive worker selection combining multiple strategies."""
        scores = {}
        
        for worker in workers:
            score = 0.0
            
            # Round robin component
            if self.strategy_weights[LoadBalancingStrategy.ROUND_ROBIN] > 0:
                score += self.strategy_weights[LoadBalancingStrategy.ROUND_ROBIN] * 0.5
            
            # Least connections component
            if self.strategy_weights[LoadBalancingStrategy.LEAST_CONNECTIONS] > 0:
                max_connections = max(self.connection_counts[w.node_id] for w in workers)
                if max_connections > 0:
                    connection_score = 1.0 - (self.connection_counts[worker.node_id] / max_connections)
                    score += self.strategy_weights[LoadBalancingStrategy.LEAST_CONNECTIONS] * connection_score
            
            # Response time component
            if self.strategy_weights[LoadBalancingStrategy.WEIGHTED_RESPONSE_TIME] > 0:
                weight = self.weights[worker.node_id]
                max_weight = max(self.weights[w.node_id] for w in workers)
                if max_weight > 0:
                    time_score = weight / max_weight
                    score += self.strategy_weights[LoadBalancingStrategy.WEIGHTED_RESPONSE_TIME] * time_score
            
            # Resource-based component
            if self.strategy_weights[LoadBalancingStrategy.RESOURCE_BASED] > 0:
                resource_score = 1.0 - worker.current_load
                score += self.strategy_weights[LoadBalancingStrategy.RESOURCE_BASED] * resource_score
            
            scores[worker.node_id] = score
        
        # Select worker with highest score
        best_worker_id = max(scores.keys(), key=lambda k: scores[k])
        return next(w for w in workers if w.node_id == best_worker_id)
    
    def _round_robin_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Round robin worker selection."""
        worker = workers[self.round_robin_index % len(workers)]
        self.round_robin_index += 1
        return worker
    
    def _least_connections_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker with least connections."""
        return min(workers, key=lambda w: self.connection_counts[w.node_id])
    
    def _weighted_response_time_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker based on weighted response time."""
        return max(workers, key=lambda w: self.weights[w.node_id])
    
    def _resource_based_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker based on current resource utilization."""
        return min(workers, key=lambda w: w.current_load)
    
    def get_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self.lock:
            return {
                "strategy": self.strategy.value,
                "worker_count": len(self.worker_nodes),
                "connection_counts": dict(self.connection_counts),
                "average_response_times": {
                    node_id: sum(times) / len(times) if times else 0
                    for node_id, times in self.response_times.items()
                },
                "worker_weights": dict(self.weights)
            }


class AutoScalingEngine:
    """Auto-scaling engine with predictive capabilities."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.current_workers = config.min_workers
        self.last_scale_time = 0
        self.metrics_history = deque(maxlen=1000)
        self.scaling_decisions = []
        self.predictive_model = None
        self.lock = threading.Lock()
        
        logger.info(f"Auto-scaling engine initialized - range: {config.min_workers}-{config.max_workers}")
    
    def evaluate_scaling(self, current_metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Evaluate if scaling is needed based on current metrics."""
        with self.lock:
            # Add metrics to history
            self.metrics_history.append({
                "timestamp": time.time(),
                "metrics": asdict(current_metrics)
            })
            
            # Check cooldown period
            current_time = time.time()
            if current_time - self.last_scale_time < self.config.cooldown_period_seconds:
                return {"action": "wait", "reason": "cooldown_period"}
            
            # Evaluate scaling decision
            decision = self._make_scaling_decision(current_metrics)
            
            # Record decision
            self.scaling_decisions.append({
                "timestamp": current_time,
                "decision": decision,
                "metrics": asdict(current_metrics)
            })
            
            return decision
    
    def _make_scaling_decision(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Make scaling decision based on metrics and policies."""
        scale_factors = []
        
        # CPU-based scaling
        if metrics.cpu_utilization > self.config.scale_up_threshold:
            scale_factors.append(("cpu_high", "scale_up", metrics.cpu_utilization))
        elif metrics.cpu_utilization < self.config.scale_down_threshold:
            scale_factors.append(("cpu_low", "scale_down", metrics.cpu_utilization))
        
        # Memory-based scaling
        if metrics.memory_utilization > self.config.scale_up_threshold:
            scale_factors.append(("memory_high", "scale_up", metrics.memory_utilization))
        elif metrics.memory_utilization < self.config.scale_down_threshold:
            scale_factors.append(("memory_low", "scale_down", metrics.memory_utilization))
        
        # Queue depth scaling
        if metrics.queue_depth > self.current_workers * 10:  # More than 10 tasks per worker
            scale_factors.append(("queue_high", "scale_up", metrics.queue_depth))
        elif metrics.queue_depth < self.current_workers * 2:  # Less than 2 tasks per worker
            scale_factors.append(("queue_low", "scale_down", metrics.queue_depth))
        
        # Response time scaling
        if metrics.latency_p95_ms > 1000:  # P95 latency > 1 second
            scale_factors.append(("latency_high", "scale_up", metrics.latency_p95_ms))
        elif metrics.latency_p95_ms < 200:  # P95 latency < 200ms
            scale_factors.append(("latency_low", "scale_down", metrics.latency_p95_ms))
        
        # Predictive scaling
        if self.config.predictive_scaling:
            predictive_factor = self._predictive_scaling_decision()
            if predictive_factor:
                scale_factors.append(predictive_factor)
        
        # Aggregate decisions
        scale_up_votes = sum(1 for _, action, _ in scale_factors if action == "scale_up")
        scale_down_votes = sum(1 for _, action, _ in scale_factors if action == "scale_down")
        
        if scale_up_votes > scale_down_votes and self.current_workers < self.config.max_workers:
            new_workers = min(self.current_workers + 1, self.config.max_workers)
            return {
                "action": "scale_up",
                "from_workers": self.current_workers,
                "to_workers": new_workers,
                "reasons": [reason for reason, action, _ in scale_factors if action == "scale_up"]
            }
        elif scale_down_votes > scale_up_votes and self.current_workers > self.config.min_workers:
            new_workers = max(self.current_workers - 1, self.config.min_workers)
            return {
                "action": "scale_down",
                "from_workers": self.current_workers,
                "to_workers": new_workers,
                "reasons": [reason for reason, action, _ in scale_factors if action == "scale_down"]
            }
        else:
            return {"action": "maintain", "workers": self.current_workers}
    
    def _predictive_scaling_decision(self) -> Optional[Tuple[str, str, float]]:
        """Make predictive scaling decision based on historical patterns."""
        if len(self.metrics_history) < 60:  # Need at least 1 minute of data
            return None
        
        # Simple trend analysis
        recent_metrics = list(self.metrics_history)[-60:]  # Last 60 data points
        cpu_trend = self._calculate_trend([m["metrics"]["cpu_utilization"] for m in recent_metrics])
        
        # If CPU is trending up significantly, proactively scale up
        if cpu_trend > 0.02:  # 2% increase per data point
            return ("predictive_cpu_trend", "scale_up", cpu_trend)
        elif cpu_trend < -0.02:  # 2% decrease per data point
            return ("predictive_cpu_trend", "scale_down", abs(cpu_trend))
        
        return None
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend in values."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def apply_scaling_decision(self, decision: Dict[str, Any]):
        """Apply scaling decision."""
        if decision["action"] in ["scale_up", "scale_down"]:
            with self.lock:
                self.current_workers = decision["to_workers"]
                self.last_scale_time = time.time()
                
                logger.info(f"Scaling {decision['action']}: {decision['from_workers']} -> {decision['to_workers']} workers")
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        with self.lock:
            recent_decisions = self.scaling_decisions[-20:]  # Last 20 decisions
            
            return {
                "current_workers": self.current_workers,
                "config": asdict(self.config),
                "recent_decisions": recent_decisions,
                "metrics_history_size": len(self.metrics_history),
                "last_scale_time": self.last_scale_time
            }


class HighPerformanceDistributedEngine:
    """Main distributed computing engine coordinating all components."""
    
    def __init__(self, compute_strategy: ComputeStrategy = ComputeStrategy.HYBRID,
                 scaling_config: ScalingConfig = None):
        self.compute_strategy = compute_strategy
        self.scaling_config = scaling_config or ScalingConfig()
        
        # Core components
        self.task_scheduler = IntelligentTaskScheduler()
        self.load_balancer = AdaptiveLoadBalancer()
        self.auto_scaler = AutoScalingEngine(self.scaling_config)
        
        # Worker management
        self.worker_pool = None
        self.async_executor = None
        self.process_pool = None
        
        # Performance monitoring
        self.performance_metrics = PerformanceMetrics(
            throughput_rps=0.0, latency_p50_ms=0.0, latency_p95_ms=0.0, latency_p99_ms=0.0,
            cpu_utilization=0.0, memory_utilization=0.0, gpu_utilization=0.0,
            queue_depth=0, error_rate=0.0, cache_hit_rate=0.0
        )
        
        # Execution tracking
        self.active_tasks = {}
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_processing_time = 0.0
        
        # Locks and synchronization
        self.metrics_lock = threading.Lock()
        self.task_lock = threading.Lock()
        
        self._initialize_compute_resources()
        
        logger.info(f"High-performance distributed engine initialized with strategy: {compute_strategy.value}")
    
    def _initialize_compute_resources(self):
        """Initialize compute resources based on strategy."""
        if self.compute_strategy in [ComputeStrategy.MULTI_THREADED, ComputeStrategy.HYBRID]:
            max_workers = min(32, (psutil.cpu_count() or 1) * 2)
            self.worker_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
            logger.info(f"Thread pool initialized with {max_workers} workers")
        
        if self.compute_strategy in [ComputeStrategy.MULTI_PROCESS, ComputeStrategy.HYBRID]:
            max_processes = min(psutil.cpu_count() or 1, 8)
            self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=max_processes)
            logger.info(f"Process pool initialized with {max_processes} workers")
        
        if self.compute_strategy in [ComputeStrategy.ASYNC_CONCURRENT, ComputeStrategy.HYBRID]:
            # Async executor will be created when needed
            pass
    
    async def submit_task_async(self, task_type: str, payload: Dict[str, Any], 
                               priority: int = 0, timeout_ms: int = 30000) -> Dict[str, Any]:
        """Submit task for asynchronous processing."""
        task = TaskRequest(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            priority=priority,
            payload=payload,
            timestamp=time.time(),
            timeout_ms=timeout_ms,
            required_capabilities=[task_type]
        )
        
        # Submit to scheduler
        if not self.task_scheduler.submit_task(task):
            raise RuntimeError("Task queue full")
        
        # Process task based on strategy
        start_time = time.time()
        try:
            result = await self._process_task_async(task)
            duration_ms = (time.time() - start_time) * 1000
            
            # Update metrics
            self._update_performance_metrics(duration_ms, success=True)
            self.task_scheduler.mark_task_completed(task.task_id, result, duration_ms)
            
            return {
                "task_id": task.task_id,
                "result": result,
                "duration_ms": duration_ms,
                "status": "completed"
            }
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._update_performance_metrics(duration_ms, success=False)
            self.task_scheduler.mark_task_failed(task.task_id, str(e))
            
            raise RuntimeError(f"Task failed: {e}")
    
    def submit_task_sync(self, task_type: str, payload: Dict[str, Any], 
                        priority: int = 0, timeout_ms: int = 30000) -> Dict[str, Any]:
        """Submit task for synchronous processing."""
        # Run async task in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.submit_task_async(task_type, payload, priority, timeout_ms)
            )
        finally:
            loop.close()
    
    async def _process_task_async(self, task: TaskRequest) -> Dict[str, Any]:
        """Process task using optimal compute strategy."""
        if self.compute_strategy == ComputeStrategy.SINGLE_THREADED:
            return await self._process_single_threaded(task)
        elif self.compute_strategy == ComputeStrategy.MULTI_THREADED:
            return await self._process_multi_threaded(task)
        elif self.compute_strategy == ComputeStrategy.MULTI_PROCESS:
            return await self._process_multi_process(task)
        elif self.compute_strategy == ComputeStrategy.ASYNC_CONCURRENT:
            return await self._process_async_concurrent(task)
        elif self.compute_strategy == ComputeStrategy.DISTRIBUTED:
            return await self._process_distributed(task)
        elif self.compute_strategy == ComputeStrategy.HYBRID:
            return await self._process_hybrid(task)
        else:
            raise ValueError(f"Unknown compute strategy: {self.compute_strategy}")
    
    async def _process_single_threaded(self, task: TaskRequest) -> Dict[str, Any]:
        """Process task in single thread."""
        return self._execute_task_logic(task)
    
    async def _process_multi_threaded(self, task: TaskRequest) -> Dict[str, Any]:
        """Process task using thread pool."""
        loop = asyncio.get_event_loop()
        future = self.worker_pool.submit(self._execute_task_logic, task)
        return await loop.run_in_executor(None, future.result)
    
    async def _process_multi_process(self, task: TaskRequest) -> Dict[str, Any]:
        """Process task using process pool."""
        loop = asyncio.get_event_loop()
        future = self.process_pool.submit(self._execute_task_logic, task)
        return await loop.run_in_executor(None, future.result)
    
    async def _process_async_concurrent(self, task: TaskRequest) -> Dict[str, Any]:
        """Process task using async concurrency."""
        # Simulate async processing
        await asyncio.sleep(0.001)  # Yield control
        return self._execute_task_logic(task)
    
    async def _process_distributed(self, task: TaskRequest) -> Dict[str, Any]:
        """Process task using distributed workers."""
        # Select best worker node
        worker = self.load_balancer.select_worker(task)
        if not worker:
            raise RuntimeError("No available workers")
        
        # Track connection
        self.load_balancer.increment_connections(worker.node_id)
        
        try:
            # Simulate distributed execution
            start_time = time.time()
            result = self._execute_task_logic(task)
            duration_ms = (time.time() - start_time) * 1000
            
            # Record performance
            self.load_balancer.record_response_time(worker.node_id, duration_ms)
            
            return result
            
        finally:
            self.load_balancer.decrement_connections(worker.node_id)
    
    async def _process_hybrid(self, task: TaskRequest) -> Dict[str, Any]:
        """Process task using hybrid strategy (intelligent selection)."""
        # Choose best strategy based on task characteristics
        payload_size = len(json.dumps(task.payload))
        
        if payload_size > 1000000:  # Large payload - use process pool
            return await self._process_multi_process(task)
        elif task.priority > 8:  # High priority - use dedicated thread
            return await self._process_multi_threaded(task)
        elif self.task_scheduler.get_queue_stats()["pending_tasks"] > 50:  # High load - distribute
            return await self._process_distributed(task)
        else:  # Default - async concurrent
            return await self._process_async_concurrent(task)
    
    def _execute_task_logic(self, task: TaskRequest) -> Dict[str, Any]:
        """Execute the actual task logic."""
        # Simulate different types of ML inference tasks
        if task.task_type == "vision_encoding":
            return self._simulate_vision_encoding(task.payload)
        elif task.task_type == "text_encoding":
            return self._simulate_text_encoding(task.payload)
        elif task.task_type == "multimodal_fusion":
            return self._simulate_multimodal_fusion(task.payload)
        elif task.task_type == "answer_generation":
            return self._simulate_answer_generation(task.payload)
        else:
            return {"result": f"Processed {task.task_type}", "confidence": 0.85}
    
    def _simulate_vision_encoding(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate vision encoding task."""
        # Simulate processing time based on image size
        image_size = payload.get("image_size", 224)
        processing_time = (image_size / 224) * 0.05  # Base 50ms for 224x224
        time.sleep(processing_time)
        
        return {
            "vision_features": f"vision_features_{hash(str(payload)) % 10000}",
            "feature_dim": 768,
            "confidence": 0.92
        }
    
    def _simulate_text_encoding(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate text encoding task."""
        # Simulate processing time based on text length
        text_length = len(payload.get("text", ""))
        processing_time = min(text_length / 1000 * 0.01, 0.1)  # Max 100ms
        time.sleep(processing_time)
        
        return {
            "text_features": f"text_features_{hash(str(payload)) % 10000}",
            "feature_dim": 512,
            "confidence": 0.88
        }
    
    def _simulate_multimodal_fusion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate multimodal fusion task."""
        time.sleep(0.03)  # 30ms fusion time
        
        return {
            "fused_features": f"fused_{hash(str(payload)) % 10000}",
            "feature_dim": 1024,
            "confidence": 0.90
        }
    
    def _simulate_answer_generation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate answer generation task."""
        time.sleep(0.08)  # 80ms generation time
        
        answers = [
            "I can see a person walking in the park.",
            "There are several objects on the table.",
            "The image shows a beautiful landscape.",
            "I notice some text in the image.",
            "This appears to be an indoor scene."
        ]
        
        return {
            "answer": answers[hash(str(payload)) % len(answers)],
            "confidence": 0.85,
            "tokens_generated": 12
        }
    
    def _update_performance_metrics(self, duration_ms: float, success: bool):
        """Update performance metrics."""
        with self.metrics_lock:
            if success:
                self.completed_tasks += 1
            else:
                self.failed_tasks += 1
            
            self.total_processing_time += duration_ms
            
            # Update latency metrics (simplified)
            total_tasks = self.completed_tasks + self.failed_tasks
            if total_tasks > 0:
                avg_latency = self.total_processing_time / total_tasks
                self.performance_metrics = PerformanceMetrics(
                    throughput_rps=total_tasks / max(time.time() - getattr(self, 'start_time', time.time()), 1),
                    latency_p50_ms=avg_latency * 0.8,
                    latency_p95_ms=avg_latency * 1.2,
                    latency_p99_ms=avg_latency * 1.5,
                    cpu_utilization=psutil.cpu_percent(),
                    memory_utilization=psutil.virtual_memory().percent,
                    gpu_utilization=0.0,  # Would need GPU monitoring
                    queue_depth=self.task_scheduler.get_queue_stats()["pending_tasks"],
                    error_rate=(self.failed_tasks / total_tasks) * 100,
                    cache_hit_rate=85.0  # Placeholder
                )
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance and operational metrics."""
        with self.metrics_lock:
            scheduler_stats = self.task_scheduler.get_queue_stats()
            balancer_stats = self.load_balancer.get_balancer_stats()
            scaling_stats = self.auto_scaler.get_scaling_stats()
            
            return {
                "performance": asdict(self.performance_metrics),
                "scheduler": scheduler_stats,
                "load_balancer": balancer_stats,
                "auto_scaling": scaling_stats,
                "compute_strategy": self.compute_strategy.value,
                "total_completed": self.completed_tasks,
                "total_failed": self.failed_tasks,
                "success_rate": (self.completed_tasks / max(self.completed_tasks + self.failed_tasks, 1)) * 100
            }
    
    def auto_scale_if_needed(self):
        """Check and apply auto-scaling if needed."""
        scaling_decision = self.auto_scaler.evaluate_scaling(self.performance_metrics)
        if scaling_decision["action"] in ["scale_up", "scale_down"]:
            self.auto_scaler.apply_scaling_decision(scaling_decision)
            logger.info(f"Auto-scaling applied: {scaling_decision}")
    
    def shutdown(self):
        """Gracefully shutdown the distributed engine."""
        logger.info("Shutting down distributed engine...")
        
        if self.worker_pool:
            self.worker_pool.shutdown(wait=True)
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        
        logger.info("Distributed engine shutdown complete")


# Factory functions for easy setup
def create_high_performance_engine(
    compute_strategy: ComputeStrategy = ComputeStrategy.HYBRID,
    min_workers: int = 1,
    max_workers: int = 10,
    target_cpu_utilization: float = 0.7
) -> HighPerformanceDistributedEngine:
    """Create high-performance distributed engine with common configurations."""
    scaling_config = ScalingConfig(
        min_workers=min_workers,
        max_workers=max_workers,
        target_cpu_utilization=target_cpu_utilization
    )
    
    return HighPerformanceDistributedEngine(compute_strategy, scaling_config)


# Example usage and demonstration
if __name__ == "__main__":
    async def demo_distributed_processing():
        # Create high-performance engine
        engine = create_high_performance_engine(
            compute_strategy=ComputeStrategy.HYBRID,
            max_workers=8
        )
        
        # Submit various types of tasks
        tasks = [
            ("vision_encoding", {"image_size": 336, "channels": 3}),
            ("text_encoding", {"text": "What objects are in this image?"}),
            ("multimodal_fusion", {"vision_features": "...", "text_features": "..."}),
            ("answer_generation", {"fused_features": "...", "max_tokens": 50})
        ]
        
        # Process tasks concurrently
        start_time = time.time()
        results = []
        
        for i in range(20):  # Process 20 tasks
            task_type, payload = tasks[i % len(tasks)]
            try:
                result = await engine.submit_task_async(task_type, payload, priority=i%10)
                results.append(result)
                print(f"Task {i+1} completed: {result['status']} ({result['duration_ms']:.1f}ms)")
            except Exception as e:
                print(f"Task {i+1} failed: {e}")
        
        total_time = time.time() - start_time
        
        # Get performance metrics
        metrics = engine.get_comprehensive_metrics()
        
        print(f"\n=== Performance Summary ===")
        print(f"Total time: {total_time:.2f}s")
        print(f"Completed tasks: {len(results)}")
        print(f"Throughput: {metrics['performance']['throughput_rps']:.1f} RPS")
        print(f"Average latency: {metrics['performance']['latency_p50_ms']:.1f}ms")
        print(f"Success rate: {metrics['success_rate']:.1f}%")
        
        # Test auto-scaling
        engine.auto_scale_if_needed()
        
        # Shutdown
        engine.shutdown()
    
    # Run demo
    asyncio.run(demo_distributed_processing())