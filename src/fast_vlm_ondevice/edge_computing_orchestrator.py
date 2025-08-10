"""
Edge Computing Orchestrator for FastVLM.

Coordinates distributed edge computing nodes for scalable mobile AI deployment,
with intelligent load balancing, edge-cloud hybrid processing, and
autonomous resource management across the edge continuum.
"""

import asyncio
import logging
import time
import json
import uuid
from typing import Dict, Any, Optional, List, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
import statistics
from contextlib import asynccontextmanager
import hashlib
import random
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class EdgeNodeType(Enum):
    """Types of edge computing nodes."""
    MOBILE_DEVICE = "mobile_device"      # Smartphones, tablets
    EDGE_SERVER = "edge_server"          # Local edge servers
    MICRO_DATACENTER = "micro_datacenter" # Small local datacenters
    IOT_GATEWAY = "iot_gateway"          # IoT edge gateways
    VEHICLE_EDGE = "vehicle_edge"        # Connected vehicles
    CLOUD_REGION = "cloud_region"        # Cloud fallback


class NodeStatus(Enum):
    """Edge node status."""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_RESPONSE_TIME = "weighted_response_time"
    RESOURCE_AWARE = "resource_aware"
    PREDICTIVE = "predictive"
    QUANTUM_INSPIRED = "quantum_inspired"


@dataclass
class EdgeNodeCapabilities:
    """Capabilities of an edge node."""
    cpu_cores: int
    memory_gb: float
    storage_gb: float
    gpu_available: bool = False
    neural_engine: bool = False
    bandwidth_mbps: float = 100.0
    supported_models: List[str] = field(default_factory=list)
    max_concurrent_requests: int = 10
    specializations: List[str] = field(default_factory=list)
    
    def compute_score(self) -> float:
        """Calculate overall capability score."""
        base_score = self.cpu_cores * 2 + self.memory_gb * 1.5
        if self.gpu_available:
            base_score *= 1.5
        if self.neural_engine:
            base_score *= 2.0
        return base_score + self.bandwidth_mbps / 10


@dataclass
class EdgeNodeMetrics:
    """Real-time metrics for an edge node."""
    node_id: str
    timestamp: float = field(default_factory=time.time)
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0
    network_latency_ms: float = 0.0
    active_requests: int = 0
    request_queue_size: int = 0
    average_response_time_ms: float = 0.0
    error_rate: float = 0.0
    uptime_hours: float = 0.0
    temperature_celsius: Optional[float] = None
    power_consumption_watts: Optional[float] = None
    
    def compute_health_score(self) -> float:
        """Compute overall health score (0-1)."""
        cpu_score = max(0, 1 - self.cpu_usage_percent / 100)
        memory_score = max(0, 1 - self.memory_usage_percent / 100)
        latency_score = max(0, 1 - self.network_latency_ms / 1000)
        error_score = max(0, 1 - self.error_rate)
        
        return (cpu_score + memory_score + latency_score + error_score) / 4


@dataclass
class EdgeTask:
    """Task to be executed on edge nodes."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = "inference"
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    
    # Task requirements
    estimated_cpu_time_ms: float = 100.0
    memory_requirement_mb: float = 50.0
    requires_gpu: bool = False
    requires_neural_engine: bool = False
    model_name: str = "fastvlm"
    
    # Input data
    input_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Execution tracking
    assigned_node: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    execution_time_ms: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if task has expired."""
        if self.deadline is None:
            return False
        return time.time() > self.deadline
    
    def age_seconds(self) -> float:
        """Get task age in seconds."""
        return time.time() - self.created_at
    
    def is_completed(self) -> bool:
        """Check if task is completed."""
        return self.completed_at is not None


@dataclass
class EdgeNode:
    """Represents an edge computing node."""
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: EdgeNodeType = EdgeNodeType.EDGE_SERVER
    hostname: str = "localhost"
    port: int = 8080
    capabilities: EdgeNodeCapabilities = field(default_factory=EdgeNodeCapabilities)
    status: NodeStatus = NodeStatus.ONLINE
    
    # Runtime state
    current_metrics: EdgeNodeMetrics = field(init=False)
    task_queue: deque = field(default_factory=deque)
    active_tasks: Dict[str, EdgeTask] = field(default_factory=dict)
    completed_tasks: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Performance history
    metrics_history: deque = field(default_factory=lambda: deque(maxlen=100))
    performance_baseline: Dict[str, float] = field(default_factory=dict)
    
    # Connection info
    last_heartbeat: float = field(default_factory=time.time)
    connection_established: bool = False
    
    def __post_init__(self):
        self.current_metrics = EdgeNodeMetrics(node_id=self.node_id)
    
    def is_healthy(self) -> bool:
        """Check if node is healthy and available."""
        if self.status not in [NodeStatus.ONLINE, NodeStatus.BUSY]:
            return False
        
        heartbeat_age = time.time() - self.last_heartbeat
        if heartbeat_age > 60:  # 1 minute heartbeat timeout
            return False
        
        return self.current_metrics.compute_health_score() > 0.5
    
    def can_handle_task(self, task: EdgeTask) -> bool:
        """Check if node can handle the given task."""
        if not self.is_healthy():
            return False
        
        if task.requires_gpu and not self.capabilities.gpu_available:
            return False
        
        if task.requires_neural_engine and not self.capabilities.neural_engine:
            return False
        
        if task.model_name not in self.capabilities.supported_models:
            return False
        
        # Check resource availability
        if len(self.active_tasks) >= self.capabilities.max_concurrent_requests:
            return False
        
        return True
    
    def get_load_factor(self) -> float:
        """Calculate current load factor (0-1)."""
        cpu_load = self.current_metrics.cpu_usage_percent / 100
        memory_load = self.current_metrics.memory_usage_percent / 100
        request_load = len(self.active_tasks) / self.capabilities.max_concurrent_requests
        
        return (cpu_load + memory_load + request_load) / 3
    
    def estimate_task_completion_time(self, task: EdgeTask) -> float:
        """Estimate task completion time in seconds."""
        base_time = task.estimated_cpu_time_ms / 1000
        
        # Adjust based on current load
        load_factor = self.get_load_factor()
        adjusted_time = base_time * (1 + load_factor)
        
        # Add queue wait time
        queue_wait = len(self.task_queue) * 0.1  # Estimate 100ms per queued task
        
        return adjusted_time + queue_wait


@dataclass
class EdgeCluster:
    """A cluster of edge nodes working together."""
    cluster_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cluster_name: str = "default_cluster"
    nodes: Dict[str, EdgeNode] = field(default_factory=dict)
    
    # Cluster configuration
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.RESOURCE_AWARE
    auto_scaling_enabled: bool = True
    max_nodes: int = 100
    min_nodes: int = 1
    
    # Cluster metrics
    total_requests_processed: int = 0
    cluster_uptime_start: float = field(default_factory=time.time)
    
    def get_healthy_nodes(self) -> List[EdgeNode]:
        """Get list of healthy nodes in the cluster."""
        return [node for node in self.nodes.values() if node.is_healthy()]
    
    def get_cluster_capacity(self) -> Dict[str, float]:
        """Calculate total cluster capacity."""
        healthy_nodes = self.get_healthy_nodes()
        
        if not healthy_nodes:
            return {"total_cpu_cores": 0, "total_memory_gb": 0, "total_nodes": 0}
        
        total_cpu = sum(node.capabilities.cpu_cores for node in healthy_nodes)
        total_memory = sum(node.capabilities.memory_gb for node in healthy_nodes)
        total_requests = sum(node.capabilities.max_concurrent_requests for node in healthy_nodes)
        
        return {
            "total_cpu_cores": total_cpu,
            "total_memory_gb": total_memory,
            "total_max_requests": total_requests,
            "total_nodes": len(healthy_nodes),
            "gpu_nodes": sum(1 for node in healthy_nodes if node.capabilities.gpu_available),
            "neural_engine_nodes": sum(1 for node in healthy_nodes if node.capabilities.neural_engine)
        }
    
    def get_cluster_load(self) -> Dict[str, float]:
        """Calculate current cluster load metrics."""
        healthy_nodes = self.get_healthy_nodes()
        
        if not healthy_nodes:
            return {"cpu_usage": 0, "memory_usage": 0, "active_requests": 0}
        
        avg_cpu = statistics.mean(node.current_metrics.cpu_usage_percent for node in healthy_nodes)
        avg_memory = statistics.mean(node.current_metrics.memory_usage_percent for node in healthy_nodes)
        total_active = sum(len(node.active_tasks) for node in healthy_nodes)
        
        return {
            "avg_cpu_usage_percent": avg_cpu,
            "avg_memory_usage_percent": avg_memory,
            "total_active_requests": total_active,
            "avg_response_time_ms": statistics.mean(node.current_metrics.average_response_time_ms for node in healthy_nodes),
            "cluster_error_rate": statistics.mean(node.current_metrics.error_rate for node in healthy_nodes)
        }


class IntelligentLoadBalancer:
    """Intelligent load balancer for edge nodes."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.RESOURCE_AWARE):
        self.strategy = strategy
        self.node_weights = {}
        self.performance_history = defaultdict(list)
        self.prediction_model = None
    
    async def select_node(self, task: EdgeTask, available_nodes: List[EdgeNode]) -> Optional[EdgeNode]:
        """Select the best node for the given task."""
        if not available_nodes:
            return None
        
        # Filter nodes that can handle the task
        capable_nodes = [node for node in available_nodes if node.can_handle_task(task)]
        
        if not capable_nodes:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return await self._round_robin_selection(capable_nodes)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return await self._least_connections_selection(capable_nodes)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_RESPONSE_TIME:
            return await self._weighted_response_time_selection(capable_nodes)
        elif self.strategy == LoadBalancingStrategy.RESOURCE_AWARE:
            return await self._resource_aware_selection(capable_nodes, task)
        elif self.strategy == LoadBalancingStrategy.PREDICTIVE:
            return await self._predictive_selection(capable_nodes, task)
        elif self.strategy == LoadBalancingStrategy.QUANTUM_INSPIRED:
            return await self._quantum_inspired_selection(capable_nodes, task)
        else:
            return random.choice(capable_nodes)
    
    async def _round_robin_selection(self, nodes: List[EdgeNode]) -> EdgeNode:
        """Simple round-robin selection."""
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0
        
        selected_node = nodes[self._round_robin_index % len(nodes)]
        self._round_robin_index += 1
        return selected_node
    
    async def _least_connections_selection(self, nodes: List[EdgeNode]) -> EdgeNode:
        """Select node with fewest active connections."""
        return min(nodes, key=lambda node: len(node.active_tasks))
    
    async def _weighted_response_time_selection(self, nodes: List[EdgeNode]) -> EdgeNode:
        """Select node based on weighted response time."""
        def score_function(node: EdgeNode) -> float:
            response_time = node.current_metrics.average_response_time_ms
            load_factor = node.get_load_factor()
            return response_time * (1 + load_factor)
        
        return min(nodes, key=score_function)
    
    async def _resource_aware_selection(self, nodes: List[EdgeNode], task: EdgeTask) -> EdgeNode:
        """Select node based on resource requirements and availability."""
        def score_function(node: EdgeNode) -> float:
            # Calculate resource fit score
            cpu_availability = 1 - node.current_metrics.cpu_usage_percent / 100
            memory_availability = 1 - node.current_metrics.memory_usage_percent / 100
            
            # Task-specific scoring
            score = cpu_availability * 0.4 + memory_availability * 0.3
            
            # Bonus for special capabilities
            if task.requires_gpu and node.capabilities.gpu_available:
                score += 0.2
            if task.requires_neural_engine and node.capabilities.neural_engine:
                score += 0.3
            
            # Penalty for high load
            load_penalty = node.get_load_factor() * 0.5
            score -= load_penalty
            
            return -score  # Negative because we want max score
        
        return min(nodes, key=score_function)
    
    async def _predictive_selection(self, nodes: List[EdgeNode], task: EdgeTask) -> EdgeNode:
        """Select node based on predicted performance."""
        predictions = {}
        
        for node in nodes:
            # Predict completion time
            predicted_time = node.estimate_task_completion_time(task)
            
            # Consider historical performance
            historical_performance = self._get_historical_performance(node, task.task_type)
            
            # Combine prediction with history
            combined_score = predicted_time * 0.7 + historical_performance * 0.3
            predictions[node.node_id] = combined_score
        
        # Select node with best predicted performance
        best_node_id = min(predictions, key=predictions.get)
        return next(node for node in nodes if node.node_id == best_node_id)
    
    async def _quantum_inspired_selection(self, nodes: List[EdgeNode], task: EdgeTask) -> EdgeNode:
        """Quantum-inspired node selection using superposition and entanglement concepts."""
        # Create "quantum" state for each node
        node_states = {}
        
        for node in nodes:
            # Calculate node "energy" (lower is better)
            energy = (
                node.get_load_factor() * 0.4 +
                (node.current_metrics.average_response_time_ms / 1000) * 0.3 +
                node.current_metrics.error_rate * 0.3
            )
            
            # Create superposition amplitude (higher for better nodes)
            amplitude = math.exp(-energy)
            
            node_states[node.node_id] = {
                'node': node,
                'energy': energy,
                'amplitude': amplitude,
                'probability': amplitude ** 2
            }
        
        # Normalize probabilities
        total_prob = sum(state['probability'] for state in node_states.values())
        if total_prob > 0:
            for state in node_states.values():
                state['probability'] /= total_prob
        
        # "Measure" the quantum system (probabilistic selection)
        rand_val = random.random()
        cumulative_prob = 0.0
        
        for state in node_states.values():
            cumulative_prob += state['probability']
            if rand_val <= cumulative_prob:
                return state['node']
        
        # Fallback
        return nodes[0]
    
    def _get_historical_performance(self, node: EdgeNode, task_type: str) -> float:
        """Get historical performance score for node and task type."""
        history_key = f"{node.node_id}_{task_type}"
        
        if history_key not in self.performance_history:
            return 0.5  # Neutral score for new nodes
        
        recent_performances = self.performance_history[history_key][-10:]  # Last 10 tasks
        return statistics.mean(recent_performances) if recent_performances else 0.5
    
    async def record_task_performance(self, node: EdgeNode, task: EdgeTask, performance_score: float):
        """Record task performance for future predictions."""
        history_key = f"{node.node_id}_{task.task_type}"
        self.performance_history[history_key].append(performance_score)
        
        # Keep only recent history
        if len(self.performance_history[history_key]) > 100:
            self.performance_history[history_key] = self.performance_history[history_key][-50:]


class EdgeComputingOrchestrator:
    """Main orchestrator for edge computing operations."""
    
    def __init__(self, cluster: EdgeCluster = None):
        self.cluster = cluster or EdgeCluster()
        self.load_balancer = IntelligentLoadBalancer()
        
        # Task management
        self.global_task_queue = asyncio.PriorityQueue()
        self.completed_tasks = deque(maxlen=10000)
        self.failed_tasks = deque(maxlen=1000)
        
        # Orchestrator state
        self.is_running = False
        self.start_time = None
        self.metrics = {
            "tasks_processed": 0,
            "tasks_failed": 0,
            "total_processing_time_ms": 0.0,
            "average_response_time_ms": 0.0,
            "nodes_online": 0,
            "cluster_utilization": 0.0
        }
        
        # Background tasks
        self._background_tasks = []
        self._shutdown_event = asyncio.Event()
        
        # Auto-scaling
        self.auto_scaler = EdgeAutoScaler(self.cluster)
        
        # Health monitoring
        self.health_monitor = EdgeHealthMonitor(self.cluster)
    
    async def start(self):
        """Start the edge computing orchestrator."""
        if self.is_running:
            return
        
        logger.info("Starting Edge Computing Orchestrator")
        self.is_running = True
        self.start_time = time.time()
        
        # Start background tasks
        self._background_tasks = [
            asyncio.create_task(self._task_dispatcher_loop()),
            asyncio.create_task(self._health_monitoring_loop()),
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._auto_scaling_loop()),
            asyncio.create_task(self._task_cleanup_loop())
        ]
        
        logger.info(f"Edge orchestrator started with {len(self.cluster.nodes)} nodes")
        
        # Wait for shutdown
        await self._shutdown_event.wait()
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
    
    async def stop(self):
        """Stop the edge computing orchestrator."""
        if not self.is_running:
            return
        
        logger.info("Stopping Edge Computing Orchestrator")
        self.is_running = False
        self._shutdown_event.set()
    
    async def submit_task(self, task: EdgeTask) -> str:
        """Submit a task for execution on the edge cluster."""
        logger.debug(f"Submitting task {task.task_id[:8]} (type: {task.task_type})")
        
        # Add to global queue with priority
        priority_value = task.priority.value
        await self.global_task_queue.put((priority_value, task.created_at, task))
        
        return task.task_id
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a specific task."""
        # Check active tasks in all nodes
        for node in self.cluster.nodes.values():
            if task_id in node.active_tasks:
                task = node.active_tasks[task_id]
                return {
                    "task_id": task_id,
                    "status": "running",
                    "assigned_node": node.node_id,
                    "progress": self._estimate_task_progress(task),
                    "estimated_completion": time.time() + node.estimate_task_completion_time(task)
                }
        
        # Check completed tasks
        for task_record in self.completed_tasks:
            if task_record["task_id"] == task_id:
                return {
                    "task_id": task_id,
                    "status": "completed",
                    "result": task_record["result"],
                    "execution_time_ms": task_record["execution_time_ms"],
                    "completed_at": task_record["completed_at"]
                }
        
        # Check failed tasks
        for task_record in self.failed_tasks:
            if task_record["task_id"] == task_id:
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": task_record["error"],
                    "failed_at": task_record["failed_at"]
                }
        
        return {"task_id": task_id, "status": "not_found"}
    
    async def add_edge_node(self, node: EdgeNode) -> bool:
        """Add a new edge node to the cluster."""
        if node.node_id in self.cluster.nodes:
            logger.warning(f"Node {node.node_id} already exists in cluster")
            return False
        
        self.cluster.nodes[node.node_id] = node
        node.connection_established = True
        node.last_heartbeat = time.time()
        
        logger.info(f"Added edge node {node.node_id} ({node.node_type.value}) to cluster")
        return True
    
    async def remove_edge_node(self, node_id: str) -> bool:
        """Remove an edge node from the cluster."""
        if node_id not in self.cluster.nodes:
            return False
        
        node = self.cluster.nodes[node_id]
        
        # Migrate active tasks to other nodes
        await self._migrate_tasks_from_node(node)
        
        del self.cluster.nodes[node_id]
        logger.info(f"Removed edge node {node_id} from cluster")
        return True
    
    async def _task_dispatcher_loop(self):
        """Main task dispatcher loop."""
        logger.info("Starting task dispatcher loop")
        
        while self.is_running:
            try:
                # Get next task from queue
                try:
                    priority, timestamp, task = await asyncio.wait_for(
                        self.global_task_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Check if task has expired
                if task.is_expired():
                    logger.warning(f"Task {task.task_id[:8]} expired, skipping")
                    continue
                
                # Select best node for the task
                healthy_nodes = self.cluster.get_healthy_nodes()
                selected_node = await self.load_balancer.select_node(task, healthy_nodes)
                
                if selected_node is None:
                    logger.warning(f"No suitable node found for task {task.task_id[:8]}, requeueing")
                    await asyncio.sleep(1)
                    await self.global_task_queue.put((priority, timestamp, task))
                    continue
                
                # Assign task to node
                await self._assign_task_to_node(task, selected_node)
                
            except Exception as e:
                logger.error(f"Task dispatcher error: {e}")
                await asyncio.sleep(5)
    
    async def _assign_task_to_node(self, task: EdgeTask, node: EdgeNode):
        """Assign a task to a specific node."""
        task.assigned_node = node.node_id
        task.started_at = time.time()
        
        # Add to node's active tasks
        node.active_tasks[task.task_id] = task
        
        # Simulate task execution
        asyncio.create_task(self._execute_task_on_node(task, node))
        
        logger.debug(f"Assigned task {task.task_id[:8]} to node {node.node_id[:8]}")
    
    async def _execute_task_on_node(self, task: EdgeTask, node: EdgeNode):
        """Execute a task on a specific node."""
        try:
            # Simulate task execution
            execution_time = node.estimate_task_completion_time(task)
            await asyncio.sleep(execution_time)
            
            # Simulate successful completion
            task.completed_at = time.time()
            task.execution_time_ms = (task.completed_at - task.started_at) * 1000
            task.result = {
                "answer": f"Processed by node {node.node_id[:8]}",
                "confidence": 0.85,
                "processing_time_ms": task.execution_time_ms
            }
            
            # Record completion
            await self._record_task_completion(task, node)
            
        except Exception as e:
            logger.error(f"Task execution failed on node {node.node_id}: {e}")
            task.error = str(e)
            await self._record_task_failure(task, node)
        
        finally:
            # Remove from active tasks
            if task.task_id in node.active_tasks:
                del node.active_tasks[task.task_id]
    
    async def _record_task_completion(self, task: EdgeTask, node: EdgeNode):
        """Record successful task completion."""
        # Update metrics
        self.metrics["tasks_processed"] += 1
        self.metrics["total_processing_time_ms"] += task.execution_time_ms
        self.metrics["average_response_time_ms"] = (
            self.metrics["total_processing_time_ms"] / self.metrics["tasks_processed"]
        )
        
        # Store completion record
        completion_record = {
            "task_id": task.task_id,
            "node_id": node.node_id,
            "completed_at": task.completed_at,
            "execution_time_ms": task.execution_time_ms,
            "result": task.result
        }
        self.completed_tasks.append(completion_record)
        
        # Record performance for load balancer
        performance_score = 1.0 - (task.execution_time_ms / 10000)  # Simple scoring
        await self.load_balancer.record_task_performance(node, task, performance_score)
        
        logger.debug(f"Task {task.task_id[:8]} completed successfully on node {node.node_id[:8]}")
    
    async def _record_task_failure(self, task: EdgeTask, node: EdgeNode):
        """Record task failure."""
        self.metrics["tasks_failed"] += 1
        
        failure_record = {
            "task_id": task.task_id,
            "node_id": node.node_id,
            "failed_at": time.time(),
            "error": task.error
        }
        self.failed_tasks.append(failure_record)
        
        # Record poor performance
        await self.load_balancer.record_task_performance(node, task, 0.0)
        
        logger.error(f"Task {task.task_id[:8]} failed on node {node.node_id[:8]}: {task.error}")
    
    async def _health_monitoring_loop(self):
        """Monitor health of all edge nodes."""
        logger.info("Starting health monitoring loop")
        
        while self.is_running:
            try:
                await self.health_monitor.check_all_nodes()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_collection_loop(self):
        """Collect and update cluster metrics."""
        logger.info("Starting metrics collection loop")
        
        while self.is_running:
            try:
                # Update cluster metrics
                healthy_nodes = self.cluster.get_healthy_nodes()
                self.metrics["nodes_online"] = len(healthy_nodes)
                
                if healthy_nodes:
                    cluster_load = self.cluster.get_cluster_load()
                    self.metrics["cluster_utilization"] = (
                        cluster_load["avg_cpu_usage_percent"] + 
                        cluster_load["avg_memory_usage_percent"]
                    ) / 200  # Normalize to 0-1
                
                await asyncio.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(30)
    
    async def _auto_scaling_loop(self):
        """Auto-scaling loop for dynamic node management."""
        if not self.cluster.auto_scaling_enabled:
            return
        
        logger.info("Starting auto-scaling loop")
        
        while self.is_running:
            try:
                await self.auto_scaler.evaluate_scaling_needs()
                await asyncio.sleep(60)  # Evaluate every minute
            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")
                await asyncio.sleep(300)  # Wait longer on error
    
    async def _task_cleanup_loop(self):
        """Clean up old completed/failed tasks."""
        logger.info("Starting task cleanup loop")
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Clean up old completed tasks (keep last 24 hours)
                old_threshold = current_time - 86400  # 24 hours
                self.completed_tasks = deque(
                    [task for task in self.completed_tasks 
                     if task["completed_at"] > old_threshold],
                    maxlen=10000
                )
                
                # Clean up old failed tasks
                self.failed_tasks = deque(
                    [task for task in self.failed_tasks 
                     if task["failed_at"] > old_threshold],
                    maxlen=1000
                )
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                logger.error(f"Task cleanup error: {e}")
                await asyncio.sleep(1800)  # Wait longer on error
    
    def _estimate_task_progress(self, task: EdgeTask) -> float:
        """Estimate task progress based on elapsed time."""
        if task.started_at is None:
            return 0.0
        
        elapsed_time = time.time() - task.started_at
        estimated_total = task.estimated_cpu_time_ms / 1000
        
        return min(0.95, elapsed_time / estimated_total)  # Cap at 95%
    
    async def _migrate_tasks_from_node(self, node: EdgeNode):
        """Migrate active tasks from a node to other nodes."""
        tasks_to_migrate = list(node.active_tasks.values())
        
        for task in tasks_to_migrate:
            # Remove from current node
            if task.task_id in node.active_tasks:
                del node.active_tasks[task.task_id]
            
            # Reset task state
            task.assigned_node = None
            task.started_at = None
            
            # Re-queue task
            priority = task.priority.value
            await self.global_task_queue.put((priority, task.created_at, task))
        
        logger.info(f"Migrated {len(tasks_to_migrate)} tasks from node {node.node_id}")
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status."""
        cluster_capacity = self.cluster.get_cluster_capacity()
        cluster_load = self.cluster.get_cluster_load()
        
        return {
            "orchestrator": {
                "is_running": self.is_running,
                "uptime_hours": (time.time() - (self.start_time or time.time())) / 3600,
                "queue_size": self.global_task_queue.qsize()
            },
            "cluster": {
                "cluster_id": self.cluster.cluster_id,
                "total_nodes": len(self.cluster.nodes),
                "healthy_nodes": len(self.cluster.get_healthy_nodes()),
                "capacity": cluster_capacity,
                "current_load": cluster_load
            },
            "metrics": self.metrics.copy(),
            "load_balancer": {
                "strategy": self.load_balancer.strategy.value,
                "nodes_tracked": len(self.load_balancer.node_weights)
            }
        }


class EdgeAutoScaler:
    """Auto-scaler for edge computing clusters."""
    
    def __init__(self, cluster: EdgeCluster):
        self.cluster = cluster
        self.scaling_history = deque(maxlen=100)
        self.last_scaling_action = 0.0
        self.scaling_cooldown = 300  # 5 minutes
    
    async def evaluate_scaling_needs(self):
        """Evaluate if cluster needs scaling up or down."""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_scaling_action < self.scaling_cooldown:
            return
        
        cluster_load = self.cluster.get_cluster_load()
        healthy_nodes = self.cluster.get_healthy_nodes()
        
        # Scale up conditions
        should_scale_up = (
            cluster_load["avg_cpu_usage_percent"] > 80 or
            cluster_load["avg_memory_usage_percent"] > 85 or
            cluster_load["total_active_requests"] > sum(node.capabilities.max_concurrent_requests for node in healthy_nodes) * 0.9
        )
        
        # Scale down conditions
        should_scale_down = (
            len(healthy_nodes) > self.cluster.min_nodes and
            cluster_load["avg_cpu_usage_percent"] < 30 and
            cluster_load["avg_memory_usage_percent"] < 40 and
            cluster_load["total_active_requests"] < sum(node.capabilities.max_concurrent_requests for node in healthy_nodes) * 0.3
        )
        
        if should_scale_up and len(healthy_nodes) < self.cluster.max_nodes:
            await self._scale_up()
        elif should_scale_down:
            await self._scale_down()
    
    async def _scale_up(self):
        """Scale up the cluster by adding nodes."""
        logger.info("Auto-scaling: Adding new edge node")
        
        # Create new node (in real implementation, this would provision actual resources)
        new_node = EdgeNode(
            node_type=EdgeNodeType.EDGE_SERVER,
            capabilities=EdgeNodeCapabilities(
                cpu_cores=4,
                memory_gb=8,
                storage_gb=100,
                gpu_available=False,
                supported_models=["fastvlm"],
                max_concurrent_requests=10
            )
        )
        
        self.cluster.nodes[new_node.node_id] = new_node
        self.last_scaling_action = time.time()
        
        self.scaling_history.append({
            "timestamp": self.last_scaling_action,
            "action": "scale_up",
            "node_count_before": len(self.cluster.nodes) - 1,
            "node_count_after": len(self.cluster.nodes)
        })
        
        logger.info(f"Scaled up cluster to {len(self.cluster.nodes)} nodes")
    
    async def _scale_down(self):
        """Scale down the cluster by removing underutilized nodes."""
        healthy_nodes = self.cluster.get_healthy_nodes()
        
        if len(healthy_nodes) <= self.cluster.min_nodes:
            return
        
        # Find node with lowest utilization
        least_utilized = min(healthy_nodes, key=lambda node: node.get_load_factor())
        
        # Only scale down if node is truly underutilized and has no active tasks
        if least_utilized.get_load_factor() < 0.2 and len(least_utilized.active_tasks) == 0:
            logger.info(f"Auto-scaling: Removing underutilized node {least_utilized.node_id[:8]}")
            
            del self.cluster.nodes[least_utilized.node_id]
            self.last_scaling_action = time.time()
            
            self.scaling_history.append({
                "timestamp": self.last_scaling_action,
                "action": "scale_down",
                "node_count_before": len(self.cluster.nodes) + 1,
                "node_count_after": len(self.cluster.nodes),
                "removed_node": least_utilized.node_id
            })
            
            logger.info(f"Scaled down cluster to {len(self.cluster.nodes)} nodes")


class EdgeHealthMonitor:
    """Health monitoring for edge nodes."""
    
    def __init__(self, cluster: EdgeCluster):
        self.cluster = cluster
        self.health_history = defaultdict(list)
    
    async def check_all_nodes(self):
        """Check health of all nodes in the cluster."""
        current_time = time.time()
        
        for node in self.cluster.nodes.values():
            health_score = await self._check_node_health(node)
            
            # Update node status based on health
            if health_score > 0.8:
                node.status = NodeStatus.ONLINE
            elif health_score > 0.5:
                node.status = NodeStatus.DEGRADED
            elif health_score > 0.2:
                node.status = NodeStatus.BUSY
            else:
                node.status = NodeStatus.OFFLINE
            
            # Store health history
            self.health_history[node.node_id].append({
                "timestamp": current_time,
                "health_score": health_score,
                "status": node.status.value
            })
            
            # Keep only recent history
            if len(self.health_history[node.node_id]) > 100:
                self.health_history[node.node_id] = self.health_history[node.node_id][-50:]
    
    async def _check_node_health(self, node: EdgeNode) -> float:
        """Check health of a specific node."""
        # Simulate health check
        base_health = node.current_metrics.compute_health_score()
        
        # Simulate random node metrics
        node.current_metrics.cpu_usage_percent = random.uniform(20, 90)
        node.current_metrics.memory_usage_percent = random.uniform(30, 85)
        node.current_metrics.network_latency_ms = random.uniform(10, 200)
        node.current_metrics.active_requests = len(node.active_tasks)
        node.current_metrics.error_rate = random.uniform(0, 0.1)
        
        # Update heartbeat
        node.last_heartbeat = time.time()
        
        return base_health


# Factory functions
def create_edge_orchestrator(cluster_config: Dict[str, Any] = None) -> EdgeComputingOrchestrator:
    """Create an edge computing orchestrator with specified configuration."""
    cluster = EdgeCluster()
    
    if cluster_config:
        cluster.cluster_name = cluster_config.get("cluster_name", "default_cluster")
        cluster.load_balancing_strategy = LoadBalancingStrategy(
            cluster_config.get("load_balancing_strategy", "resource_aware")
        )
        cluster.auto_scaling_enabled = cluster_config.get("auto_scaling_enabled", True)
    
    return EdgeComputingOrchestrator(cluster)


def create_mobile_edge_node() -> EdgeNode:
    """Create an edge node optimized for mobile devices."""
    return EdgeNode(
        node_type=EdgeNodeType.MOBILE_DEVICE,
        capabilities=EdgeNodeCapabilities(
            cpu_cores=8,
            memory_gb=6,
            storage_gb=128,
            gpu_available=True,
            neural_engine=True,
            bandwidth_mbps=50,
            supported_models=["fastvlm", "fastvlm-tiny"],
            max_concurrent_requests=3,
            specializations=["mobile_inference", "low_latency"]
        )
    )


def create_edge_server_node() -> EdgeNode:
    """Create an edge server node for higher capacity processing."""
    return EdgeNode(
        node_type=EdgeNodeType.EDGE_SERVER,
        capabilities=EdgeNodeCapabilities(
            cpu_cores=16,
            memory_gb=32,
            storage_gb=1000,
            gpu_available=True,
            neural_engine=False,
            bandwidth_mbps=1000,
            supported_models=["fastvlm", "fastvlm-base", "fastvlm-large"],
            max_concurrent_requests=50,
            specializations=["high_throughput", "batch_processing"]
        )
    )
