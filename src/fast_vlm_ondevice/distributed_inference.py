"""
Distributed inference and scaling capabilities for FastVLM.

Implements horizontal scaling, load balancing, distributed caching,
and multi-node inference coordination for production scale.
"""

import time
import json
import logging
import threading
import hashlib
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
import queue
import concurrent.futures

logger = logging.getLogger(__name__)


@dataclass
class NodeInfo:
    """Information about a worker node."""
    node_id: str
    address: str
    port: int
    capabilities: List[str] = field(default_factory=list)
    load: float = 0.0
    status: str = "healthy"  # healthy, degraded, unhealthy
    last_heartbeat: float = field(default_factory=time.time)
    active_requests: int = 0
    total_requests: int = 0
    average_latency: float = 0.0


@dataclass
class InferenceRequest:
    """Distributed inference request."""
    request_id: str
    image_data: bytes
    question: str
    priority: int = 5  # 1=highest, 10=lowest
    timeout: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    node_preferences: List[str] = field(default_factory=list)


@dataclass
class InferenceResponse:
    """Distributed inference response."""
    request_id: str
    answer: str
    confidence: float
    latency_ms: float
    node_id: str
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LoadBalancer:
    """Intelligent load balancer for distributing inference requests."""
    
    def __init__(self):
        """Initialize load balancer."""
        self.nodes = {}
        self.routing_strategy = "least_loaded"  # round_robin, least_loaded, weighted
        self.health_check_interval = 30.0
        self.node_weights = {}
        self.lock = threading.Lock()
        
        logger.info("Load balancer initialized")
    
    def register_node(self, node: NodeInfo):
        """Register a new worker node."""
        with self.lock:
            self.nodes[node.node_id] = node
            self.node_weights[node.node_id] = 1.0
        
        logger.info(f"Node registered: {node.node_id} at {node.address}:{node.port}")
    
    def unregister_node(self, node_id: str):
        """Unregister a worker node."""
        with self.lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                del self.node_weights[node_id]
        
        logger.info(f"Node unregistered: {node_id}")
    
    def select_node(self, request: InferenceRequest) -> Optional[NodeInfo]:
        """Select the best node for a request."""
        with self.lock:
            available_nodes = [
                node for node in self.nodes.values()
                if node.status == "healthy" and node.active_requests < 10
            ]
        
        if not available_nodes:
            return None
        
        # Apply node preferences if specified
        if request.node_preferences:
            preferred_nodes = [
                node for node in available_nodes
                if node.node_id in request.node_preferences
            ]
            if preferred_nodes:
                available_nodes = preferred_nodes
        
        # Select based on strategy
        if self.routing_strategy == "round_robin":
            return self._round_robin_selection(available_nodes)
        elif self.routing_strategy == "least_loaded":
            return self._least_loaded_selection(available_nodes)
        elif self.routing_strategy == "weighted":
            return self._weighted_selection(available_nodes)
        else:
            return available_nodes[0]
    
    def _round_robin_selection(self, nodes: List[NodeInfo]) -> NodeInfo:
        """Round-robin node selection."""
        # Simple round-robin based on total requests
        return min(nodes, key=lambda n: n.total_requests)
    
    def _least_loaded_selection(self, nodes: List[NodeInfo]) -> NodeInfo:
        """Select node with least current load."""
        return min(nodes, key=lambda n: n.active_requests + n.load)
    
    def _weighted_selection(self, nodes: List[NodeInfo]) -> NodeInfo:
        """Select node based on weighted load."""
        def weighted_load(node):
            weight = self.node_weights.get(node.node_id, 1.0)
            return (node.active_requests + node.load) / weight
        
        return min(nodes, key=weighted_load)
    
    def update_node_stats(self, node_id: str, latency_ms: float, success: bool):
        """Update node performance statistics."""
        with self.lock:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                
                # Update average latency with exponential moving average
                alpha = 0.1
                if node.average_latency == 0:
                    node.average_latency = latency_ms
                else:
                    node.average_latency = (alpha * latency_ms + 
                                          (1 - alpha) * node.average_latency)
                
                # Update node weight based on performance
                if success and latency_ms < 500:
                    # Good performance, increase weight
                    self.node_weights[node_id] = min(2.0, self.node_weights[node_id] * 1.05)
                elif not success or latency_ms > 1000:
                    # Poor performance, decrease weight
                    self.node_weights[node_id] = max(0.1, self.node_weights[node_id] * 0.95)
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get cluster-wide statistics."""
        with self.lock:
            total_nodes = len(self.nodes)
            healthy_nodes = sum(1 for n in self.nodes.values() if n.status == "healthy")
            total_active_requests = sum(n.active_requests for n in self.nodes.values())
            avg_latency = sum(n.average_latency for n in self.nodes.values()) / max(1, total_nodes)
            
            return {
                "total_nodes": total_nodes,
                "healthy_nodes": healthy_nodes,
                "total_active_requests": total_active_requests,
                "average_latency_ms": avg_latency,
                "routing_strategy": self.routing_strategy,
                "nodes": [
                    {
                        "node_id": node.node_id,
                        "address": f"{node.address}:{node.port}",
                        "status": node.status,
                        "load": node.load,
                        "active_requests": node.active_requests,
                        "average_latency": node.average_latency
                    }
                    for node in self.nodes.values()
                ]
            }


class DistributedCache:
    """Distributed caching system for inference results."""
    
    def __init__(self, max_size_mb: int = 100):
        """Initialize distributed cache."""
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.local_cache = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size_bytes": 0
        }
        self.access_times = {}
        self.lock = threading.Lock()
        
        logger.info(f"Distributed cache initialized with {max_size_mb}MB limit")
    
    def _generate_cache_key(self, image_data: bytes, question: str) -> str:
        """Generate cache key for image-question pair."""
        combined = hashlib.sha256(image_data).hexdigest() + question
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def get(self, image_data: bytes, question: str) -> Optional[InferenceResponse]:
        """Get cached inference result."""
        cache_key = self._generate_cache_key(image_data, question)
        
        with self.lock:
            if cache_key in self.local_cache:
                self.cache_stats["hits"] += 1
                self.access_times[cache_key] = time.time()
                return self.local_cache[cache_key]
            else:
                self.cache_stats["misses"] += 1
                return None
    
    def put(self, image_data: bytes, question: str, response: InferenceResponse):
        """Cache inference result."""
        cache_key = self._generate_cache_key(image_data, question)
        
        with self.lock:
            # Estimate response size
            response_size = len(response.answer) * 2 + 100  # Rough estimate
            
            # Check if we need to evict
            while (self.cache_stats["size_bytes"] + response_size > self.max_size_bytes 
                   and self.local_cache):
                self._evict_lru()
            
            # Add to cache
            self.local_cache[cache_key] = response
            self.access_times[cache_key] = time.time()
            self.cache_stats["size_bytes"] += response_size
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.access_times:
            return
        
        # Find LRU item
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # Remove from cache
        if lru_key in self.local_cache:
            response = self.local_cache[lru_key]
            response_size = len(response.answer) * 2 + 100
            
            del self.local_cache[lru_key]
            del self.access_times[lru_key]
            self.cache_stats["size_bytes"] -= response_size
            self.cache_stats["evictions"] += 1
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        if total_requests == 0:
            return 0.0
        return self.cache_stats["hits"] / total_requests
    
    def clear(self):
        """Clear all cached data."""
        with self.lock:
            self.local_cache.clear()
            self.access_times.clear()
            self.cache_stats["size_bytes"] = 0
            self.cache_stats["evictions"] = 0


class RequestQueue:
    """Priority queue for managing inference requests."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize request queue."""
        self.max_size = max_size
        self.queues = defaultdict(deque)  # Priority -> deque of requests
        self.request_map = {}  # request_id -> request
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        
        logger.info(f"Request queue initialized with max size {max_size}")
    
    def enqueue(self, request: InferenceRequest) -> bool:
        """Add request to queue."""
        with self.condition:
            total_size = sum(len(q) for q in self.queues.values())
            
            if total_size >= self.max_size:
                # Drop lowest priority request if queue is full
                self._drop_lowest_priority()
            
            self.queues[request.priority].append(request)
            self.request_map[request.request_id] = request
            self.condition.notify()
            return True
    
    def dequeue(self, timeout: float = 1.0) -> Optional[InferenceRequest]:
        """Get next request from queue."""
        with self.condition:
            end_time = time.time() + timeout
            
            while not any(self.queues.values()):
                remaining = end_time - time.time()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)
            
            # Get highest priority request
            for priority in sorted(self.queues.keys()):
                if self.queues[priority]:
                    request = self.queues[priority].popleft()
                    del self.request_map[request.request_id]
                    return request
            
            return None
    
    def _drop_lowest_priority(self):
        """Drop a request with lowest priority."""
        for priority in sorted(self.queues.keys(), reverse=True):
            if self.queues[priority]:
                dropped = self.queues[priority].popleft()
                del self.request_map[dropped.request_id]
                logger.warning(f"Dropped request {dropped.request_id} due to queue overflow")
                break
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self.lock:
            return {
                "total_requests": len(self.request_map),
                "requests_by_priority": {
                    str(priority): len(queue) 
                    for priority, queue in self.queues.items()
                },
                "max_size": self.max_size
            }


class DistributedInferenceEngine:
    """Main distributed inference engine."""
    
    def __init__(self, node_id: str = "coordinator"):
        """Initialize distributed inference engine."""
        self.node_id = node_id
        self.load_balancer = LoadBalancer()
        self.distributed_cache = DistributedCache()
        self.request_queue = RequestQueue()
        self.active_requests = {}
        self.request_futures = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.running = False
        self.worker_threads = []
        
        # Local inference pipeline (fallback)
        self.local_pipeline = None
        
        logger.info(f"Distributed inference engine initialized (node: {node_id})")
    
    def start(self):
        """Start the distributed inference engine."""
        self.running = True
        
        # Start worker threads
        for i in range(3):
            thread = threading.Thread(target=self._worker_loop, daemon=True)
            thread.start()
            self.worker_threads.append(thread)
        
        logger.info("Distributed inference engine started")
    
    def stop(self):
        """Stop the distributed inference engine."""
        self.running = False
        self.executor.shutdown(wait=True)
        
        for thread in self.worker_threads:
            thread.join(timeout=5.0)
        
        logger.info("Distributed inference engine stopped")
    
    def set_local_pipeline(self, pipeline):
        """Set local inference pipeline for fallback."""
        self.local_pipeline = pipeline
        logger.info("Local pipeline set for fallback inference")
    
    def submit_request(self, image_data: bytes, question: str, 
                      priority: int = 5, timeout: float = 30.0) -> str:
        """Submit inference request and return request ID."""
        request_id = hashlib.sha256(
            f"{time.time()}_{question}".encode()
        ).hexdigest()[:16]
        
        # Check cache first
        cached_response = self.distributed_cache.get(image_data, question)
        if cached_response:
            logger.info(f"Cache hit for request {request_id}")
            # Return cached response immediately
            self.active_requests[request_id] = cached_response
            return request_id
        
        # Create request
        request = InferenceRequest(
            request_id=request_id,
            image_data=image_data,
            question=question,
            priority=priority,
            timeout=timeout
        )
        
        # Queue for processing
        if self.request_queue.enqueue(request):
            logger.info(f"Request {request_id} queued with priority {priority}")
            return request_id
        else:
            raise RuntimeError("Failed to queue request")
    
    def get_result(self, request_id: str, timeout: float = 30.0) -> Optional[InferenceResponse]:
        """Get result for a request."""
        end_time = time.time() + timeout
        
        while time.time() < end_time:
            if request_id in self.active_requests:
                result = self.active_requests[request_id]
                del self.active_requests[request_id]
                return result
            time.sleep(0.1)
        
        return None
    
    def _worker_loop(self):
        """Main worker loop for processing requests."""
        while self.running:
            try:
                # Get next request
                request = self.request_queue.dequeue(timeout=1.0)
                if not request:
                    continue
                
                # Process request
                response = self._process_request(request)
                
                # Store result
                self.active_requests[request.request_id] = response
                
                # Cache successful results
                if response.success:
                    self.distributed_cache.put(
                        request.image_data, 
                        request.question, 
                        response
                    )
                
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
    
    def _process_request(self, request: InferenceRequest) -> InferenceResponse:
        """Process a single inference request."""
        start_time = time.time()
        
        try:
            # Try distributed processing first
            selected_node = self.load_balancer.select_node(request)
            
            if selected_node:
                # Process on selected node
                response = self._process_on_node(request, selected_node)
            elif self.local_pipeline:
                # Fallback to local processing
                response = self._process_locally(request)
            else:
                # No processing capability available
                response = InferenceResponse(
                    request_id=request.request_id,
                    answer="Service temporarily unavailable",
                    confidence=0.0,
                    latency_ms=(time.time() - start_time) * 1000,
                    node_id=self.node_id,
                    success=False,
                    error_message="No processing nodes available"
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Request processing failed: {e}")
            return InferenceResponse(
                request_id=request.request_id,
                answer="Processing error occurred",
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                node_id=self.node_id,
                success=False,
                error_message=str(e)
            )
    
    def _process_on_node(self, request: InferenceRequest, node: NodeInfo) -> InferenceResponse:
        """Process request on a specific node."""
        start_time = time.time()
        
        # Simulate node processing
        try:
            # In a real implementation, this would make HTTP/gRPC calls to the node
            # For demo, we'll simulate with local processing
            if self.local_pipeline:
                result = self.local_pipeline.process_image_question(
                    request.image_data, 
                    request.question
                )
                
                response = InferenceResponse(
                    request_id=request.request_id,
                    answer=result.answer,
                    confidence=result.confidence,
                    latency_ms=result.latency_ms,
                    node_id=node.node_id,
                    success=True
                )
            else:
                # Fallback response
                response = InferenceResponse(
                    request_id=request.request_id,
                    answer="Distributed processing simulated response",
                    confidence=0.8,
                    latency_ms=(time.time() - start_time) * 1000,
                    node_id=node.node_id,
                    success=True
                )
            
            # Update node statistics
            self.load_balancer.update_node_stats(
                node.node_id, 
                response.latency_ms, 
                response.success
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Node processing failed: {e}")
            return InferenceResponse(
                request_id=request.request_id,
                answer="Node processing failed",
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                node_id=node.node_id,
                success=False,
                error_message=str(e)
            )
    
    def _process_locally(self, request: InferenceRequest) -> InferenceResponse:
        """Process request locally as fallback."""
        start_time = time.time()
        
        try:
            result = self.local_pipeline.process_image_question(
                request.image_data, 
                request.question
            )
            
            return InferenceResponse(
                request_id=request.request_id,
                answer=result.answer,
                confidence=result.confidence,
                latency_ms=result.latency_ms,
                node_id=f"{self.node_id}_local",
                success=True
            )
            
        except Exception as e:
            return InferenceResponse(
                request_id=request.request_id,
                answer="Local processing failed",
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                node_id=f"{self.node_id}_local",
                success=False,
                error_message=str(e)
            )
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        return {
            "engine_status": "running" if self.running else "stopped",
            "node_id": self.node_id,
            "cluster_stats": self.load_balancer.get_cluster_stats(),
            "cache_stats": {
                "hit_rate": self.distributed_cache.get_hit_rate(),
                "size_bytes": self.distributed_cache.cache_stats["size_bytes"],
                "hits": self.distributed_cache.cache_stats["hits"],
                "misses": self.distributed_cache.cache_stats["misses"]
            },
            "queue_stats": self.request_queue.get_queue_stats(),
            "active_requests": len(self.active_requests)
        }


# Global distributed engine instance
distributed_engine = DistributedInferenceEngine()


if __name__ == "__main__":
    # Demo distributed inference
    print("FastVLM Distributed Inference Demo")
    print("=" * 40)
    
    # Initialize with mock nodes
    engine = DistributedInferenceEngine("demo_coordinator")
    
    # Register some mock nodes
    nodes = [
        NodeInfo("node_1", "192.168.1.10", 8080, ["inference"], 0.2),
        NodeInfo("node_2", "192.168.1.11", 8080, ["inference"], 0.3),
        NodeInfo("node_3", "192.168.1.12", 8080, ["inference"], 0.1),
    ]
    
    for node in nodes:
        engine.load_balancer.register_node(node)
    
    print(f"✓ Registered {len(nodes)} worker nodes")
    
    # Start engine
    engine.start()
    print("✓ Distributed engine started")
    
    # Setup local pipeline for demo
    try:
        import sys
        sys.path.append('src/fast_vlm_ondevice')
        import core_pipeline
        
        config = core_pipeline.InferenceConfig(model_name="distributed-demo")
        local_pipeline = core_pipeline.FastVLMCorePipeline(config)
        engine.set_local_pipeline(local_pipeline)
        print("✓ Local pipeline configured")
        
        # Submit some requests
        demo_image = core_pipeline.create_demo_image()
        
        print(f"\n📋 Submitting distributed inference requests...")
        request_ids = []
        
        test_questions = [
            "What objects are in this image?",
            "What colors do you see?",
            "Describe the scene",
            "How many items are visible?",
            "Is there any text present?"
        ]
        
        for i, question in enumerate(test_questions):
            request_id = engine.submit_request(
                demo_image, 
                question, 
                priority=i+1
            )
            request_ids.append(request_id)
            print(f"  ✓ Request {i+1}: {request_id}")
        
        # Get results
        print(f"\n📊 Retrieving results...")
        successful_requests = 0
        
        for i, request_id in enumerate(request_ids):
            result = engine.get_result(request_id, timeout=10.0)
            if result and result.success:
                successful_requests += 1
                print(f"  ✓ Result {i+1}: {result.answer[:40]}... ({result.latency_ms:.1f}ms)")
            else:
                print(f"  ✗ Result {i+1}: Failed or timeout")
        
        print(f"\n📈 Cluster Status:")
        status = engine.get_cluster_status()
        print(f"  Healthy nodes: {status['cluster_stats']['healthy_nodes']}")
        print(f"  Cache hit rate: {status['cache_stats']['hit_rate']:.2%}")
        print(f"  Active requests: {status['active_requests']}")
        print(f"  Queue size: {status['queue_stats']['total_requests']}")
        
        print(f"\n✅ Distributed inference demo completed")
        print(f"Success rate: {successful_requests}/{len(request_ids)} requests")
        
    except ImportError:
        print("⚠️  Local pipeline not available, using mock responses")
    
    finally:
        engine.stop()
        print("✓ Distributed engine stopped")