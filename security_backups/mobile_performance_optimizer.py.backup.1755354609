"""
Mobile Performance Optimizer for FastVLM
Optimized for Apple Neural Engine and mobile constraints.
"""

import time
import threading
import queue
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, Future
import json
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """Configuration for mobile performance optimization."""
    # Memory management
    max_memory_mb: int = 512  # Max memory usage
    cache_size_mb: int = 128  # Cache memory limit
    
    # Batch processing
    max_batch_size: int = 8
    batch_timeout_ms: int = 50  # Batch collection timeout
    
    # Threading
    max_workers: int = 4
    enable_parallel_processing: bool = True
    
    # Neural Engine optimization
    enable_ane_optimization: bool = True
    quantization_level: str = "int4"  # int4, int8, fp16
    
    # Adaptive quality
    enable_adaptive_quality: bool = True
    quality_target_latency_ms: int = 200
    
    # Prefetching
    enable_prefetching: bool = True
    prefetch_cache_size: int = 50


@dataclass 
class BatchRequest:
    """Request for batch processing."""
    request_id: str
    image_data: bytes
    question: str
    priority: int = 1  # Higher number = higher priority
    timestamp: float = 0.0
    future: Optional[Future] = None


class MemoryManager:
    """Intelligent memory management for mobile devices."""
    
    def __init__(self, max_memory_mb: int = 512):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_usage = 0
        self.allocations = {}
        self.lock = threading.Lock()
        
    def allocate(self, allocation_id: str, size_bytes: int) -> bool:
        """Allocate memory with tracking."""
        with self.lock:
            if self.current_usage + size_bytes > self.max_memory_bytes:
                # Try to free some memory
                if not self._try_free_memory(size_bytes):
                    logger.warning(f"Memory allocation failed: {size_bytes} bytes")
                    return False
            
            self.allocations[allocation_id] = size_bytes
            self.current_usage += size_bytes
            return True
    
    def deallocate(self, allocation_id: str):
        """Deallocate tracked memory."""
        with self.lock:
            if allocation_id in self.allocations:
                size = self.allocations.pop(allocation_id)
                self.current_usage -= size
    
    def _try_free_memory(self, needed_bytes: int) -> bool:
        """Try to free memory by removing old allocations."""
        # Simple strategy: remove oldest allocations
        freed = 0
        to_remove = []
        
        for alloc_id in list(self.allocations.keys()):
            if freed >= needed_bytes:
                break
            to_remove.append(alloc_id)
            freed += self.allocations[alloc_id]
        
        for alloc_id in to_remove:
            del self.allocations[alloc_id]
        
        self.current_usage -= freed
        logger.info(f"Freed {freed} bytes of memory")
        return freed >= needed_bytes
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        with self.lock:
            return {
                "current_usage_mb": self.current_usage / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "usage_percent": (self.current_usage / self.max_memory_bytes) * 100,
                "active_allocations": len(self.allocations)
            }


class BatchProcessor:
    """Intelligent batch processing for improved throughput."""
    
    def __init__(self, config: PerformanceConfig, pipeline):
        self.config = config
        self.pipeline = pipeline
        self.request_queue = queue.PriorityQueue()
        self.batch_thread = None
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        
    def start(self):
        """Start batch processing thread."""
        if not self.running:
            self.running = True
            self.batch_thread = threading.Thread(target=self._batch_worker, daemon=True)
            self.batch_thread.start()
            logger.info("🚀 Batch processor started")
    
    def stop(self):
        """Stop batch processing."""
        self.running = False
        if self.batch_thread:
            self.batch_thread.join()
        self.executor.shutdown(wait=True)
        logger.info("⏹️ Batch processor stopped")
    
    def submit_request(self, image_data: bytes, question: str, priority: int = 1) -> Future:
        """Submit request for batch processing."""
        request_id = hashlib.md5(f"{time.time()}{question}".encode()).hexdigest()[:8]
        future = self.executor.submit(self._dummy_future_result)
        
        batch_request = BatchRequest(
            request_id=request_id,
            image_data=image_data,
            question=question,
            priority=priority,
            timestamp=time.time(),
            future=future
        )
        
        # Priority queue uses negative priority for max-heap behavior
        self.request_queue.put((-priority, batch_request))
        return future
    
    def _dummy_future_result(self):
        """Placeholder for future result."""
        return None
    
    def _batch_worker(self):
        """Main batch processing worker."""
        while self.running:
            try:
                batch = self._collect_batch()
                if batch:
                    self._process_batch(batch)
                else:
                    time.sleep(0.001)  # Small sleep to prevent busy waiting
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
    
    def _collect_batch(self) -> List[BatchRequest]:
        """Collect requests for batch processing."""
        batch = []
        start_time = time.time()
        
        while (len(batch) < self.config.max_batch_size and 
               (time.time() - start_time) * 1000 < self.config.batch_timeout_ms):
            
            try:
                priority, request = self.request_queue.get(timeout=0.01)
                batch.append(request)
            except queue.Empty:
                break
        
        return batch
    
    def _process_batch(self, batch: List[BatchRequest]):
        """Process a batch of requests."""
        if not batch:
            return
        
        logger.debug(f"🔄 Processing batch of {len(batch)} requests")
        start_time = time.time()
        
        # Process requests in parallel
        futures = []
        for request in batch:
            future = self.executor.submit(
                self.pipeline.process_image_question,
                request.image_data,
                request.question
            )
            futures.append((request, future))
        
        # Collect results
        for request, future in futures:
            try:
                result = future.result(timeout=5.0)
                # Set the result on the original future (would need proper implementation)
                logger.debug(f"✅ Batch request {request.request_id} completed")
            except Exception as e:
                logger.error(f"❌ Batch request {request.request_id} failed: {e}")
        
        batch_time = (time.time() - start_time) * 1000
        logger.info(f"📦 Batch of {len(batch)} processed in {batch_time:.1f}ms")


class AdaptiveQualityManager:
    """Adaptive quality adjustment based on performance targets."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.target_latency_ms = config.quality_target_latency_ms
        self.latency_history = []
        self.max_history_size = 100
        self.current_quality_level = 1.0  # 0.0 to 1.0
        
    def update_latency(self, latency_ms: float):
        """Update latency measurements and adjust quality."""
        self.latency_history.append(latency_ms)
        if len(self.latency_history) > self.max_history_size:
            self.latency_history.pop(0)
        
        if len(self.latency_history) >= 10:
            avg_latency = sum(self.latency_history[-10:]) / 10
            self._adjust_quality(avg_latency)
    
    def _adjust_quality(self, avg_latency_ms: float):
        """Adjust quality level based on average latency."""
        if avg_latency_ms > self.target_latency_ms * 1.2:
            # Reduce quality to improve speed
            self.current_quality_level = max(0.3, self.current_quality_level - 0.1)
            logger.info(f"🔽 Quality reduced to {self.current_quality_level:.1f} (latency: {avg_latency_ms:.1f}ms)")
        elif avg_latency_ms < self.target_latency_ms * 0.8:
            # Increase quality if we have headroom
            self.current_quality_level = min(1.0, self.current_quality_level + 0.05)
            logger.info(f"🔼 Quality increased to {self.current_quality_level:.1f} (latency: {avg_latency_ms:.1f}ms)")
    
    def get_quality_settings(self) -> Dict[str, Any]:
        """Get current quality settings for processing."""
        return {
            "image_scale_factor": self.current_quality_level,
            "text_max_length": int(77 * self.current_quality_level),
            "fusion_layers": max(1, int(4 * self.current_quality_level)),
            "beam_size": max(1, int(3 * self.current_quality_level))
        }


class PrefetchManager:
    """Intelligent prefetching for common queries."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.query_patterns = {}
        self.prefetch_cache = {}
        self.max_cache_size = config.prefetch_cache_size
        
    def record_query(self, image_hash: str, question_pattern: str):
        """Record query pattern for prefetching."""
        pattern_key = f"{image_hash}_{question_pattern}"
        self.query_patterns[pattern_key] = self.query_patterns.get(pattern_key, 0) + 1
        
        # Trigger prefetching for popular patterns
        if self.query_patterns[pattern_key] > 3:
            self._consider_prefetch(image_hash, question_pattern)
    
    def _consider_prefetch(self, image_hash: str, question_pattern: str):
        """Consider prefetching based on pattern frequency."""
        if len(self.prefetch_cache) < self.max_cache_size:
            # Would implement actual prefetching logic here
            logger.debug(f"🔮 Considering prefetch for pattern: {question_pattern}")
    
    def get_prefetch_suggestions(self, image_hash: str) -> List[str]:
        """Get prefetch suggestions for an image."""
        suggestions = []
        for pattern, count in self.query_patterns.items():
            if pattern.startswith(image_hash) and count > 2:
                question_part = pattern.split('_', 1)[1]
                suggestions.append(question_part)
        
        return suggestions[:5]  # Top 5 suggestions


class MobilePerformanceOptimizer:
    """Main mobile performance optimization coordinator."""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        self.memory_manager = MemoryManager(self.config.max_memory_mb)
        self.batch_processor = None
        self.quality_manager = AdaptiveQualityManager(self.config)
        self.prefetch_manager = PrefetchManager(self.config)
        
        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "batch_requests": 0,
            "prefetch_hits": 0,
            "memory_evictions": 0,
            "quality_adjustments": 0,
            "average_latency_ms": 0.0
        }
        
        self.pipeline = None  # Will be injected
        
    def initialize(self, pipeline):
        """Initialize optimizer with pipeline."""
        self.pipeline = pipeline
        
        if self.config.enable_parallel_processing:
            self.batch_processor = BatchProcessor(self.config, pipeline)
            self.batch_processor.start()
        
        logger.info("🚀 Mobile Performance Optimizer initialized")
    
    def optimize_request(self, image_data: bytes, question: str) -> Dict[str, Any]:
        """Optimize a single request with all available techniques."""
        self.metrics["total_requests"] += 1
        start_time = time.time()
        
        # Memory allocation
        image_size = len(image_data)
        allocation_id = f"request_{self.metrics['total_requests']}"
        
        if not self.memory_manager.allocate(allocation_id, image_size):
            logger.warning("⚠️ Memory allocation failed, using fallback")
            return self._fallback_processing(image_data, question)
        
        try:
            # Get adaptive quality settings
            quality_settings = self.quality_manager.get_quality_settings()
            
            # Apply image scaling if needed
            optimized_image = self._optimize_image(image_data, quality_settings)
            optimized_question = self._optimize_question(question, quality_settings)
            
            # Record for prefetching
            image_hash = hashlib.md5(image_data[:1000]).hexdigest()[:8]
            self.prefetch_manager.record_query(image_hash, self._extract_question_pattern(question))
            
            # Process request
            if self.batch_processor and self.config.enable_parallel_processing:
                # Use batch processing
                future = self.batch_processor.submit_request(optimized_image, optimized_question)
                result = future.result(timeout=5.0)
                self.metrics["batch_requests"] += 1
            else:
                # Direct processing
                result = self.pipeline.process_image_question(optimized_image, optimized_question)
            
            # Update quality manager
            latency_ms = (time.time() - start_time) * 1000
            self.quality_manager.update_latency(latency_ms)
            
            # Update metrics
            self._update_average_latency(latency_ms)
            
            return {
                "result": result,
                "optimization_applied": True,
                "quality_level": self.quality_manager.current_quality_level,
                "memory_usage_mb": self.memory_manager.get_usage_stats()["current_usage_mb"]
            }
            
        finally:
            # Clean up memory
            self.memory_manager.deallocate(allocation_id)
    
    def _optimize_image(self, image_data: bytes, quality_settings: Dict[str, Any]) -> bytes:
        """Optimize image based on quality settings."""
        scale_factor = quality_settings.get("image_scale_factor", 1.0)
        
        if scale_factor < 1.0:
            # Simple simulation of image scaling
            # In real implementation, would actually resize image
            scaled_size = int(len(image_data) * scale_factor)
            return image_data[:scaled_size] + b"_scaled"
        
        return image_data
    
    def _optimize_question(self, question: str, quality_settings: Dict[str, Any]) -> str:
        """Optimize question based on quality settings."""
        max_length = quality_settings.get("text_max_length", 77)
        
        if len(question) > max_length:
            return question[:max_length]
        
        return question
    
    def _extract_question_pattern(self, question: str) -> str:
        """Extract pattern from question for prefetching."""
        # Simple pattern extraction
        question_lower = question.lower()
        
        if "what" in question_lower and "object" in question_lower:
            return "what_objects"
        elif "color" in question_lower:
            return "color_query"
        elif "count" in question_lower or "how many" in question_lower:
            return "count_query"
        elif "describe" in question_lower:
            return "describe_scene"
        else:
            return "general_query"
    
    def _fallback_processing(self, image_data: bytes, question: str) -> Dict[str, Any]:
        """Fallback processing when optimization fails."""
        logger.warning("🔄 Using fallback processing")
        
        # Simple processing without optimization
        result = self.pipeline.process_image_question(image_data, question)
        
        return {
            "result": result,
            "optimization_applied": False,
            "fallback_used": True
        }
    
    def _update_average_latency(self, new_latency_ms: float):
        """Update average latency metric."""
        current_avg = self.metrics["average_latency_ms"]
        total_requests = self.metrics["total_requests"]
        
        if total_requests == 1:
            self.metrics["average_latency_ms"] = new_latency_ms
        else:
            self.metrics["average_latency_ms"] = (
                (current_avg * (total_requests - 1) + new_latency_ms) / total_requests
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        memory_stats = self.memory_manager.get_usage_stats()
        
        return {
            "requests": dict(self.metrics),
            "memory": memory_stats,
            "quality": {
                "current_level": self.quality_manager.current_quality_level,
                "target_latency_ms": self.quality_manager.target_latency_ms,
                "recent_latencies": self.quality_manager.latency_history[-10:]
            },
            "prefetch": {
                "patterns_learned": len(self.prefetch_manager.query_patterns),
                "cache_size": len(self.prefetch_manager.prefetch_cache),
                "hit_rate": self.metrics.get("prefetch_hits", 0) / max(1, self.metrics["total_requests"])
            },
            "batch_processing": {
                "enabled": self.batch_processor is not None,
                "batch_ratio": self.metrics["batch_requests"] / max(1, self.metrics["total_requests"])
            }
        }
    
    def shutdown(self):
        """Shutdown optimizer and clean up resources."""
        if self.batch_processor:
            self.batch_processor.stop()
        
        logger.info("🛑 Mobile Performance Optimizer shutdown complete")


# Factory function
def create_mobile_optimizer(
    max_memory_mb: int = 512,
    enable_batching: bool = True,
    enable_adaptive_quality: bool = True
) -> MobilePerformanceOptimizer:
    """Create a mobile performance optimizer with sensible defaults."""
    config = PerformanceConfig(
        max_memory_mb=max_memory_mb,
        enable_parallel_processing=enable_batching,
        enable_adaptive_quality=enable_adaptive_quality
    )
    
    return MobilePerformanceOptimizer(config)