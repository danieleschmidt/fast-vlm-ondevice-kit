"""
Auto-scaling and adaptive resource management for FastVLM.

Implements intelligent scaling decisions, resource optimization,
and adaptive configuration based on load patterns and performance metrics.
"""

import time
import logging
import threading
import json
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from collections import deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    request_rate: float = 0.0
    average_latency: float = 0.0
    queue_size: int = 0
    error_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingTarget:
    """Target configuration for scaling."""
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    target_latency_ms: float = 250.0
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    cooldown_period: float = 300.0  # 5 minutes


@dataclass
class ResourceConfiguration:
    """Resource configuration for instances."""
    cpu_cores: float = 1.0
    memory_gb: float = 2.0
    batch_size: int = 1
    cache_size_mb: int = 100
    worker_threads: int = 2
    priority_queue_size: int = 100


class MetricsCollector:
    """Collects and analyzes metrics for scaling decisions."""
    
    def __init__(self, window_size: int = 60):
        """Initialize metrics collector."""
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.lock = threading.Lock()
        
        logger.info(f"Metrics collector initialized with window size {window_size}")
    
    def record_metrics(self, metrics: ScalingMetrics):
        """Record new metrics."""
        with self.lock:
            self.metrics_history.append(metrics)
    
    def get_average_metrics(self, time_window: float = 300.0) -> Optional[ScalingMetrics]:
        """Get average metrics over time window."""
        cutoff_time = time.time() - time_window
        
        with self.lock:
            recent_metrics = [
                m for m in self.metrics_history 
                if m.timestamp >= cutoff_time
            ]
        
        if not recent_metrics:
            return None
        
        return ScalingMetrics(
            cpu_usage=statistics.mean(m.cpu_usage for m in recent_metrics),
            memory_usage=statistics.mean(m.memory_usage for m in recent_metrics),
            request_rate=statistics.mean(m.request_rate for m in recent_metrics),
            average_latency=statistics.mean(m.average_latency for m in recent_metrics),
            queue_size=int(statistics.mean(m.queue_size for m in recent_metrics)),
            error_rate=statistics.mean(m.error_rate for m in recent_metrics),
            timestamp=time.time()
        )
    
    def get_trend(self, metric_name: str, time_window: float = 600.0) -> float:
        """Get trend for a specific metric (positive = increasing)."""
        cutoff_time = time.time() - time_window
        
        with self.lock:
            recent_metrics = [
                m for m in self.metrics_history 
                if m.timestamp >= cutoff_time
            ]
        
        if len(recent_metrics) < 2:
            return 0.0
        
        values = [getattr(m, metric_name) for m in recent_metrics]
        
        # Simple linear trend calculation
        x = list(range(len(values)))
        y = values
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        # Calculate slope
        slope = (n * sum_xy - sum_x * sum_y) / max(1, n * sum_x2 - sum_x ** 2)
        return slope


class ScalingDecisionEngine:
    """Makes intelligent scaling decisions based on metrics."""
    
    def __init__(self, target: ScalingTarget):
        """Initialize scaling decision engine."""
        self.target = target
        self.last_scale_time = 0
        self.current_instances = target.min_instances
        
        logger.info("Scaling decision engine initialized")
    
    def should_scale_up(self, metrics: ScalingMetrics, trend: Dict[str, float]) -> bool:
        """Determine if we should scale up."""
        if self.current_instances >= self.target.max_instances:
            return False
        
        # Check cooldown period
        if time.time() - self.last_scale_time < self.target.cooldown_period:
            return False
        
        # Check multiple conditions
        conditions_met = 0
        total_conditions = 0
        
        # CPU utilization
        if metrics.cpu_usage > self.target.target_cpu_utilization:
            conditions_met += 1
        total_conditions += 1
        
        # Memory utilization
        if metrics.memory_usage > self.target.target_memory_utilization:
            conditions_met += 1
        total_conditions += 1
        
        # Latency
        if metrics.average_latency > self.target.target_latency_ms:
            conditions_met += 1
        total_conditions += 1
        
        # Queue size
        if metrics.queue_size > 10:
            conditions_met += 1
        total_conditions += 1
        
        # Trend analysis
        if trend.get('request_rate', 0) > 0.1:  # Increasing request rate
            conditions_met += 1
        total_conditions += 1
        
        # Scale up if threshold of conditions are met
        threshold_met = conditions_met / total_conditions >= self.target.scale_up_threshold
        
        if threshold_met:
            logger.info(f"Scale up triggered: {conditions_met}/{total_conditions} conditions met")
        
        return threshold_met
    
    def should_scale_down(self, metrics: ScalingMetrics, trend: Dict[str, float]) -> bool:
        """Determine if we should scale down."""
        if self.current_instances <= self.target.min_instances:
            return False
        
        # Check cooldown period
        if time.time() - self.last_scale_time < self.target.cooldown_period:
            return False
        
        # Check multiple conditions for scale down
        conditions_met = 0
        total_conditions = 0
        
        # CPU utilization
        if metrics.cpu_usage < self.target.target_cpu_utilization * 0.5:
            conditions_met += 1
        total_conditions += 1
        
        # Memory utilization
        if metrics.memory_usage < self.target.target_memory_utilization * 0.5:
            conditions_met += 1
        total_conditions += 1
        
        # Latency
        if metrics.average_latency < self.target.target_latency_ms * 0.5:
            conditions_met += 1
        total_conditions += 1
        
        # Queue size
        if metrics.queue_size < 2:
            conditions_met += 1
        total_conditions += 1
        
        # Trend analysis
        if trend.get('request_rate', 0) < -0.05:  # Decreasing request rate
            conditions_met += 1
        total_conditions += 1
        
        # Scale down if threshold of conditions are met
        threshold_met = conditions_met / total_conditions >= (1 - self.target.scale_down_threshold)
        
        if threshold_met:
            logger.info(f"Scale down triggered: {conditions_met}/{total_conditions} conditions met")
        
        return threshold_met
    
    def execute_scaling(self, action: str) -> bool:
        """Execute scaling action."""
        if action == "scale_up" and self.current_instances < self.target.max_instances:
            self.current_instances += 1
            self.last_scale_time = time.time()
            logger.info(f"Scaled UP to {self.current_instances} instances")
            return True
        elif action == "scale_down" and self.current_instances > self.target.min_instances:
            self.current_instances -= 1
            self.last_scale_time = time.time()
            logger.info(f"Scaled DOWN to {self.current_instances} instances")
            return True
        
        return False


class ResourceOptimizer:
    """Optimizes resource configuration based on workload patterns."""
    
    def __init__(self):
        """Initialize resource optimizer."""
        self.optimization_history = []
        self.current_config = ResourceConfiguration()
        
        logger.info("Resource optimizer initialized")
    
    def optimize_configuration(self, metrics: ScalingMetrics, 
                             workload_pattern: Dict[str, Any]) -> ResourceConfiguration:
        """Optimize resource configuration based on current metrics."""
        new_config = ResourceConfiguration(
            cpu_cores=self.current_config.cpu_cores,
            memory_gb=self.current_config.memory_gb,
            batch_size=self.current_config.batch_size,
            cache_size_mb=self.current_config.cache_size_mb,
            worker_threads=self.current_config.worker_threads,
            priority_queue_size=self.current_config.priority_queue_size
        )
        
        # Optimize based on CPU usage
        if metrics.cpu_usage > 80:
            new_config.cpu_cores = min(8.0, self.current_config.cpu_cores * 1.2)
        elif metrics.cpu_usage < 30:
            new_config.cpu_cores = max(0.5, self.current_config.cpu_cores * 0.9)
        
        # Optimize based on memory usage
        if metrics.memory_usage > 85:
            new_config.memory_gb = min(16.0, self.current_config.memory_gb * 1.3)
            new_config.cache_size_mb = min(500, self.current_config.cache_size_mb * 1.2)
        elif metrics.memory_usage < 40:
            new_config.memory_gb = max(1.0, self.current_config.memory_gb * 0.9)
            new_config.cache_size_mb = max(50, self.current_config.cache_size_mb * 0.9)
        
        # Optimize based on latency
        if metrics.average_latency > 500:
            new_config.worker_threads = min(8, self.current_config.worker_threads + 1)
            new_config.batch_size = max(1, self.current_config.batch_size - 1)
        elif metrics.average_latency < 100:
            new_config.batch_size = min(8, self.current_config.batch_size + 1)
        
        # Optimize based on queue size
        if metrics.queue_size > 50:
            new_config.priority_queue_size = min(1000, self.current_config.priority_queue_size * 1.5)
            new_config.worker_threads = min(8, self.current_config.worker_threads + 1)
        
        # Record optimization
        self.optimization_history.append({
            "timestamp": time.time(),
            "old_config": self.current_config,
            "new_config": new_config,
            "metrics": metrics,
            "reason": self._get_optimization_reason(metrics)
        })
        
        self.current_config = new_config
        return new_config
    
    def _get_optimization_reason(self, metrics: ScalingMetrics) -> str:
        """Get reason for optimization."""
        reasons = []
        
        if metrics.cpu_usage > 80:
            reasons.append("high_cpu")
        elif metrics.cpu_usage < 30:
            reasons.append("low_cpu")
        
        if metrics.memory_usage > 85:
            reasons.append("high_memory")
        elif metrics.memory_usage < 40:
            reasons.append("low_memory")
        
        if metrics.average_latency > 500:
            reasons.append("high_latency")
        elif metrics.average_latency < 100:
            reasons.append("low_latency")
        
        if metrics.queue_size > 50:
            reasons.append("large_queue")
        
        return ",".join(reasons) if reasons else "routine_optimization"


class PredictiveScaler:
    """Predictive scaling based on historical patterns."""
    
    def __init__(self):
        """Initialize predictive scaler."""
        self.historical_patterns = {}
        self.prediction_accuracy = 0.0
        
        logger.info("Predictive scaler initialized")
    
    def learn_pattern(self, timestamp: float, metrics: ScalingMetrics):
        """Learn from historical load patterns."""
        # Extract time features
        time_struct = time.localtime(timestamp)
        hour = time_struct.tm_hour
        day_of_week = time_struct.tm_wday
        
        # Create pattern key
        pattern_key = f"{day_of_week}_{hour}"
        
        if pattern_key not in self.historical_patterns:
            self.historical_patterns[pattern_key] = []
        
        self.historical_patterns[pattern_key].append({
            "request_rate": metrics.request_rate,
            "cpu_usage": metrics.cpu_usage,
            "memory_usage": metrics.memory_usage,
            "timestamp": timestamp
        })
        
        # Keep only recent data (last 30 days)
        cutoff = timestamp - (30 * 24 * 3600)
        self.historical_patterns[pattern_key] = [
            p for p in self.historical_patterns[pattern_key]
            if p["timestamp"] > cutoff
        ]
    
    def predict_load(self, future_time: float) -> Optional[Dict[str, float]]:
        """Predict load at future time."""
        time_struct = time.localtime(future_time)
        hour = time_struct.tm_hour
        day_of_week = time_struct.tm_wday
        pattern_key = f"{day_of_week}_{hour}"
        
        if pattern_key not in self.historical_patterns:
            return None
        
        patterns = self.historical_patterns[pattern_key]
        if not patterns:
            return None
        
        # Calculate averages for this time pattern
        return {
            "predicted_request_rate": statistics.mean(p["request_rate"] for p in patterns),
            "predicted_cpu_usage": statistics.mean(p["cpu_usage"] for p in patterns),
            "predicted_memory_usage": statistics.mean(p["memory_usage"] for p in patterns),
            "confidence": min(1.0, len(patterns) / 10.0)  # More data = higher confidence
        }
    
    def should_preemptive_scale(self, current_time: float) -> Optional[str]:
        """Determine if preemptive scaling is needed."""
        # Look ahead 15 minutes
        future_time = current_time + (15 * 60)
        prediction = self.predict_load(future_time)
        
        if not prediction or prediction["confidence"] < 0.3:
            return None
        
        # If predicted load is significantly higher, recommend scale up
        if (prediction["predicted_request_rate"] > 50 and 
            prediction["predicted_cpu_usage"] > 70):
            return "preemptive_scale_up"
        
        # If predicted load is significantly lower, recommend scale down
        if (prediction["predicted_request_rate"] < 10 and 
            prediction["predicted_cpu_usage"] < 30):
            return "preemptive_scale_down"
        
        return None


class AutoScalingManager:
    """Main auto-scaling manager that coordinates all scaling activities."""
    
    def __init__(self, target: ScalingTarget):
        """Initialize auto-scaling manager."""
        self.target = target
        self.metrics_collector = MetricsCollector()
        self.decision_engine = ScalingDecisionEngine(target)
        self.resource_optimizer = ResourceOptimizer()
        self.predictive_scaler = PredictiveScaler()
        
        self.scaling_callbacks = []
        self.optimization_callbacks = []
        
        self.running = False
        self.monitor_thread = None
        
        logger.info("Auto-scaling manager initialized")
    
    def add_scaling_callback(self, callback: Callable[[str, int], None]):
        """Add callback for scaling events."""
        self.scaling_callbacks.append(callback)
    
    def add_optimization_callback(self, callback: Callable[[ResourceConfiguration], None]):
        """Add callback for resource optimization events."""
        self.optimization_callbacks.append(callback)
    
    def start(self):
        """Start auto-scaling monitoring."""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Auto-scaling monitoring started")
    
    def stop(self):
        """Stop auto-scaling monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("Auto-scaling monitoring stopped")
    
    def record_metrics(self, cpu_usage: float, memory_usage: float, 
                      request_rate: float, average_latency: float, 
                      queue_size: int, error_rate: float = 0.0):
        """Record current system metrics."""
        metrics = ScalingMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            request_rate=request_rate,
            average_latency=average_latency,
            queue_size=queue_size,
            error_rate=error_rate
        )
        
        self.metrics_collector.record_metrics(metrics)
        self.predictive_scaler.learn_pattern(time.time(), metrics)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._check_scaling_conditions()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Auto-scaling monitor error: {e}")
    
    def _check_scaling_conditions(self):
        """Check if scaling is needed."""
        # Get recent metrics
        avg_metrics = self.metrics_collector.get_average_metrics(300)  # 5 minutes
        if not avg_metrics:
            return
        
        # Get trends
        trends = {
            'request_rate': self.metrics_collector.get_trend('request_rate'),
            'cpu_usage': self.metrics_collector.get_trend('cpu_usage'),
            'memory_usage': self.metrics_collector.get_trend('memory_usage'),
            'average_latency': self.metrics_collector.get_trend('average_latency')
        }
        
        # Check for scaling decisions
        if self.decision_engine.should_scale_up(avg_metrics, trends):
            if self.decision_engine.execute_scaling("scale_up"):
                self._notify_scaling_callbacks("scale_up", self.decision_engine.current_instances)
        
        elif self.decision_engine.should_scale_down(avg_metrics, trends):
            if self.decision_engine.execute_scaling("scale_down"):
                self._notify_scaling_callbacks("scale_down", self.decision_engine.current_instances)
        
        # Check for predictive scaling
        current_time = time.time()
        preemptive_action = self.predictive_scaler.should_preemptive_scale(current_time)
        if preemptive_action:
            logger.info(f"Predictive scaling recommendation: {preemptive_action}")
        
        # Check for resource optimization
        workload_pattern = self._analyze_workload_pattern(avg_metrics)
        if self._should_optimize_resources(avg_metrics):
            new_config = self.resource_optimizer.optimize_configuration(avg_metrics, workload_pattern)
            self._notify_optimization_callbacks(new_config)
    
    def _analyze_workload_pattern(self, metrics: ScalingMetrics) -> Dict[str, Any]:
        """Analyze current workload pattern."""
        return {
            "load_level": "high" if metrics.cpu_usage > 70 else "medium" if metrics.cpu_usage > 40 else "low",
            "memory_intensive": metrics.memory_usage > 70,
            "latency_sensitive": metrics.average_latency > 300,
            "queue_backed_up": metrics.queue_size > 20
        }
    
    def _should_optimize_resources(self, metrics: ScalingMetrics) -> bool:
        """Determine if resource optimization is needed."""
        # Optimize every 10 minutes or when metrics are extreme
        last_optimization = getattr(self, '_last_optimization_time', 0)
        time_since_optimization = time.time() - last_optimization
        
        if time_since_optimization > 600:  # 10 minutes
            self._last_optimization_time = time.time()
            return True
        
        # Immediate optimization for extreme conditions
        if (metrics.cpu_usage > 90 or metrics.memory_usage > 90 or 
            metrics.average_latency > 1000):
            self._last_optimization_time = time.time()
            return True
        
        return False
    
    def _notify_scaling_callbacks(self, action: str, instances: int):
        """Notify scaling callbacks."""
        for callback in self.scaling_callbacks:
            try:
                callback(action, instances)
            except Exception as e:
                logger.error(f"Scaling callback error: {e}")
    
    def _notify_optimization_callbacks(self, config: ResourceConfiguration):
        """Notify optimization callbacks."""
        for callback in self.optimization_callbacks:
            try:
                callback(config)
            except Exception as e:
                logger.error(f"Optimization callback error: {e}")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        avg_metrics = self.metrics_collector.get_average_metrics(300)
        
        return {
            "current_instances": self.decision_engine.current_instances,
            "target_config": {
                "min_instances": self.target.min_instances,
                "max_instances": self.target.max_instances,
                "target_cpu": self.target.target_cpu_utilization,
                "target_memory": self.target.target_memory_utilization,
                "target_latency": self.target.target_latency_ms
            },
            "current_metrics": {
                "cpu_usage": avg_metrics.cpu_usage if avg_metrics else 0,
                "memory_usage": avg_metrics.memory_usage if avg_metrics else 0,
                "request_rate": avg_metrics.request_rate if avg_metrics else 0,
                "average_latency": avg_metrics.average_latency if avg_metrics else 0,
                "queue_size": avg_metrics.queue_size if avg_metrics else 0
            },
            "resource_config": {
                "cpu_cores": self.resource_optimizer.current_config.cpu_cores,
                "memory_gb": self.resource_optimizer.current_config.memory_gb,
                "batch_size": self.resource_optimizer.current_config.batch_size,
                "cache_size_mb": self.resource_optimizer.current_config.cache_size_mb,
                "worker_threads": self.resource_optimizer.current_config.worker_threads
            },
            "last_scale_time": self.decision_engine.last_scale_time,
            "monitoring_active": self.running
        }


if __name__ == "__main__":
    # Demo auto-scaling capabilities
    print("FastVLM Auto-Scaling Demo")
    print("=" * 40)
    
    # Create scaling target
    target = ScalingTarget(
        min_instances=1,
        max_instances=5,
        target_cpu_utilization=70.0,
        target_latency_ms=250.0,
        cooldown_period=60.0  # Shorter for demo
    )
    
    # Initialize auto-scaling manager
    manager = AutoScalingManager(target)
    
    # Add callbacks
    def scaling_callback(action: str, instances: int):
        print(f"🔄 Scaling event: {action} -> {instances} instances")
    
    def optimization_callback(config: ResourceConfiguration):
        print(f"⚙️  Resource optimization: CPU={config.cpu_cores:.1f}, Memory={config.memory_gb:.1f}GB")
    
    manager.add_scaling_callback(scaling_callback)
    manager.add_optimization_callback(optimization_callback)
    
    # Start monitoring
    manager.start()
    print("✓ Auto-scaling monitoring started")
    
    # Simulate varying load
    print("\n📊 Simulating load patterns...")
    
    import random
    
    for minute in range(5):
        # Simulate increasing load
        cpu_usage = min(95, 20 + minute * 15 + random.uniform(-5, 5))
        memory_usage = min(90, 30 + minute * 10 + random.uniform(-5, 5))
        request_rate = max(0, minute * 20 + random.uniform(-10, 10))
        latency = 100 + minute * 50 + random.uniform(-20, 20)
        queue_size = max(0, int(minute * 10 + random.uniform(-5, 5)))
        
        manager.record_metrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            request_rate=request_rate,
            average_latency=latency,
            queue_size=queue_size,
            error_rate=random.uniform(0, 5)
        )
        
        print(f"  Minute {minute+1}: CPU={cpu_usage:.1f}%, Memory={memory_usage:.1f}%, "
              f"Latency={latency:.1f}ms, Queue={queue_size}")
        
        time.sleep(2)  # Compressed time for demo
    
    # Show final status
    print(f"\n📈 Final Scaling Status:")
    status = manager.get_scaling_status()
    print(f"  Current instances: {status['current_instances']}")
    print(f"  CPU usage: {status['current_metrics']['cpu_usage']:.1f}%")
    print(f"  Memory usage: {status['current_metrics']['memory_usage']:.1f}%")
    print(f"  Average latency: {status['current_metrics']['average_latency']:.1f}ms")
    print(f"  Resource config: {status['resource_config']['cpu_cores']:.1f} CPU cores, "
          f"{status['resource_config']['memory_gb']:.1f}GB RAM")
    
    manager.stop()
    print("✓ Auto-scaling demo completed")