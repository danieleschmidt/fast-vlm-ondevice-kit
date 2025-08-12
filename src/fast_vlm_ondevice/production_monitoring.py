"""
Production monitoring and observability for FastVLM.

Provides comprehensive monitoring, alerting, and telemetry
for production deployments.
"""

import time
import logging
import threading
import json
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque, defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class AlertRule:
    """Configuration for an alert rule."""
    name: str
    metric_name: str
    threshold: float
    comparison: str = ">"  # >, <, >=, <=, ==, !=
    duration_seconds: float = 60.0
    enabled: bool = True
    severity: str = "warning"  # info, warning, error, critical


class MetricsCollector:
    """Collects and stores metrics for monitoring."""
    
    def __init__(self, max_points: int = 10000):
        """Initialize metrics collector."""
        self.max_points = max_points
        self.metrics = defaultdict(lambda: deque(maxlen=max_points))
        self.lock = threading.Lock()
        self.metadata = {}
        
        logger.info(f"Metrics collector initialized with max {max_points} points per metric")
    
    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None, unit: str = ""):
        """Record a metric value."""
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
            unit=unit
        )
        
        with self.lock:
            self.metrics[name].append(point)
    
    def increment(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        current_value = self.get_latest_value(name, default=0.0)
        self.record(name, current_value + value, tags, "count")
    
    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None, unit: str = ""):
        """Record a gauge metric."""
        self.record(name, value, tags, unit)
    
    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram metric."""
        self.record(name, value, tags, "histogram")
    
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        return TimingContext(self, name, tags)
    
    def get_latest_value(self, name: str, default: float = 0.0) -> float:
        """Get the latest value for a metric."""
        with self.lock:
            if name in self.metrics and self.metrics[name]:
                return self.metrics[name][-1].value
            return default
    
    def get_metric_summary(self, name: str, time_window_seconds: float = 300) -> Dict[str, float]:
        """Get statistical summary for a metric over time window."""
        cutoff_time = time.time() - time_window_seconds
        
        with self.lock:
            if name not in self.metrics:
                return {}
            
            values = [
                point.value for point in self.metrics[name]
                if point.timestamp >= cutoff_time
            ]
        
        if not values:
            return {}
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p95": self._percentile(values, 0.95),
            "p99": self._percentile(values, 0.99)
        }
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_all_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all metrics as serializable data."""
        with self.lock:
            result = {}
            for name, points in self.metrics.items():
                result[name] = [
                    {
                        "value": point.value,
                        "timestamp": point.timestamp,
                        "tags": point.tags,
                        "unit": point.unit
                    }
                    for point in points
                ]
            return result
    
    def clear_metrics(self):
        """Clear all collected metrics."""
        with self.lock:
            self.metrics.clear()
        logger.info("All metrics cleared")


class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, name: str, tags: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.name = name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (time.time() - self.start_time) * 1000  # Convert to milliseconds
            self.collector.record(self.name, duration, self.tags, "ms")


class AlertManager:
    """Manages alerts based on metric thresholds."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize alert manager."""
        self.metrics_collector = metrics_collector
        self.rules = {}
        self.alert_history = deque(maxlen=1000)
        self.active_alerts = {}
        self.notification_callbacks = []
        self.check_interval = 10.0  # seconds
        self.monitoring = False
        self.monitor_thread = None
        
        logger.info("Alert manager initialized")
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.rules[rule.name] = rule
        logger.info(f"Alert rule added: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove an alert rule."""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Alert rule removed: {rule_name}")
    
    def add_notification_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for alert notifications."""
        self.notification_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start alert monitoring."""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Alert monitoring started")
    
    def stop_monitoring(self):
        """Stop alert monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Alert monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                self._check_alerts()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in alert monitoring: {e}")
    
    def _check_alerts(self):
        """Check all alert rules."""
        current_time = time.time()
        
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            try:
                should_alert = self._evaluate_rule(rule, current_time)
                
                if should_alert and rule_name not in self.active_alerts:
                    # New alert
                    alert = {
                        "rule_name": rule_name,
                        "metric_name": rule.metric_name,
                        "severity": rule.severity,
                        "message": f"Alert: {rule.name}",
                        "timestamp": current_time,
                        "status": "active"
                    }
                    
                    self.active_alerts[rule_name] = alert
                    self.alert_history.append(alert)
                    self._notify(alert)
                    
                elif not should_alert and rule_name in self.active_alerts:
                    # Alert resolved
                    alert = self.active_alerts[rule_name].copy()
                    alert["status"] = "resolved"
                    alert["resolved_timestamp"] = current_time
                    
                    del self.active_alerts[rule_name]
                    self.alert_history.append(alert)
                    self._notify(alert)
                    
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule_name}: {e}")
    
    def _evaluate_rule(self, rule: AlertRule, current_time: float) -> bool:
        """Evaluate if an alert rule should trigger."""
        # Get recent values for the metric
        summary = self.metrics_collector.get_metric_summary(
            rule.metric_name, 
            rule.duration_seconds
        )
        
        if not summary:
            return False
        
        # Use mean value for evaluation
        value = summary.get("mean", 0.0)
        
        # Compare against threshold
        if rule.comparison == ">":
            return value > rule.threshold
        elif rule.comparison == "<":
            return value < rule.threshold
        elif rule.comparison == ">=":
            return value >= rule.threshold
        elif rule.comparison == "<=":
            return value <= rule.threshold
        elif rule.comparison == "==":
            return abs(value - rule.threshold) < 0.001  # Float equality
        elif rule.comparison == "!=":
            return abs(value - rule.threshold) >= 0.001
        
        return False
    
    def _notify(self, alert: Dict[str, Any]):
        """Send alert notification."""
        for callback in self.notification_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in notification callback: {e}")
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent alert history."""
        return list(self.alert_history)[-limit:]


class PerformanceProfiler:
    """Profiles FastVLM performance characteristics."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize performance profiler."""
        self.metrics_collector = metrics_collector
        self.profiles = {}
        
    def start_profile(self, name: str):
        """Start a performance profile."""
        self.profiles[name] = {
            "start_time": time.time(),
            "checkpoints": []
        }
    
    def checkpoint(self, profile_name: str, checkpoint_name: str):
        """Add a checkpoint to a profile."""
        if profile_name in self.profiles:
            current_time = time.time()
            start_time = self.profiles[profile_name]["start_time"]
            
            self.profiles[profile_name]["checkpoints"].append({
                "name": checkpoint_name,
                "timestamp": current_time,
                "elapsed_ms": (current_time - start_time) * 1000
            })
    
    def end_profile(self, profile_name: str) -> Dict[str, Any]:
        """End a profile and record metrics."""
        if profile_name not in self.profiles:
            return {}
        
        profile = self.profiles[profile_name]
        end_time = time.time()
        total_duration = (end_time - profile["start_time"]) * 1000
        
        # Record total duration
        self.metrics_collector.record(
            f"profile.{profile_name}.total_duration",
            total_duration,
            unit="ms"
        )
        
        # Record checkpoint durations
        for i, checkpoint in enumerate(profile["checkpoints"]):
            self.metrics_collector.record(
                f"profile.{profile_name}.{checkpoint['name']}",
                checkpoint["elapsed_ms"],
                unit="ms"
            )
        
        result = {
            "name": profile_name,
            "total_duration_ms": total_duration,
            "checkpoints": profile["checkpoints"]
        }
        
        del self.profiles[profile_name]
        return result


class ProductionMonitor:
    """Comprehensive production monitoring system."""
    
    def __init__(self):
        """Initialize production monitor."""
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        self.profiler = PerformanceProfiler(self.metrics_collector)
        
        # Setup default alert rules
        self._setup_default_alerts()
        
        # Setup default notification
        self.alert_manager.add_notification_callback(self._default_alert_handler)
        
        logger.info("Production monitor initialized")
    
    def start(self):
        """Start monitoring."""
        self.alert_manager.start_monitoring()
        logger.info("Production monitoring started")
    
    def stop(self):
        """Stop monitoring."""
        self.alert_manager.stop_monitoring()
        logger.info("Production monitoring stopped")
    
    def record_inference(self, latency_ms: float, confidence: float, success: bool):
        """Record inference metrics."""
        self.metrics_collector.record("inference.latency", latency_ms, unit="ms")
        self.metrics_collector.record("inference.confidence", confidence)
        self.metrics_collector.increment("inference.count")
        
        if success:
            self.metrics_collector.increment("inference.success")
        else:
            self.metrics_collector.increment("inference.failure")
    
    def record_error(self, error_type: str, component: str):
        """Record error metrics."""
        self.metrics_collector.increment("errors.total")
        self.metrics_collector.increment(f"errors.{error_type}")
        self.metrics_collector.increment(f"errors.component.{component}")
    
    def record_memory_usage(self, memory_mb: float):
        """Record memory usage."""
        self.metrics_collector.gauge("system.memory_usage", memory_mb, unit="MB")
    
    def record_cache_hit_rate(self, hit_rate: float):
        """Record cache hit rate."""
        self.metrics_collector.gauge("cache.hit_rate", hit_rate, unit="percent")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            "inference_metrics": {
                "latency": self.metrics_collector.get_metric_summary("inference.latency"),
                "confidence": self.metrics_collector.get_metric_summary("inference.confidence"),
                "success_rate": self._calculate_success_rate(),
                "throughput": self._calculate_throughput()
            },
            "system_metrics": {
                "memory_usage": self.metrics_collector.get_metric_summary("system.memory_usage"),
                "cache_hit_rate": self.metrics_collector.get_latest_value("cache.hit_rate"),
                "error_rate": self._calculate_error_rate()
            },
            "active_alerts": self.alert_manager.get_active_alerts(),
            "recent_alerts": self.alert_manager.get_alert_history(10)
        }
    
    def _setup_default_alerts(self):
        """Setup default alert rules."""
        alerts = [
            AlertRule(
                name="High Latency",
                metric_name="inference.latency",
                threshold=500.0,
                comparison=">",
                severity="warning"
            ),
            AlertRule(
                name="Very High Latency",
                metric_name="inference.latency", 
                threshold=1000.0,
                comparison=">",
                severity="error"
            ),
            AlertRule(
                name="Low Confidence",
                metric_name="inference.confidence",
                threshold=0.3,
                comparison="<",
                severity="warning"
            ),
            AlertRule(
                name="High Memory Usage",
                metric_name="system.memory_usage",
                threshold=1500.0,
                comparison=">",
                severity="warning"
            ),
            AlertRule(
                name="High Error Rate",
                metric_name="errors.total",
                threshold=10.0,
                comparison=">",
                duration_seconds=60.0,
                severity="error"
            )
        ]
        
        for alert in alerts:
            self.alert_manager.add_rule(alert)
    
    def _default_alert_handler(self, alert: Dict[str, Any]):
        """Default alert notification handler."""
        status = alert["status"]
        severity = alert["severity"]
        message = alert["message"]
        
        log_level = {
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL
        }.get(severity, logging.WARNING)
        
        logger.log(log_level, f"ALERT {status.upper()}: {message}")
    
    def _calculate_success_rate(self) -> float:
        """Calculate inference success rate."""
        success_count = self.metrics_collector.get_latest_value("inference.success", 0)
        total_count = self.metrics_collector.get_latest_value("inference.count", 0)
        
        if total_count > 0:
            return (success_count / total_count) * 100.0
        return 100.0
    
    def _calculate_throughput(self) -> float:
        """Calculate inference throughput per second."""
        summary = self.metrics_collector.get_metric_summary("inference.count", 60.0)
        if summary and summary.get("count", 0) > 0:
            return summary["count"] / 60.0  # per second
        return 0.0
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate per minute."""
        summary = self.metrics_collector.get_metric_summary("errors.total", 60.0)
        if summary:
            return summary.get("count", 0)
        return 0.0


# Global production monitor instance
production_monitor = ProductionMonitor()


def monitor_inference(func: Callable):
    """Decorator to monitor inference function performance."""
    def wrapper(*args, **kwargs):
        with production_monitor.metrics_collector.timer("inference.duration"):
            try:
                result = func(*args, **kwargs)
                
                # Extract metrics from result if it's an InferenceResult
                if hasattr(result, 'latency_ms') and hasattr(result, 'confidence'):
                    production_monitor.record_inference(
                        result.latency_ms,
                        result.confidence,
                        True
                    )
                
                return result
            except Exception as e:
                production_monitor.record_error(
                    type(e).__name__,
                    func.__name__
                )
                production_monitor.record_inference(0, 0, False)
                raise
    
    return wrapper


if __name__ == "__main__":
    # Demo monitoring capabilities
    print("FastVLM Production Monitoring Demo")
    print("=" * 40)
    
    # Start monitoring
    production_monitor.start()
    print("✓ Monitoring started")
    
    # Simulate some metrics
    import random
    
    print("\n📊 Simulating inference metrics...")
    for i in range(10):
        latency = random.uniform(100, 300)
        confidence = random.uniform(0.6, 0.95)
        success = random.choice([True, True, True, False])  # 75% success rate
        
        production_monitor.record_inference(latency, confidence, success)
        time.sleep(0.1)
    
    # Record some system metrics
    production_monitor.record_memory_usage(850.5)
    production_monitor.record_cache_hit_rate(92.5)
    
    # Get dashboard data
    print("\n📋 Dashboard Data:")
    dashboard = production_monitor.get_dashboard_data()
    
    print(f"Inference latency (avg): {dashboard['inference_metrics']['latency'].get('mean', 0):.1f}ms")
    print(f"Success rate: {dashboard['inference_metrics']['success_rate']:.1f}%")
    print(f"Memory usage: {dashboard['system_metrics']['memory_usage'].get('mean', 0):.1f}MB")
    print(f"Active alerts: {len(dashboard['active_alerts'])}")
    
    production_monitor.stop()
    print("✓ Monitoring stopped")