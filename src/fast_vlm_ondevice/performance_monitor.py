"""
Real-time Performance Monitoring for FastVLM
Tracks latency, memory, throughput, and system health.
"""

import time
import threading
import logging
import json
import statistics
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import gc

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""
    timestamp: float
    metric_name: str
    value: float
    tags: Dict[str, str]
    unit: str = ""


@dataclass
class SystemResource:
    """System resource usage snapshot."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    threads_count: int
    gc_collections: int


class MetricsBuffer:
    """Thread-safe circular buffer for metrics."""
    
    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        
    def add(self, metric: PerformanceMetric):
        """Add metric to buffer."""
        with self.lock:
            self.buffer.append(metric)
    
    def get_recent(self, count: int = 100) -> List[PerformanceMetric]:
        """Get recent metrics."""
        with self.lock:
            return list(self.buffer)[-count:]
    
    def get_by_name(self, metric_name: str, count: int = 100) -> List[PerformanceMetric]:
        """Get recent metrics by name."""
        with self.lock:
            filtered = [m for m in self.buffer if m.metric_name == metric_name]
            return filtered[-count:]
    
    def clear(self):
        """Clear all metrics."""
        with self.lock:
            self.buffer.clear()


class PerformanceAnalyzer:
    """Analyze performance metrics and detect anomalies."""
    
    def __init__(self):
        self.baseline_metrics = {}
        self.thresholds = {
            "latency_ms": {"warning": 200, "critical": 500},
            "memory_mb": {"warning": 400, "critical": 800},
            "error_rate": {"warning": 0.05, "critical": 0.15}
        }
    
    def analyze_latency(self, latencies: List[float]) -> Dict[str, Any]:
        """Analyze latency distribution and trends."""
        if not latencies:
            return {"status": "no_data"}
        
        analysis = {
            "count": len(latencies),
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "p95": self._percentile(latencies, 95),
            "p99": self._percentile(latencies, 99),
            "min": min(latencies),
            "max": max(latencies),
            "std_dev": statistics.stdev(latencies) if len(latencies) > 1 else 0
        }
        
        # Determine status
        p95_latency = analysis["p95"]
        if p95_latency > self.thresholds["latency_ms"]["critical"]:
            analysis["status"] = "critical"
            analysis["message"] = f"P95 latency ({p95_latency:.1f}ms) exceeds critical threshold"
        elif p95_latency > self.thresholds["latency_ms"]["warning"]:
            analysis["status"] = "warning"
            analysis["message"] = f"P95 latency ({p95_latency:.1f}ms) exceeds warning threshold"
        else:
            analysis["status"] = "healthy"
            analysis["message"] = "Latency within acceptable bounds"
        
        return analysis
    
    def analyze_memory_usage(self, memory_readings: List[float]) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        if not memory_readings:
            return {"status": "no_data"}
        
        current_memory = memory_readings[-1]
        analysis = {
            "current_mb": current_memory,
            "mean_mb": statistics.mean(memory_readings),
            "peak_mb": max(memory_readings),
            "trend": self._calculate_trend(memory_readings[-10:])
        }
        
        # Determine status
        if current_memory > self.thresholds["memory_mb"]["critical"]:
            analysis["status"] = "critical"
            analysis["message"] = f"Memory usage ({current_memory:.1f}MB) is critically high"
        elif current_memory > self.thresholds["memory_mb"]["warning"]:
            analysis["status"] = "warning"
            analysis["message"] = f"Memory usage ({current_memory:.1f}MB) is elevated"
        else:
            analysis["status"] = "healthy"
            analysis["message"] = "Memory usage within acceptable bounds"
        
        return analysis
    
    def analyze_error_patterns(self, success_count: int, error_count: int) -> Dict[str, Any]:
        """Analyze error rate and patterns."""
        total_requests = success_count + error_count
        if total_requests == 0:
            return {"status": "no_data"}
        
        error_rate = error_count / total_requests
        analysis = {
            "error_rate": error_rate,
            "error_count": error_count,
            "success_count": success_count,
            "total_requests": total_requests
        }
        
        if error_rate > self.thresholds["error_rate"]["critical"]:
            analysis["status"] = "critical"
            analysis["message"] = f"Error rate ({error_rate:.1%}) is critically high"
        elif error_rate > self.thresholds["error_rate"]["warning"]:
            analysis["status"] = "warning"
            analysis["message"] = f"Error rate ({error_rate:.1%}) is elevated"
        else:
            analysis["status"] = "healthy"
            analysis["message"] = "Error rate within acceptable bounds"
        
        return analysis
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _calculate_trend(self, recent_values: List[float]) -> str:
        """Calculate trend direction from recent values."""
        if len(recent_values) < 3:
            return "insufficient_data"
        
        # Simple linear trend
        x = list(range(len(recent_values)))
        y = recent_values
        
        # Calculate correlation
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(y)
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
        denominator = (sum((x[i] - x_mean) ** 2 for i in range(len(x))) * 
                      sum((y[i] - y_mean) ** 2 for i in range(len(y)))) ** 0.5
        
        if denominator == 0:
            return "stable"
        
        correlation = numerator / denominator
        
        if correlation > 0.3:
            return "increasing"
        elif correlation < -0.3:
            return "decreasing"
        else:
            return "stable"


class PerformanceMonitor:
    """Real-time performance monitoring system."""
    
    def __init__(self, buffer_size: int = 10000):
        self.metrics_buffer = MetricsBuffer(buffer_size)
        self.resource_buffer = MetricsBuffer(buffer_size)
        self.analyzer = PerformanceAnalyzer()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        self.monitor_interval = 1.0  # seconds
        
        # Alert callbacks
        self.alert_callbacks: List[Callable] = []
        
        # System resource tracking
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        
        # Thread safety
        self.lock = threading.Lock()
        
    def start_monitoring(self, interval: float = 1.0):
        """Start background monitoring thread."""
        if not self.monitoring_active:
            self.monitor_interval = interval
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info(f"📊 Performance monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        if self.monitoring_active:
            self.monitoring_active = False
            if self.monitor_thread:
                self.monitor_thread.join()
            logger.info("⏹️ Performance monitoring stopped")
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None, unit: str = ""):
        """Record a performance metric."""
        metric = PerformanceMetric(
            timestamp=time.time(),
            metric_name=name,
            value=value,
            tags=tags or {},
            unit=unit
        )
        self.metrics_buffer.add(metric)
    
    def record_request_latency(self, latency_ms: float, success: bool = True, tags: Optional[Dict[str, str]] = None):
        """Record request latency and outcome."""
        with self.lock:
            self.request_count += 1
            if not success:
                self.error_count += 1
        
        self.record_metric("request_latency_ms", latency_ms, tags, "ms")
        self.record_metric("request_success", 1.0 if success else 0.0, tags)
    
    def record_memory_usage(self, memory_mb: float, tags: Optional[Dict[str, str]] = None):
        """Record memory usage."""
        self.record_metric("memory_usage_mb", memory_mb, tags, "MB")
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add callback for performance alerts."""
        self.alert_callbacks.append(callback)
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        current_time = time.time()
        uptime_seconds = current_time - self.start_time
        
        # Get recent metrics
        recent_latencies = [
            m.value for m in self.metrics_buffer.get_by_name("request_latency_ms", 100)
        ]
        recent_memory = [
            m.value for m in self.metrics_buffer.get_by_name("memory_usage_mb", 100)
        ]
        
        # Analyze metrics
        latency_analysis = self.analyzer.analyze_latency(recent_latencies)
        memory_analysis = self.analyzer.analyze_memory_usage(recent_memory)
        error_analysis = self.analyzer.analyze_error_patterns(
            self.request_count - self.error_count, self.error_count
        )
        
        return {
            "uptime_seconds": uptime_seconds,
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "requests_per_second": self.request_count / max(uptime_seconds, 1),
            "latency": latency_analysis,
            "memory": memory_analysis,
            "errors": error_analysis,
            "monitoring_active": self.monitoring_active,
            "timestamp": current_time
        }
    
    def get_detailed_report(self) -> Dict[str, Any]:
        """Generate detailed performance report."""
        stats = self.get_current_stats()
        
        # Add detailed breakdowns
        recent_metrics = self.metrics_buffer.get_recent(1000)
        metrics_by_name = defaultdict(list)
        
        for metric in recent_metrics:
            metrics_by_name[metric.metric_name].append(metric.value)
        
        detailed_metrics = {}
        for name, values in metrics_by_name.items():
            if values:
                detailed_metrics[name] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "recent": values[-10:]  # Last 10 values
                }
        
        return {
            "summary": stats,
            "detailed_metrics": detailed_metrics,
            "buffer_utilization": len(self.metrics_buffer.buffer) / self.metrics_buffer.buffer.maxlen,
            "report_generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                self._check_alerts()
                time.sleep(self.monitor_interval)
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    def _collect_system_metrics(self):
        """Collect system resource metrics."""
        try:
            # Get memory info (simple approximation)
            gc_stats = gc.get_stats()
            gc_collections = sum(stat.get('collections', 0) for stat in gc_stats)
            
            # Record system resource snapshot
            resource = SystemResource(
                timestamp=time.time(),
                cpu_percent=0.0,  # Would use psutil if available
                memory_mb=0.0,    # Would use psutil if available
                memory_percent=0.0,
                threads_count=threading.active_count(),
                gc_collections=gc_collections
            )
            
            self.record_metric("threads_count", resource.threads_count, unit="count")
            self.record_metric("gc_collections", resource.gc_collections, unit="count")
            
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
    
    def _check_alerts(self):
        """Check for alert conditions and trigger callbacks."""
        try:
            stats = self.get_current_stats()
            
            # Check for critical conditions
            if stats["latency"].get("status") == "critical":
                self._trigger_alert("latency_critical", stats["latency"])
            
            if stats["memory"].get("status") == "critical":
                self._trigger_alert("memory_critical", stats["memory"])
            
            if stats["errors"].get("status") == "critical":
                self._trigger_alert("error_rate_critical", stats["errors"])
            
        except Exception as e:
            logger.error(f"Alert checking failed: {e}")
    
    def _trigger_alert(self, alert_type: str, data: Dict[str, Any]):
        """Trigger alert callbacks."""
        logger.warning(f"🚨 Performance alert: {alert_type}")
        
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        if format == "json":
            return json.dumps(self.get_detailed_report(), indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def reset_metrics(self):
        """Reset all metrics and counters."""
        with self.lock:
            self.metrics_buffer.clear()
            self.resource_buffer.clear()
            self.request_count = 0
            self.error_count = 0
            self.start_time = time.time()
        
        logger.info("🔄 Performance metrics reset")


# Global performance monitor instance
_global_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def start_global_monitoring(interval: float = 1.0):
    """Start global performance monitoring."""
    monitor = get_performance_monitor()
    monitor.start_monitoring(interval)


def stop_global_monitoring():
    """Stop global performance monitoring."""
    global _global_monitor
    if _global_monitor:
        _global_monitor.stop_monitoring()


# Decorator for automatic performance tracking
def track_performance(metric_name: str = None, tags: Dict[str, str] = None):
    """Decorator to automatically track function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            start_time = time.time()
            success = True
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                latency_ms = (time.time() - start_time) * 1000
                name = metric_name or f"{func.__name__}_latency_ms"
                monitor.record_metric(name, latency_ms, tags, "ms")
                
                if "request" in name.lower():
                    monitor.record_request_latency(latency_ms, success, tags)
        
        return wrapper
    return decorator