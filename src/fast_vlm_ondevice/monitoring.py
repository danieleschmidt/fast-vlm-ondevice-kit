"""
Advanced monitoring and observability for FastVLM models.

Provides metrics collection, logging, and telemetry capabilities.
"""

import logging
import time
import json
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque
from contextlib import contextmanager

try:
    import psutil
    import numpy as np
    MONITORING_DEPS = True
except ImportError:
    MONITORING_DEPS = False


@dataclass
class InferenceMetrics:
    """Container for inference metrics."""
    timestamp: float
    latency_ms: float
    memory_mb: float
    model_id: str
    input_size: Optional[tuple] = None
    output_size: Optional[tuple] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SystemMetrics:
    """Container for system metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    temperature_c: Optional[float] = None
    battery_percent: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsCollector:
    """Collects and aggregates performance metrics."""
    
    def __init__(self, max_history: int = 1000):
        """Initialize metrics collector.
        
        Args:
            max_history: Maximum number of metrics to keep in memory
        """
        self.max_history = max_history
        self.inference_metrics = deque(maxlen=max_history)
        self.system_metrics = deque(maxlen=max_history)
        self.aggregated_stats = defaultdict(list)
        self._lock = threading.Lock()
        
    def record_inference(self, metrics: InferenceMetrics) -> None:
        """Record inference metrics."""
        with self._lock:
            self.inference_metrics.append(metrics)
            
            # Update aggregated stats
            if metrics.error is None:
                self.aggregated_stats['latencies'].append(metrics.latency_ms)
                self.aggregated_stats['memory_usage'].append(metrics.memory_mb)
                
                # Keep only recent stats for aggregation
                if len(self.aggregated_stats['latencies']) > self.max_history:
                    self.aggregated_stats['latencies'] = self.aggregated_stats['latencies'][-self.max_history:]
                    self.aggregated_stats['memory_usage'] = self.aggregated_stats['memory_usage'][-self.max_history:]
    
    def record_system(self, metrics: SystemMetrics) -> None:
        """Record system metrics."""
        with self._lock:
            self.system_metrics.append(metrics)
    
    def get_inference_stats(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get aggregated inference statistics."""
        cutoff_time = time.time() - (window_minutes * 60)
        
        with self._lock:
            recent_metrics = [
                m for m in self.inference_metrics 
                if m.timestamp >= cutoff_time and m.error is None
            ]
        
        if not recent_metrics:
            return {
                "window_minutes": window_minutes,
                "total_inferences": 0,
                "error_rate": 0.0
            }
        
        latencies = [m.latency_ms for m in recent_metrics]
        memory_usage = [m.memory_mb for m in recent_metrics]
        errors = [m for m in recent_metrics if m.error is not None]
        
        stats = {
            "window_minutes": window_minutes,
            "total_inferences": len(recent_metrics),
            "successful_inferences": len(recent_metrics) - len(errors),
            "error_rate": len(errors) / len(recent_metrics) if recent_metrics else 0.0,
            "latency_stats": {
                "mean_ms": np.mean(latencies) if latencies else 0.0,
                "median_ms": np.median(latencies) if latencies else 0.0,
                "p95_ms": np.percentile(latencies, 95) if latencies else 0.0,
                "p99_ms": np.percentile(latencies, 99) if latencies else 0.0,
                "min_ms": np.min(latencies) if latencies else 0.0,
                "max_ms": np.max(latencies) if latencies else 0.0
            },
            "memory_stats": {
                "mean_mb": np.mean(memory_usage) if memory_usage else 0.0,
                "peak_mb": np.max(memory_usage) if memory_usage else 0.0,
                "min_mb": np.min(memory_usage) if memory_usage else 0.0
            },
            "throughput_fps": len(recent_metrics) / (window_minutes * 60) if recent_metrics else 0.0
        }
        
        return stats
    
    def get_system_stats(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get aggregated system statistics."""
        cutoff_time = time.time() - (window_minutes * 60)
        
        with self._lock:
            recent_metrics = [
                m for m in self.system_metrics 
                if m.timestamp >= cutoff_time
            ]
        
        if not recent_metrics:
            return {"window_minutes": window_minutes, "samples": 0}
        
        cpu_usage = [m.cpu_percent for m in recent_metrics]
        memory_usage = [m.memory_percent for m in recent_metrics]
        disk_usage = [m.disk_usage_percent for m in recent_metrics]
        
        return {
            "window_minutes": window_minutes,
            "samples": len(recent_metrics),
            "cpu_stats": {
                "mean_percent": np.mean(cpu_usage),
                "max_percent": np.max(cpu_usage),
                "min_percent": np.min(cpu_usage)
            },
            "memory_stats": {
                "mean_percent": np.mean(memory_usage),
                "max_percent": np.max(memory_usage),
                "min_percent": np.min(memory_usage)
            },
            "disk_stats": {
                "mean_percent": np.mean(disk_usage),
                "max_percent": np.max(disk_usage),
                "min_percent": np.min(disk_usage)
            }
        }
    
    def export_metrics(self, filepath: str, format: str = "json") -> None:
        """Export metrics to file."""
        with self._lock:
            data = {
                "inference_metrics": [m.to_dict() for m in self.inference_metrics],
                "system_metrics": [m.to_dict() for m in self.system_metrics],
                "export_timestamp": time.time()
            }
        
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")


class SystemMonitor:
    """Monitors system resources in background."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize system monitor.
        
        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics_collector = metrics_collector
        self.monitoring = False
        self.monitor_thread = None
        self.sample_interval = 1.0  # seconds
        
    def start_monitoring(self) -> None:
        """Start system monitoring."""
        if not MONITORING_DEPS:
            logging.warning("System monitoring dependencies not available")
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logging.info("System monitoring started")
        
    def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logging.info("System monitoring stopped")
        
    def _monitor_loop(self) -> None:
        """System monitoring loop."""
        while self.monitoring:
            try:
                metrics = self._collect_system_metrics()
                if metrics:
                    self.metrics_collector.record_system(metrics)
                time.sleep(self.sample_interval)
            except Exception as e:
                logging.error(f"System monitoring error: {e}")
                time.sleep(self.sample_interval)
                
    def _collect_system_metrics(self) -> Optional[SystemMetrics]:
        """Collect current system metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_mb = memory.available / (1024**2)
            
            # Disk usage
            disk = psutil.disk_usage('.')
            disk_usage_percent = (disk.used / disk.total) * 100
            
            # Optional: Temperature (if available)
            temperature_c = None
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    # Get first available temperature sensor
                    for sensor_name, sensor_list in temps.items():
                        if sensor_list:
                            temperature_c = sensor_list[0].current
                            break
            except (AttributeError, OSError):
                pass  # Temperature not available on this system
            
            # Optional: Battery (if available)
            battery_percent = None
            try:
                battery = psutil.sensors_battery()
                if battery:
                    battery_percent = battery.percent
            except (AttributeError, OSError):
                pass  # Battery not available on this system
            
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_usage_percent,
                temperature_c=temperature_c,
                battery_percent=battery_percent
            )
            
        except Exception as e:
            logging.error(f"Failed to collect system metrics: {e}")
            return None


class PerformanceProfiler:
    """Performance profiler with context manager support."""
    
    def __init__(self, metrics_collector: MetricsCollector, model_id: str = "default"):
        """Initialize performance profiler.
        
        Args:
            metrics_collector: Metrics collector instance
            model_id: Identifier for the model being profiled
        """
        self.metrics_collector = metrics_collector
        self.model_id = model_id
        
    @contextmanager
    def profile_inference(self, input_size: Optional[tuple] = None):
        """Context manager for profiling inference."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        error = None
        output_size = None
        
        try:
            yield self
        except Exception as e:
            error = str(e)
            raise
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            metrics = InferenceMetrics(
                timestamp=end_time,
                latency_ms=(end_time - start_time) * 1000,
                memory_mb=max(start_memory, end_memory),
                model_id=self.model_id,
                input_size=input_size,
                output_size=output_size,
                error=error
            )
            
            self.metrics_collector.record_inference(metrics)
    
    def set_output_size(self, size: tuple) -> None:
        """Set output size for current inference (call within profile_inference context)."""
        # This would be used within the context manager
        pass
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            if MONITORING_DEPS:
                process = psutil.Process()
                return process.memory_info().rss / (1024**2)
            return 0.0
        except:
            return 0.0


class AlertManager:
    """Manages performance alerts and thresholds."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize alert manager.
        
        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics_collector = metrics_collector
        self.thresholds = {
            "max_latency_ms": 1000.0,
            "max_error_rate": 0.05,  # 5%
            "max_memory_mb": 2048.0,
            "max_cpu_percent": 90.0,
            "min_throughput_fps": 1.0
        }
        self.alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
    def add_threshold(self, metric: str, value: float) -> None:
        """Add or update alert threshold."""
        self.thresholds[metric] = value
        
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add callback function for alerts."""
        self.alert_callbacks.append(callback)
        
    def check_alerts(self, window_minutes: int = 5) -> List[Dict[str, Any]]:
        """Check for alert conditions."""
        alerts = []
        
        # Get current stats
        inference_stats = self.metrics_collector.get_inference_stats(window_minutes)
        system_stats = self.metrics_collector.get_system_stats(window_minutes)
        
        # Check inference thresholds
        if inference_stats.get("total_inferences", 0) > 0:
            # Latency alert
            if inference_stats["latency_stats"]["p95_ms"] > self.thresholds["max_latency_ms"]:
                alert = {
                    "type": "latency",
                    "severity": "warning",
                    "message": f"P95 latency ({inference_stats['latency_stats']['p95_ms']:.1f}ms) exceeds threshold ({self.thresholds['max_latency_ms']}ms)",
                    "value": inference_stats["latency_stats"]["p95_ms"],
                    "threshold": self.thresholds["max_latency_ms"]
                }
                alerts.append(alert)
                
            # Error rate alert
            if inference_stats["error_rate"] > self.thresholds["max_error_rate"]:
                alert = {
                    "type": "error_rate",
                    "severity": "critical",
                    "message": f"Error rate ({inference_stats['error_rate']:.2%}) exceeds threshold ({self.thresholds['max_error_rate']:.2%})",
                    "value": inference_stats["error_rate"],
                    "threshold": self.thresholds["max_error_rate"]
                }
                alerts.append(alert)
                
            # Memory alert  
            if inference_stats["memory_stats"]["peak_mb"] > self.thresholds["max_memory_mb"]:
                alert = {
                    "type": "memory",
                    "severity": "warning",
                    "message": f"Peak memory ({inference_stats['memory_stats']['peak_mb']:.1f}MB) exceeds threshold ({self.thresholds['max_memory_mb']}MB)",
                    "value": inference_stats["memory_stats"]["peak_mb"],
                    "threshold": self.thresholds["max_memory_mb"]
                }
                alerts.append(alert)
                
            # Throughput alert
            if inference_stats["throughput_fps"] < self.thresholds["min_throughput_fps"]:
                alert = {
                    "type": "throughput",
                    "severity": "warning", 
                    "message": f"Throughput ({inference_stats['throughput_fps']:.1f} FPS) below threshold ({self.thresholds['min_throughput_fps']} FPS)",
                    "value": inference_stats["throughput_fps"],
                    "threshold": self.thresholds["min_throughput_fps"]
                }
                alerts.append(alert)
        
        # Check system thresholds
        if system_stats.get("samples", 0) > 0:
            # CPU alert
            if system_stats["cpu_stats"]["max_percent"] > self.thresholds["max_cpu_percent"]:
                alert = {
                    "type": "cpu",
                    "severity": "warning",
                    "message": f"CPU usage ({system_stats['cpu_stats']['max_percent']:.1f}%) exceeds threshold ({self.thresholds['max_cpu_percent']}%)",
                    "value": system_stats["cpu_stats"]["max_percent"],
                    "threshold": self.thresholds["max_cpu_percent"]
                }
                alerts.append(alert)
        
        # Trigger alert callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert["type"], alert)
                except Exception as e:
                    logging.error(f"Alert callback failed: {e}")
        
        return alerts


def setup_monitoring(model_id: str = "fastvlm") -> tuple[MetricsCollector, SystemMonitor, PerformanceProfiler, AlertManager]:
    """Setup complete monitoring stack.
    
    Args:
        model_id: Identifier for the model
        
    Returns:
        Tuple of (metrics_collector, system_monitor, profiler, alert_manager)
    """
    metrics_collector = MetricsCollector()
    system_monitor = SystemMonitor(metrics_collector)
    profiler = PerformanceProfiler(metrics_collector, model_id)
    alert_manager = AlertManager(metrics_collector)
    
    # Setup default alert thresholds for mobile deployment
    alert_manager.add_threshold("max_latency_ms", 500.0)  # 500ms max latency
    alert_manager.add_threshold("max_memory_mb", 1024.0)  # 1GB max memory
    alert_manager.add_threshold("max_error_rate", 0.01)   # 1% max error rate
    
    return metrics_collector, system_monitor, profiler, alert_manager