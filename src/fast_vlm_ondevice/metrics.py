"""
Performance metrics collection and monitoring for FastVLM On-Device Kit.
Provides comprehensive metrics for model conversion, inference, and system health.
"""

import time
import logging
import json
from typing import Dict, Any, List, Optional
from functools import wraps
from dataclasses import dataclass, asdict
from datetime import datetime
import threading


@dataclass
class MetricEvent:
    """Represents a single metric event."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str]
    metadata: Optional[Dict[str, Any]] = None


class PerformanceMetrics:
    """Collect and manage performance metrics for FastVLM operations."""
    
    def __init__(self):
        self.metrics: Dict[str, List[MetricEvent]] = {}
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
    
    def record_event(self, name: str, value: float, tags: Optional[Dict[str, str]] = None, 
                     metadata: Optional[Dict[str, Any]] = None):
        """Record a metric event."""
        with self._lock:
            event = MetricEvent(
                name=name,
                value=value,
                timestamp=datetime.utcnow(),
                tags=tags or {},
                metadata=metadata
            )
            
            if name not in self.metrics:
                self.metrics[name] = []
            
            self.metrics[name].append(event)
            
            # Log the metric
            self.logger.info(
                f"Metric recorded: {name}={value}",
                extra={
                    'metric_name': name,
                    'metric_value': value,
                    'metric_tags': tags,
                    'metric_metadata': metadata
                }
            )
    
    def record_conversion_time(self, model_variant: str, quantization: str, 
                             duration: float, model_size_mb: Optional[float] = None):
        """Record model conversion performance."""
        tags = {
            'model_variant': model_variant,
            'quantization': quantization
        }
        metadata = {}
        if model_size_mb:
            metadata['model_size_mb'] = model_size_mb
        
        self.record_event('conversion_duration_seconds', duration, tags, metadata)
        
        if model_size_mb:
            self.record_event('converted_model_size_mb', model_size_mb, tags)
    
    def record_inference_performance(self, device: str, model_variant: str, 
                                   latency_ms: float, memory_mb: Optional[float] = None,
                                   accuracy: Optional[float] = None):
        """Record inference performance metrics."""
        tags = {
            'device': device,
            'model_variant': model_variant
        }
        
        self.record_event('inference_latency_ms', latency_ms, tags)
        
        if memory_mb:
            self.record_event('inference_memory_mb', memory_mb, tags)
        
        if accuracy:
            self.record_event('inference_accuracy', accuracy, tags)
    
    def record_system_metrics(self, cpu_percent: float, memory_percent: float, 
                            disk_percent: float):
        """Record system resource metrics."""
        timestamp = datetime.utcnow()
        
        self.record_event('system_cpu_percent', cpu_percent)
        self.record_event('system_memory_percent', memory_percent)
        self.record_event('system_disk_percent', disk_percent)
    
    def record_error(self, error_type: str, operation: str, error_message: str):
        """Record error occurrences."""
        tags = {
            'error_type': error_type,
            'operation': operation
        }
        metadata = {
            'error_message': error_message
        }
        
        self.record_event('error_count', 1, tags, metadata)
    
    def get_summary(self, metric_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistical summary of metrics."""
        with self._lock:
            if metric_name:
                return self._summarize_metric(metric_name)
            
            summary = {}
            for name in self.metrics:
                summary[name] = self._summarize_metric(name)
            
            return summary
    
    def _summarize_metric(self, metric_name: str) -> Dict[str, Any]:
        """Create statistical summary for a specific metric."""
        if metric_name not in self.metrics:
            return {'count': 0}
        
        events = self.metrics[metric_name]
        values = [event.value for event in events]
        
        if not values:
            return {'count': 0}
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'median': sorted(values)[len(values) // 2],
            'latest': events[-1].value,
            'latest_timestamp': events[-1].timestamp.isoformat()
        }
    
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        with self._lock:
            for metric_name, events in self.metrics.items():
                if not events:
                    continue
                
                # Group by tags for aggregation
                tag_groups: Dict[str, List[MetricEvent]] = {}
                
                for event in events:
                    tag_key = json.dumps(event.tags, sort_keys=True)
                    if tag_key not in tag_groups:
                        tag_groups[tag_key] = []
                    tag_groups[tag_key].append(event)
                
                for tag_key, group_events in tag_groups.items():
                    latest_event = group_events[-1]
                    tags_str = ""
                    
                    if latest_event.tags:
                        tag_pairs = [f'{k}="{v}"' for k, v in latest_event.tags.items()]
                        tags_str = "{" + ",".join(tag_pairs) + "}"
                    
                    lines.append(f"{metric_name}{tags_str} {latest_event.value}")
        
        return "\n".join(lines)
    
    def export_json(self) -> str:
        """Export all metrics as JSON."""
        export_data = {}
        
        with self._lock:
            for metric_name, events in self.metrics.items():
                export_data[metric_name] = [
                    {
                        'value': event.value,
                        'timestamp': event.timestamp.isoformat(),
                        'tags': event.tags,
                        'metadata': event.metadata
                    }
                    for event in events
                ]
        
        return json.dumps(export_data, indent=2)
    
    def clear_metrics(self, older_than: Optional[datetime] = None):
        """Clear metrics, optionally only those older than specified time."""
        with self._lock:
            if older_than is None:
                self.metrics.clear()
                self.logger.info("All metrics cleared")
            else:
                cleared_count = 0
                for metric_name in list(self.metrics.keys()):
                    events = self.metrics[metric_name]
                    filtered_events = [e for e in events if e.timestamp > older_than]
                    cleared_count += len(events) - len(filtered_events)
                    
                    if filtered_events:
                        self.metrics[metric_name] = filtered_events
                    else:
                        del self.metrics[metric_name]
                
                self.logger.info(f"Cleared {cleared_count} old metric events")


class MetricsExporter:
    """Export metrics to various monitoring systems."""
    
    def __init__(self, metrics: PerformanceMetrics):
        self.metrics = metrics
        self.logger = logging.getLogger(__name__)
    
    def export_to_prometheus_gateway(self, gateway_url: str, job_name: str = "fastvm"):
        """Push metrics to Prometheus Push Gateway."""
        try:
            import requests
            
            metrics_data = self.metrics.export_prometheus_format()
            
            response = requests.post(
                f"{gateway_url}/metrics/job/{job_name}",
                data=metrics_data,
                headers={'Content-Type': 'text/plain'}
            )
            
            if response.status_code == 200:
                self.logger.info("Metrics successfully pushed to Prometheus")
            else:
                self.logger.error(f"Failed to push metrics: {response.status_code}")
                
        except ImportError:
            self.logger.warning("requests not available - cannot push to Prometheus")
        except Exception as e:
            self.logger.error(f"Error pushing metrics to Prometheus: {e}")
    
    def export_to_file(self, filepath: str, format: str = "json"):
        """Export metrics to file."""
        try:
            if format == "json":
                data = self.metrics.export_json()
            elif format == "prometheus":
                data = self.metrics.export_prometheus_format()
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            with open(filepath, 'w') as f:
                f.write(data)
            
            self.logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting metrics to file: {e}")


def performance_monitor(operation_name: str, include_memory: bool = False):
    """Decorator to automatically monitor function performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get metrics instance from kwargs or create new one
            metrics = kwargs.get('_metrics') or PerformanceMetrics()
            
            start_time = time.time()
            start_memory = None
            
            if include_memory:
                try:
                    import psutil
                    process = psutil.Process()
                    start_memory = process.memory_info().rss / 1024 / 1024  # MB
                except ImportError:
                    pass
            
            try:
                result = func(*args, **kwargs)
                
                # Record success metrics
                duration = time.time() - start_time
                metrics.record_event(
                    f"{operation_name}_duration_seconds",
                    duration,
                    tags={'status': 'success', 'function': func.__name__}
                )
                
                if include_memory and start_memory:
                    try:
                        import psutil
                        process = psutil.Process()
                        end_memory = process.memory_info().rss / 1024 / 1024
                        memory_delta = end_memory - start_memory
                        
                        metrics.record_event(
                            f"{operation_name}_memory_delta_mb",
                            memory_delta,
                            tags={'function': func.__name__}
                        )
                    except ImportError:
                        pass
                
                return result
                
            except Exception as e:
                # Record failure metrics
                duration = time.time() - start_time
                metrics.record_event(
                    f"{operation_name}_duration_seconds",
                    duration,
                    tags={'status': 'error', 'function': func.__name__}
                )
                
                metrics.record_error(
                    error_type=type(e).__name__,
                    operation=operation_name,
                    error_message=str(e)
                )
                
                raise
        
        return wrapper
    return decorator


# Global metrics instance for convenience
_global_metrics = PerformanceMetrics()

def get_global_metrics() -> PerformanceMetrics:
    """Get the global metrics instance."""
    return _global_metrics

def record_metric(name: str, value: float, tags: Optional[Dict[str, str]] = None):
    """Convenience function to record a metric using global instance."""
    _global_metrics.record_event(name, value, tags)