# Observability and Monitoring Setup

This document outlines monitoring and observability configurations for the Fast VLM On-Device Kit.

## Overview

Observability includes metrics collection, logging, tracing, and alerting for both development and production environments.

## Metrics Collection

### Application Metrics

Create `src/fast_vlm_ondevice/metrics.py`:

```python
import time
import logging
from typing import Dict, Any
from functools import wraps

class PerformanceMetrics:
    """Collect performance metrics for model conversion and inference."""
    
    def __init__(self):
        self.metrics = {}
        self.logger = logging.getLogger(__name__)
    
    def record_conversion_time(self, model_size: str, duration: float):
        """Record model conversion time."""
        key = f"conversion_time_{model_size}"
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append(duration)
        self.logger.info(f"Conversion time for {model_size}: {duration:.2f}s")
    
    def record_model_size(self, model_name: str, size_mb: float):
        """Record converted model size."""
        self.metrics[f"model_size_{model_name}"] = size_mb
        self.logger.info(f"Model {model_name} size: {size_mb:.1f}MB")
    
    def record_inference_time(self, device: str, latency_ms: float):
        """Record inference latency."""
        key = f"inference_latency_{device}"
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append(latency_ms)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        summary = {}
        for key, values in self.metrics.items():
            if isinstance(values, list):
                summary[key] = {
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
            else:
                summary[key] = values
        return summary

def performance_monitor(metric_name: str):
    """Decorator to monitor function performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logging.info(f"{metric_name} completed in {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logging.error(f"{metric_name} failed after {duration:.2f}s: {e}")
                raise
        return wrapper
    return decorator
```

### Prometheus Integration

Create `monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'fastvlm-converter'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /metrics
    scrape_interval: 30s
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

## Logging Configuration

### Structured Logging

Create `src/fast_vlm_ondevice/logging_config.py`:

```python
import logging
import json
from datetime import datetime
from typing import Dict, Any

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'model_size'):
            log_entry['model_size'] = record.model_size
        if hasattr(record, 'device'):
            log_entry['device'] = record.device
        if hasattr(record, 'latency_ms'):
            log_entry['latency_ms'] = record.latency_ms
            
        return json.dumps(log_entry)

def setup_logging(level: str = "INFO", json_format: bool = True):
    """Setup logging configuration."""
    formatter = JSONFormatter() if json_format else logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    root_logger.addHandler(handler)
    
    # Configure specific loggers
    logging.getLogger('fast_vlm_ondevice').setLevel(logging.INFO)
    logging.getLogger('coremltools').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
```

## Error Tracking

### Sentry Integration

Add to `pyproject.toml` dependencies:
```toml
sentry-sdk = "^1.32.0"
```

Create `src/fast_vlm_ondevice/error_tracking.py`:

```python
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration
from typing import Optional

def setup_error_tracking(dsn: Optional[str] = None, environment: str = "development"):
    """Setup Sentry error tracking."""
    if not dsn:
        return
    
    sentry_logging = LoggingIntegration(
        level=logging.INFO,
        event_level=logging.ERROR
    )
    
    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        integrations=[sentry_logging],
        traces_sample_rate=0.1,
        release=get_version()
    )

def get_version() -> str:
    """Get package version for release tracking."""
    try:
        from importlib.metadata import version
        return version('fast-vlm-ondevice')
    except ImportError:
        return 'unknown'
```

## Health Checks

Create `src/fast_vlm_ondevice/health.py`:

```python
from typing import Dict, Any
import torch
import logging

class HealthChecker:
    """Application health checker."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check critical dependencies."""
        checks = {}
        
        # PyTorch check
        try:
            checks['pytorch'] = {
                'available': True,
                'version': torch.__version__,
                'cuda_available': torch.cuda.is_available()
            }
        except Exception as e:
            checks['pytorch'] = {'available': False, 'error': str(e)}
        
        # Core ML tools check
        try:
            import coremltools
            checks['coremltools'] = {
                'available': True,
                'version': coremltools.__version__
            }
        except Exception as e:
            checks['coremltools'] = {'available': False, 'error': str(e)}
        
        return checks
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource availability."""
        import psutil
        
        return {
            'memory': {
                'total_gb': psutil.virtual_memory().total / (1024**3),
                'available_gb': psutil.virtual_memory().available / (1024**3),
                'percent_used': psutil.virtual_memory().percent
            },
            'disk': {
                'total_gb': psutil.disk_usage('/').total / (1024**3),
                'free_gb': psutil.disk_usage('/').free / (1024**3),
                'percent_used': (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100
            },
            'cpu_percent': psutil.cpu_percent(interval=1)
        }
    
    def full_health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        return {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'dependencies': self.check_dependencies(),
            'system': self.check_system_resources()
        }
```

## Alert Rules

Create `monitoring/alert_rules.yml`:

```yaml
groups:
  - name: fastvlm_alerts
    rules:
      - alert: HighConversionTime
        expr: conversion_time_seconds > 300
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Model conversion taking too long"
          description: "Conversion time is {{ $value }} seconds"
      
      - alert: HighMemoryUsage
        expr: memory_usage_percent > 85
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is {{ $value }}%"
      
      - alert: ConversionFailures
        expr: rate(conversion_failures_total[5m]) > 0.1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "High conversion failure rate"
          description: "Conversion failure rate is {{ $value }} per second"
```

## Dashboard Configuration

### Grafana Dashboard JSON

Create `monitoring/dashboards/fastvlm-overview.json` with key metrics:
- Model conversion times
- Inference latencies
- Memory usage
- Error rates
- System health

## Implementation Checklist

- [ ] Install monitoring dependencies (`psutil`, `sentry-sdk`, `prometheus-client`)
- [ ] Configure structured logging in application
- [ ] Set up Prometheus metrics collection
- [ ] Deploy Grafana dashboards
- [ ] Configure Sentry error tracking
- [ ] Set up alerting rules and notification channels
- [ ] Test health check endpoints
- [ ] Document monitoring runbooks

## Usage Examples

```python
from fast_vlm_ondevice.metrics import PerformanceMetrics, performance_monitor
from fast_vlm_ondevice.logging_config import setup_logging

# Setup logging
setup_logging(level="INFO", json_format=True)

# Initialize metrics
metrics = PerformanceMetrics()

# Use performance monitoring
@performance_monitor("model_conversion")
def convert_model():
    # Conversion logic here
    pass

# Record custom metrics
metrics.record_conversion_time("base", duration)
metrics.record_inference_time("iphone15pro", latency_ms)
```