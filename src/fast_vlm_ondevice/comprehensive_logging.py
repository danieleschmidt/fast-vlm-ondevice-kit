"""
Comprehensive logging and observability framework for FastVLM.

Provides structured logging, performance metrics, distributed tracing,
and comprehensive monitoring for production deployment.
"""

import logging
import logging.handlers
import json
import time
import threading
import sys
import os
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import traceback
from pathlib import Path
import queue
import contextlib
from datetime import datetime
import socket

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False


class LogLevel(Enum):
    """Enhanced log levels."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class LogCategory(Enum):
    """Log categories for better organization."""
    SYSTEM = "system"
    SECURITY = "security"
    PERFORMANCE = "performance"
    USER_ACTION = "user_action"
    MODEL_INFERENCE = "model_inference"
    ERROR_RECOVERY = "error_recovery"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"


@dataclass
class LogContext:
    """Structured log context."""
    session_id: str
    request_id: str
    user_id: Optional[str] = None
    operation: Optional[str] = None
    component: Optional[str] = None
    environment: str = "production"
    version: str = "1.0.0"


@dataclass
class PerformanceMetric:
    """Performance metric data."""
    name: str
    value: float
    unit: str
    timestamp: float
    context: Dict[str, Any]
    tags: Dict[str, str]


class StructuredLogger:
    """Structured logger with enhanced context and formatting."""
    
    def __init__(self, name: str, context: LogContext = None):
        self.name = name
        self.context = context or LogContext(
            session_id=str(uuid.uuid4()),
            request_id=str(uuid.uuid4())
        )
        self.logger = logging.getLogger(name)
        self.performance_metrics = []
        self.thread_local = threading.local()
        
        # Configure structured logging if available
        if STRUCTLOG_AVAILABLE:
            self._configure_structlog()
    
    def _configure_structlog(self):
        """Configure structlog for better structured logging."""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.dev.ConsoleRenderer()
            ],
            wrapper_class=structlog.stdlib.LoggerAdapter,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True
        )
    
    def _format_message(self, level: LogLevel, message: str, category: LogCategory,
                       extra: Dict[str, Any] = None) -> Dict[str, Any]:
        """Format log message with structured context."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level.name,
            "message": message,
            "category": category.value,
            "logger": self.name,
            "context": asdict(self.context),
            "thread_id": threading.current_thread().ident,
            "hostname": socket.gethostname(),
            "process_id": os.getpid()
        }
        
        if extra:
            log_entry["extra"] = extra
        
        # Add any thread-local context
        if hasattr(self.thread_local, 'context'):
            log_entry["thread_context"] = self.thread_local.context
        
        return log_entry
    
    def log(self, level: LogLevel, message: str, category: LogCategory = LogCategory.SYSTEM,
            **kwargs):
        """Log message with structured format."""
        log_entry = self._format_message(level, message, category, kwargs)
        
        # Log to standard logger
        self.logger.log(level.value, json.dumps(log_entry, default=str))
        
        # Store performance metrics if applicable
        if category == LogCategory.PERFORMANCE and 'metric' in kwargs:
            self.performance_metrics.append(kwargs['metric'])
    
    def trace(self, message: str, **kwargs):
        """Log trace level message."""
        self.log(LogLevel.TRACE, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug level message."""
        self.log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info level message."""
        self.log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning level message."""
        self.log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, error: Exception = None, **kwargs):
        """Log error level message with optional exception."""
        extra = kwargs.copy()
        if error:
            extra.update({
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc()
            })
        self.log(LogLevel.ERROR, message, LogCategory.SYSTEM, **extra)
    
    def critical(self, message: str, **kwargs):
        """Log critical level message."""
        self.log(LogLevel.CRITICAL, message, **kwargs)
    
    def security(self, message: str, **kwargs):
        """Log security-related message."""
        self.log(LogLevel.WARNING, message, LogCategory.SECURITY, **kwargs)
    
    def performance(self, message: str, metric: PerformanceMetric, **kwargs):
        """Log performance metric."""
        kwargs["metric"] = asdict(metric)
        self.log(LogLevel.INFO, message, LogCategory.PERFORMANCE, **kwargs)
    
    def user_action(self, message: str, **kwargs):
        """Log user action."""
        self.log(LogLevel.INFO, message, LogCategory.USER_ACTION, **kwargs)
    
    def model_inference(self, message: str, **kwargs):
        """Log model inference activity."""
        self.log(LogLevel.INFO, message, LogCategory.MODEL_INFERENCE, **kwargs)
    
    def set_thread_context(self, **context):
        """Set thread-local context."""
        if not hasattr(self.thread_local, 'context'):
            self.thread_local.context = {}
        self.thread_local.context.update(context)
    
    def clear_thread_context(self):
        """Clear thread-local context."""
        if hasattr(self.thread_local, 'context'):
            self.thread_local.context.clear()
    
    @contextlib.contextmanager
    def context(self, **kwargs):
        """Context manager for temporary context."""
        old_context = getattr(self.thread_local, 'context', {}).copy()
        self.set_thread_context(**kwargs)
        try:
            yield
        finally:
            self.thread_local.context = old_context


class PerformanceTracker:
    """Performance tracking and metrics collection."""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.active_timers = {}
        self.metrics_history = []
        self.lock = threading.Lock()
    
    @contextlib.contextmanager
    def track_operation(self, operation_name: str, **tags):
        """Context manager for tracking operation performance."""
        start_time = time.time()
        timer_id = f"{operation_name}_{int(start_time * 1000)}"
        
        with self.lock:
            self.active_timers[timer_id] = {
                "operation": operation_name,
                "start_time": start_time,
                "tags": tags
            }
        
        try:
            yield
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            with self.lock:
                timer_info = self.active_timers.pop(timer_id, {})
            
            metric = PerformanceMetric(
                name=f"{operation_name}_duration",
                value=duration_ms,
                unit="milliseconds",
                timestamp=end_time,
                context={"operation": operation_name},
                tags=tags
            )
            
            self.metrics_history.append(metric)
            self.logger.performance(
                f"Operation {operation_name} completed",
                metric=metric,
                duration_ms=duration_ms
            )
    
    def record_metric(self, name: str, value: float, unit: str, **tags):
        """Record a custom metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=time.time(),
            context={},
            tags=tags
        )
        
        self.metrics_history.append(metric)
        self.logger.performance(f"Metric recorded: {name}", metric=metric)
    
    def get_metrics_summary(self, operation_name: str = None) -> Dict[str, Any]:
        """Get performance metrics summary."""
        filtered_metrics = self.metrics_history
        if operation_name:
            filtered_metrics = [m for m in filtered_metrics 
                              if operation_name in m.name]
        
        if not filtered_metrics:
            return {"message": "No metrics found"}
        
        values = [m.value for m in filtered_metrics]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "recent_metrics": [asdict(m) for m in filtered_metrics[-10:]]
        }


class SecurityAuditLogger:
    """Security-focused audit logging."""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.audit_events = []
        self.lock = threading.Lock()
    
    def log_authentication(self, user_id: str, success: bool, method: str, **kwargs):
        """Log authentication attempt."""
        self.logger.security(
            f"Authentication {'successful' if success else 'failed'}",
            user_id=user_id,
            success=success,
            method=method,
            **kwargs
        )
    
    def log_authorization(self, user_id: str, resource: str, action: str, 
                         granted: bool, **kwargs):
        """Log authorization decision."""
        self.logger.security(
            f"Authorization {'granted' if granted else 'denied'}",
            user_id=user_id,
            resource=resource,
            action=action,
            granted=granted,
            **kwargs
        )
    
    def log_data_access(self, user_id: str, data_type: str, operation: str, **kwargs):
        """Log data access."""
        self.logger.security(
            f"Data access: {operation} on {data_type}",
            user_id=user_id,
            data_type=data_type,
            operation=operation,
            **kwargs
        )
    
    def log_security_event(self, event_type: str, severity: str, description: str, **kwargs):
        """Log general security event."""
        event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "event_type": event_type,
            "severity": severity,
            "description": description,
            **kwargs
        }
        
        with self.lock:
            self.audit_events.append(event)
        
        self.logger.security(
            f"Security event: {event_type}",
            severity=severity,
            description=description,
            **kwargs
        )
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """Get security audit summary."""
        with self.lock:
            recent_events = self.audit_events[-50:]  # Last 50 events
        
        return {
            "total_events": len(self.audit_events),
            "recent_events": recent_events,
            "event_types": list(set(e.get("event_type") for e in recent_events))
        }


class DistributedTracingLogger:
    """Distributed tracing support for microservices."""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.active_spans = {}
        self.trace_tree = {}
        self.lock = threading.Lock()
    
    def start_span(self, operation_name: str, parent_span_id: str = None) -> str:
        """Start a new tracing span."""
        span_id = str(uuid.uuid4())
        trace_id = parent_span_id or str(uuid.uuid4())
        
        span = {
            "span_id": span_id,
            "trace_id": trace_id,
            "parent_span_id": parent_span_id,
            "operation_name": operation_name,
            "start_time": time.time(),
            "tags": {},
            "logs": []
        }
        
        with self.lock:
            self.active_spans[span_id] = span
        
        self.logger.trace(
            f"Span started: {operation_name}",
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id
        )
        
        return span_id
    
    def finish_span(self, span_id: str, status: str = "ok", **tags):
        """Finish a tracing span."""
        with self.lock:
            span = self.active_spans.pop(span_id, None)
        
        if not span:
            return
        
        span["end_time"] = time.time()
        span["duration_ms"] = (span["end_time"] - span["start_time"]) * 1000
        span["status"] = status
        span["tags"].update(tags)
        
        self.logger.trace(
            f"Span finished: {span['operation_name']}",
            span_id=span_id,
            duration_ms=span["duration_ms"],
            status=status
        )
    
    @contextlib.contextmanager
    def trace_operation(self, operation_name: str, parent_span_id: str = None):
        """Context manager for tracing operations."""
        span_id = self.start_span(operation_name, parent_span_id)
        try:
            yield span_id
            self.finish_span(span_id, "ok")
        except Exception as e:
            self.finish_span(span_id, "error", error=str(e))
            raise


class ComprehensiveLoggerFactory:
    """Factory for creating comprehensive loggers."""
    
    @staticmethod
    def setup_logging(
        log_level: LogLevel = LogLevel.INFO,
        log_file: Optional[str] = None,
        max_file_size_mb: int = 100,
        backup_count: int = 5,
        enable_console: bool = True,
        log_format: str = "json"
    ):
        """Set up comprehensive logging configuration."""
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level.value)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        formatters = {
            "json": logging.Formatter('%(message)s'),
            "standard": logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        }
        
        formatter = formatters.get(log_format, formatters["json"])
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # File handler with rotation
        if log_file:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size_mb * 1024 * 1024,
                backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        # Add TRACE level to logging
        logging.addLevelName(LogLevel.TRACE.value, "TRACE")
        
        logging.info("Comprehensive logging configured")
    
    @staticmethod
    def create_logger(
        name: str,
        context: LogContext = None,
        enable_performance_tracking: bool = True,
        enable_security_audit: bool = True,
        enable_distributed_tracing: bool = True
    ) -> StructuredLogger:
        """Create a comprehensive logger with all features."""
        
        logger = StructuredLogger(name, context)
        
        # Add performance tracking
        if enable_performance_tracking:
            logger.performance_tracker = PerformanceTracker(logger)
        
        # Add security audit logging
        if enable_security_audit:
            logger.security_audit = SecurityAuditLogger(logger)
        
        # Add distributed tracing
        if enable_distributed_tracing:
            logger.distributed_tracing = DistributedTracingLogger(logger)
        
        return logger


# Convenience functions
def setup_production_logging(log_file: str = "/var/log/fastvlm/app.log"):
    """Set up production-ready logging configuration."""
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    ComprehensiveLoggerFactory.setup_logging(
        log_level=LogLevel.INFO,
        log_file=log_file,
        max_file_size_mb=100,
        backup_count=10,
        enable_console=True,
        log_format="json"
    )


def get_logger(name: str, session_id: str = None) -> StructuredLogger:
    """Get a comprehensive logger instance."""
    context = LogContext(
        session_id=session_id or str(uuid.uuid4()),
        request_id=str(uuid.uuid4())
    )
    
    return ComprehensiveLoggerFactory.create_logger(name, context)


# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_production_logging("/tmp/fastvlm.log")
    
    # Create logger
    logger = get_logger("fastvlm.demo")
    
    # Demonstrate various logging features
    logger.info("FastVLM system starting up")
    
    # Performance tracking
    if hasattr(logger, 'performance_tracker'):
        with logger.performance_tracker.track_operation("model_loading"):
            time.sleep(0.1)  # Simulate model loading
        
        logger.performance_tracker.record_metric("memory_usage", 850.5, "MB")
    
    # Security audit
    if hasattr(logger, 'security_audit'):
        logger.security_audit.log_authentication("user123", True, "api_key")
        logger.security_audit.log_data_access("user123", "model_weights", "read")
    
    # Distributed tracing
    if hasattr(logger, 'distributed_tracing'):
        with logger.distributed_tracing.trace_operation("inference_request"):
            with logger.distributed_tracing.trace_operation("image_preprocessing"):
                time.sleep(0.05)
            
            with logger.distributed_tracing.trace_operation("model_inference"):
                time.sleep(0.15)
    
    # Context-aware logging
    with logger.context(user_id="user123", model_version="v1.0"):
        logger.model_inference("Model inference completed successfully")
    
    logger.info("Demo completed")