"""
Advanced logging configuration for FastVLM On-Device Kit.

Provides structured logging, multiple handlers, and log analysis.
"""

import logging
import logging.handlers
import json
import time
import sys
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime, timezone
import threading


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def __init__(self, include_extra: bool = True):
        """Initialize structured formatter.
        
        Args:
            include_extra: Whether to include extra fields in log records
        """
        super().__init__()
        self.include_extra = include_extra
        
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
            
        # Add extra fields if enabled
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'lineno', 'funcName', 'created',
                    'msecs', 'relativeCreated', 'thread', 'threadName',
                    'processName', 'process', 'getMessage', 'exc_info',
                    'exc_text', 'stack_info'
                }:
                    log_entry[key] = value
                    
        return json.dumps(log_entry)


class PerformanceFilter(logging.Filter):
    """Filter to track performance metrics in logs."""
    
    def __init__(self):
        """Initialize performance filter."""
        super().__init__()
        self.request_times = {}
        
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter and enhance log records with performance data."""
        # Add performance context if available
        if hasattr(record, 'request_id'):
            request_id = record.request_id
            current_time = time.time()
            
            # Track request start time
            if not hasattr(record, 'duration_ms'):
                if request_id not in self.request_times:
                    self.request_times[request_id] = current_time
                    record.request_start = True
                else:
                    # Calculate duration
                    start_time = self.request_times[request_id]
                    record.duration_ms = (current_time - start_time) * 1000
                    record.request_end = True
                    
                    # Clean up
                    del self.request_times[request_id]
                    
        return True


class SecurityFilter(logging.Filter):
    """Filter to sanitize sensitive information from logs."""
    
    def __init__(self):
        """Initialize security filter."""
        super().__init__()
        self.sensitive_patterns = [
            r'password["\']?\s*[:=]\s*["\']?([^"\']+)',
            r'token["\']?\s*[:=]\s*["\']?([^"\']+)',
            r'key["\']?\s*[:=]\s*["\']?([^"\']+)',
            r'secret["\']?\s*[:=]\s*["\']?([^"\']+)',
            r'api[_-]?key["\']?\s*[:=]\s*["\']?([^"\']+)',
        ]
        
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter and sanitize log records."""
        import re
        
        # Sanitize message
        message = record.getMessage()
        for pattern in self.sensitive_patterns:
            message = re.sub(pattern, r'***REDACTED***', message, flags=re.IGNORECASE)
        
        # Update record message
        record.msg = message
        record.args = ()
        
        return True


class LogAnalyzer:
    """Analyzes log patterns and metrics."""
    
    def __init__(self, log_file: str):
        """Initialize log analyzer.
        
        Args:
            log_file: Path to log file to analyze
        """
        self.log_file = Path(log_file)
        
    def analyze_performance_logs(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze performance logs from recent time period.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Performance analysis results
        """
        cutoff_time = time.time() - (hours * 3600)
        
        performance_data = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_duration_ms": 0.0,
            "max_duration_ms": 0.0,
            "error_rate": 0.0,
            "errors_by_type": {},
            "hourly_stats": {}
        }
        
        try:
            if not self.log_file.exists():
                return performance_data
                
            durations = []
            errors = []
            hourly_counts = {}
            
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        
                        # Parse timestamp
                        timestamp = datetime.fromisoformat(log_entry['timestamp'].replace('Z', '+00:00')).timestamp()
                        
                        if timestamp < cutoff_time:
                            continue
                            
                        # Count requests by hour
                        hour_key = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:00')
                        hourly_counts[hour_key] = hourly_counts.get(hour_key, 0) + 1
                        
                        # Track performance metrics
                        if 'duration_ms' in log_entry:
                            performance_data["total_requests"] += 1
                            durations.append(log_entry['duration_ms'])
                            
                            if log_entry.get('level') == 'ERROR':
                                performance_data["failed_requests"] += 1
                                error_type = log_entry.get('exception', 'Unknown')
                                errors.append(error_type)
                            else:
                                performance_data["successful_requests"] += 1
                                
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
                        
            # Compute statistics
            if durations:
                performance_data["avg_duration_ms"] = sum(durations) / len(durations)
                performance_data["max_duration_ms"] = max(durations)
                
            if performance_data["total_requests"] > 0:
                performance_data["error_rate"] = performance_data["failed_requests"] / performance_data["total_requests"]
                
            # Count errors by type
            for error in errors:
                performance_data["errors_by_type"][error] = performance_data["errors_by_type"].get(error, 0) + 1
                
            performance_data["hourly_stats"] = hourly_counts
            
        except Exception as e:
            logging.error(f"Log analysis failed: {e}")
            
        return performance_data
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of errors from recent logs.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Error summary
        """
        cutoff_time = time.time() - (hours * 3600)
        
        error_summary = {
            "total_errors": 0,
            "unique_errors": 0,
            "error_frequency": {},
            "recent_errors": []
        }
        
        try:
            if not self.log_file.exists():
                return error_summary
                
            errors = []
            
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        
                        if log_entry.get('level') != 'ERROR':
                            continue
                            
                        timestamp = datetime.fromisoformat(log_entry['timestamp'].replace('Z', '+00:00')).timestamp()
                        
                        if timestamp < cutoff_time:
                            continue
                            
                        error_info = {
                            "timestamp": log_entry['timestamp'],
                            "message": log_entry['message'],
                            "module": log_entry.get('module', 'unknown'),
                            "function": log_entry.get('function', 'unknown'),
                            "exception": log_entry.get('exception', '')
                        }
                        
                        errors.append(error_info)
                        
                        # Count frequency
                        error_key = f"{error_info['module']}.{error_info['function']}"
                        error_summary["error_frequency"][error_key] = error_summary["error_frequency"].get(error_key, 0) + 1
                        
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
                        
            error_summary["total_errors"] = len(errors)
            error_summary["unique_errors"] = len(error_summary["error_frequency"])
            error_summary["recent_errors"] = errors[-10:]  # Last 10 errors
            
        except Exception as e:
            logging.error(f"Error summary generation failed: {e}")
            
        return error_summary


class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler for high-performance logging."""
    
    def __init__(self, target_handler: logging.Handler, queue_size: int = 1000):
        """Initialize async log handler.
        
        Args:
            target_handler: Underlying handler to write logs
            queue_size: Maximum queue size
        """
        super().__init__()
        self.target_handler = target_handler
        self.queue_size = queue_size
        self.log_queue = []
        self.queue_lock = threading.Lock()
        self.worker_thread = None
        self.shutdown_event = threading.Event()
        self._start_worker()
        
    def _start_worker(self):
        """Start worker thread for async logging."""
        self.worker_thread = threading.Thread(target=self._worker_loop)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
    def _worker_loop(self):
        """Worker loop for processing log records."""
        while not self.shutdown_event.is_set():
            try:
                # Get records from queue
                with self.queue_lock:
                    records_to_process = self.log_queue[:]
                    self.log_queue.clear()
                
                # Process records
                for record in records_to_process:
                    try:
                        self.target_handler.emit(record)
                    except Exception:
                        pass  # Avoid logging errors in logger
                
                # Sleep briefly
                time.sleep(0.01)
                
            except Exception:
                pass
                
    def emit(self, record: logging.LogRecord):
        """Emit log record asynchronously."""
        try:
            with self.queue_lock:
                if len(self.log_queue) < self.queue_size:
                    self.log_queue.append(record)
                # If queue is full, drop the record silently
        except Exception:
            pass
            
    def close(self):
        """Close handler and shutdown worker thread."""
        self.shutdown_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)
        self.target_handler.close()
        super().close()


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    structured: bool = True,
    async_logging: bool = False,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> Dict[str, Any]:
    """Setup comprehensive logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, only console logging)
        structured: Whether to use structured JSON logging
        async_logging: Whether to use asynchronous logging
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Logging configuration info
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Setup formatter
    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    
    # Add filters
    performance_filter = PerformanceFilter()
    security_filter = SecurityFilter()
    console_handler.addFilter(performance_filter)
    console_handler.addFilter(security_filter)
    
    # Wrap in async handler if requested
    if async_logging:
        console_handler = AsyncLogHandler(console_handler)
    
    root_logger.addHandler(console_handler)
    
    handlers_info = {"console": True}
    
    # Setup file handler if requested
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(performance_filter)
        file_handler.addFilter(security_filter)
        
        if async_logging:
            file_handler = AsyncLogHandler(file_handler)
            
        root_logger.addHandler(file_handler)
        handlers_info["file"] = str(log_path)
    
    # Log configuration
    config_info = {
        "level": level,
        "structured": structured,
        "async": async_logging,
        "handlers": handlers_info,
        "filters": ["performance", "security"]
    }
    
    logging.info(f"Logging configured: {config_info}")
    
    return config_info


def get_logger(name: str, **kwargs) -> logging.Logger:
    """Get logger with optional extra context.
    
    Args:
        name: Logger name
        **kwargs: Extra context to include in all log messages
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    if kwargs:
        # Create adapter to add extra context
        logger = logging.LoggerAdapter(logger, kwargs)
        
    return logger


def create_request_logger(request_id: str) -> logging.Logger:
    """Create logger with request ID context.
    
    Args:
        request_id: Unique request identifier
        
    Returns:
        Logger with request context
    """
    return get_logger("fastvlm.request", request_id=request_id)


# Example usage functions
def log_performance_metrics(logger: logging.Logger, metrics: Dict[str, Any]):
    """Log performance metrics in structured format."""
    logger.info(
        "Performance metrics recorded",
        extra={
            "metric_type": "performance",
            **metrics
        }
    )


def log_security_event(logger: logging.Logger, event_type: str, details: Dict[str, Any]):
    """Log security event."""
    logger.warning(
        f"Security event: {event_type}",
        extra={
            "event_type": "security",
            "security_event": event_type,
            **details
        }
    )


def log_model_conversion(logger: logging.Logger, model_path: str, output_path: str, metrics: Dict[str, Any]):
    """Log model conversion event."""
    logger.info(
        "Model conversion completed",
        extra={
            "event_type": "model_conversion",
            "input_model": model_path,
            "output_model": output_path,
            **metrics
        }
    )