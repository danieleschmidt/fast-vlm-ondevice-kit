"""
Reliability engine for FastVLM on-device deployment.

Implements comprehensive error handling, circuit breakers, health monitoring,
and self-healing capabilities for production mobile deployment.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from pathlib import Path
import weakref
from contextlib import contextmanager
import traceback

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    """Recovery actions for error handling."""
    RETRY = "retry"
    FALLBACK = "fallback"
    RESTART = "restart"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CIRCUIT_BREAK = "circuit_break"


@dataclass
class ErrorEvent:
    """Error event tracking."""
    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    component: str = ""
    error_type: str = ""
    message: str = ""
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    recovery_action: Optional[RecoveryAction] = None
    resolved: bool = False
    resolution_time: Optional[float] = None


@dataclass
class HealthMetrics:
    """Health metrics for system monitoring."""
    component: str
    status: HealthStatus = HealthStatus.HEALTHY
    last_check: float = field(default_factory=time.time)
    uptime_seconds: float = 0.0
    error_count: int = 0
    success_rate: float = 1.0
    average_latency_ms: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    timeout_seconds: float = 60.0
    success_threshold: int = 3  # For half-open state
    max_requests_half_open: int = 10


class CircuitBreaker:
    """Circuit breaker implementation for error handling."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.half_open_requests = 0
        self._lock = threading.Lock()
        
        logger.info(f"Circuit breaker '{name}' initialized")
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time < self.config.timeout_seconds:
                    raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is open")
                else:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_requests = 0
                    logger.info(f"Circuit breaker '{self.name}' transitioning to half-open")
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_requests >= self.config.max_requests_half_open:
                    raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' half-open limit reached")
                self.half_open_requests += 1
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful execution."""
        with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' closed (recovered)")
            else:
                self.failure_count = 0  # Reset failure count on success
    
    def _on_failure(self):
        """Handle failed execution."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' opened from half-open")
            elif self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' opened (threshold reached)")


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self):
        self.components = {}
        self.health_checks = {}
        self.monitoring_active = False
        self.check_interval = 30  # seconds
        self.history = []
        self.alerts = []
        self._lock = threading.Lock()
        
    def register_component(self, name: str, health_check: Callable = None):
        """Register a component for health monitoring."""
        with self._lock:
            self.components[name] = HealthMetrics(component=name)
            if health_check:
                self.health_checks[name] = health_check
        
        logger.info(f"Registered component '{name}' for health monitoring")
    
    def start_monitoring(self):
        """Start health monitoring."""
        if self.monitoring_active:
            return
            
        logger.info("Starting health monitoring")
        self.monitoring_active = True
        
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        logger.info("Stopping health monitoring")
        self.monitoring_active = False
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._perform_health_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                
    def _perform_health_checks(self):
        """Perform health checks for all components."""
        for component_name in list(self.components.keys()):
            try:
                self._check_component_health(component_name)
            except Exception as e:
                logger.error(f"Health check failed for {component_name}: {e}")
                self._update_component_health(component_name, HealthStatus.CRITICAL)
    
    def _check_component_health(self, component_name: str):
        """Check health of a specific component."""
        health_check = self.health_checks.get(component_name)
        
        if health_check:
            start_time = time.time()
            is_healthy = health_check()
            check_duration = (time.time() - start_time) * 1000
            
            status = HealthStatus.HEALTHY if is_healthy else HealthStatus.DEGRADED
            self._update_component_health(component_name, status, check_duration)
        else:
            # Default health check - component exists
            self._update_component_health(component_name, HealthStatus.HEALTHY)
    
    def _update_component_health(self, component_name: str, status: HealthStatus, 
                               latency_ms: float = 0.0):
        """Update component health status."""
        with self._lock:
            if component_name in self.components:
                metrics = self.components[component_name]
                metrics.status = status
                metrics.last_check = time.time()
                if latency_ms > 0:
                    metrics.average_latency_ms = latency_ms
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        with self._lock:
            component_statuses = [comp.status for comp in self.components.values()]
            
            if not component_statuses:
                overall_status = HealthStatus.OFFLINE
            elif any(status == HealthStatus.CRITICAL for status in component_statuses):
                overall_status = HealthStatus.CRITICAL
            elif any(status == HealthStatus.DEGRADED for status in component_statuses):
                overall_status = HealthStatus.DEGRADED
            else:
                overall_status = HealthStatus.HEALTHY
            
            return {
                "overall_status": overall_status.value,
                "components": {name: {
                    "status": metrics.status.value,
                    "last_check": metrics.last_check,
                    "error_count": metrics.error_count,
                    "success_rate": metrics.success_rate,
                    "average_latency_ms": metrics.average_latency_ms
                } for name, metrics in self.components.items()},
                "monitoring_active": self.monitoring_active,
                "check_count": len(self.history)
            }


class ErrorRecoveryManager:
    """Manages error recovery strategies and self-healing."""
    
    def __init__(self):
        self.error_history = []
        self.recovery_strategies = {}
        self.circuit_breakers = {}
        self.fallback_handlers = {}
        self.retry_policies = {}
        self._lock = threading.Lock()
        
    def register_recovery_strategy(self, error_type: str, strategy: RecoveryAction,
                                 handler: Callable = None, max_retries: int = 3):
        """Register a recovery strategy for a specific error type."""
        self.recovery_strategies[error_type] = {
            "action": strategy,
            "handler": handler,
            "max_retries": max_retries
        }
        logger.info(f"Registered recovery strategy for '{error_type}': {strategy.value}")
    
    def register_circuit_breaker(self, component: str, config: CircuitBreakerConfig = None):
        """Register a circuit breaker for a component."""
        self.circuit_breakers[component] = CircuitBreaker(component, config)
        logger.info(f"Registered circuit breaker for component '{component}'")
    
    def register_fallback_handler(self, component: str, handler: Callable):
        """Register a fallback handler for a component."""
        self.fallback_handlers[component] = handler
        logger.info(f"Registered fallback handler for component '{component}'")
    
    def handle_error(self, error: Exception, component: str, context: Dict[str, Any] = None) -> Any:
        """Handle an error with appropriate recovery strategy."""
        error_event = ErrorEvent(
            component=component,
            error_type=type(error).__name__,
            message=str(error),
            context=context or {},
            stack_trace=traceback.format_exc()
        )
        
        with self._lock:
            self.error_history.append(error_event)
        
        logger.error(f"Handling error in {component}: {error}")
        
        # Determine recovery action
        recovery_action = self._determine_recovery_action(error_event)
        error_event.recovery_action = recovery_action
        
        return self._execute_recovery_action(error_event, recovery_action)
    
    def _determine_recovery_action(self, error_event: ErrorEvent) -> RecoveryAction:
        """Determine appropriate recovery action for an error."""
        error_type = error_event.error_type
        
        # Check registered strategies
        if error_type in self.recovery_strategies:
            return self.recovery_strategies[error_type]["action"]
        
        # Default strategies based on error type
        if "TimeoutError" in error_type or "ConnectionError" in error_type:
            return RecoveryAction.RETRY
        elif "MemoryError" in error_type or "OutOfMemoryError" in error_type:
            return RecoveryAction.GRACEFUL_DEGRADATION
        elif "ValidationError" in error_type:
            return RecoveryAction.FALLBACK
        else:
            return RecoveryAction.RETRY
    
    def _execute_recovery_action(self, error_event: ErrorEvent, action: RecoveryAction) -> Any:
        """Execute the determined recovery action."""
        try:
            if action == RecoveryAction.RETRY:
                return self._execute_retry(error_event)
            elif action == RecoveryAction.FALLBACK:
                return self._execute_fallback(error_event)
            elif action == RecoveryAction.GRACEFUL_DEGRADATION:
                return self._execute_graceful_degradation(error_event)
            elif action == RecoveryAction.CIRCUIT_BREAK:
                return self._execute_circuit_break(error_event)
            else:
                logger.warning(f"Unknown recovery action: {action}")
                return None
                
        except Exception as recovery_error:
            logger.error(f"Recovery action failed: {recovery_error}")
            return self._execute_fallback(error_event)
    
    def _execute_retry(self, error_event: ErrorEvent) -> Any:
        """Execute retry recovery strategy."""
        logger.info(f"Executing retry for {error_event.component}")
        # Retry logic would be implemented here
        return {"status": "retried", "action": "retry"}
    
    def _execute_fallback(self, error_event: ErrorEvent) -> Any:
        """Execute fallback recovery strategy."""
        logger.info(f"Executing fallback for {error_event.component}")
        
        component = error_event.component
        if component in self.fallback_handlers:
            try:
                return self.fallback_handlers[component](error_event)
            except Exception as e:
                logger.error(f"Fallback handler failed: {e}")
        
        return {"status": "fallback", "message": "Using default fallback response"}
    
    def _execute_graceful_degradation(self, error_event: ErrorEvent) -> Any:
        """Execute graceful degradation strategy."""
        logger.info(f"Executing graceful degradation for {error_event.component}")
        return {"status": "degraded", "quality": "reduced"}
    
    def _execute_circuit_break(self, error_event: ErrorEvent) -> Any:
        """Execute circuit breaker strategy."""
        logger.info(f"Executing circuit break for {error_event.component}")
        component = error_event.component
        
        if component not in self.circuit_breakers:
            self.register_circuit_breaker(component)
        
        # Circuit breaker is handled in the calling code
        return {"status": "circuit_break", "component": component}


class ReliabilityEngine:
    """Main reliability engine coordinating all reliability features."""
    
    def __init__(self):
        self.health_monitor = HealthMonitor()
        self.error_recovery = ErrorRecoveryManager()
        self.is_initialized = False
        self.reliability_metrics = {
            "uptime_start": time.time(),
            "total_errors": 0,
            "recovered_errors": 0,
            "critical_failures": 0,
            "availability": 1.0
        }
        
    def initialize(self):
        """Initialize the reliability engine."""
        if self.is_initialized:
            return
            
        logger.info("Initializing FastVLM reliability engine")
        
        # Register core components
        self._register_core_components()
        
        # Register default recovery strategies
        self._register_default_recovery_strategies()
        
        # Start health monitoring
        self.health_monitor.start_monitoring()
        
        self.is_initialized = True
        logger.info("Reliability engine initialized successfully")
    
    def _register_core_components(self):
        """Register core system components for monitoring."""
        components = [
            "vision_encoder",
            "text_encoder", 
            "fusion_module",
            "decoder",
            "preprocessor",
            "postprocessor",
            "cache_manager",
            "model_loader"
        ]
        
        for component in components:
            self.health_monitor.register_component(
                component,
                lambda: self._default_health_check(component)
            )
    
    def _default_health_check(self, component: str) -> bool:
        """Default health check implementation."""
        # Simple health check - component should be responsive
        try:
            # Simulate health check
            time.sleep(0.001)  # 1ms check time
            return True
        except Exception:
            return False
    
    def _register_default_recovery_strategies(self):
        """Register default error recovery strategies."""
        strategies = [
            ("TimeoutError", RecoveryAction.RETRY),
            ("ConnectionError", RecoveryAction.RETRY),
            ("MemoryError", RecoveryAction.GRACEFUL_DEGRADATION),
            ("ValidationError", RecoveryAction.FALLBACK),
            ("ModelLoadError", RecoveryAction.FALLBACK),
            ("InferenceError", RecoveryAction.RETRY),
        ]
        
        for error_type, action in strategies:
            self.error_recovery.register_recovery_strategy(error_type, action)
    
    @contextmanager
    def reliability_context(self, component: str):
        """Context manager for reliable execution."""
        start_time = time.time()
        
        try:
            yield
            # Success
            duration_ms = (time.time() - start_time) * 1000
            self._record_success(component, duration_ms)
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._record_error(component, e, duration_ms)
            
            # Attempt recovery
            recovery_result = self.error_recovery.handle_error(e, component)
            
            if recovery_result.get("status") != "recovered":
                raise
    
    def _record_success(self, component: str, duration_ms: float):
        """Record successful execution."""
        if component in self.health_monitor.components:
            metrics = self.health_monitor.components[component]
            metrics.uptime_seconds += duration_ms / 1000
            # Update success rate
            total_ops = metrics.error_count + 1  # Assume 1 success
            metrics.success_rate = 1.0 / total_ops if total_ops > 0 else 1.0
    
    def _record_error(self, component: str, error: Exception, duration_ms: float):
        """Record error execution."""
        self.reliability_metrics["total_errors"] += 1
        
        if component in self.health_monitor.components:
            metrics = self.health_monitor.components[component]
            metrics.error_count += 1
            # Update success rate
            total_ops = metrics.error_count + 1
            metrics.success_rate = 1.0 / total_ops if total_ops > 0 else 0.0
    
    def get_reliability_report(self) -> Dict[str, Any]:
        """Get comprehensive reliability report."""
        uptime_seconds = time.time() - self.reliability_metrics["uptime_start"]
        
        return {
            "engine_status": "initialized" if self.is_initialized else "not_initialized",
            "uptime_seconds": uptime_seconds,
            "system_health": self.health_monitor.get_system_health(),
            "error_statistics": {
                "total_errors": self.reliability_metrics["total_errors"],
                "recovered_errors": self.reliability_metrics["recovered_errors"],
                "critical_failures": self.reliability_metrics["critical_failures"],
                "error_rate": self.reliability_metrics["total_errors"] / max(1, uptime_seconds / 3600),  # errors per hour
            },
            "availability": self._calculate_availability(),
            "circuit_breakers": {name: {
                "state": cb.state.value,
                "failure_count": cb.failure_count
            } for name, cb in self.error_recovery.circuit_breakers.items()},
            "recovery_strategies": list(self.error_recovery.recovery_strategies.keys())
        }
    
    def _calculate_availability(self) -> float:
        """Calculate system availability."""
        total_errors = self.reliability_metrics["total_errors"]
        critical_failures = self.reliability_metrics["critical_failures"]
        
        if total_errors == 0:
            return 1.0
        
        # Simple availability calculation
        non_critical_errors = total_errors - critical_failures
        availability = 1.0 - (critical_failures * 0.1 + non_critical_errors * 0.01)
        return max(0.0, availability)
    
    def shutdown(self):
        """Shutdown reliability engine."""
        logger.info("Shutting down reliability engine")
        self.health_monitor.stop_monitoring()
        self.is_initialized = False