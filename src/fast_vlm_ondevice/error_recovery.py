"""
Advanced error recovery and fault tolerance for FastVLM.

Implements circuit breakers, retry logic, graceful degradation,
and self-healing capabilities for production resilience.
"""

import time
import logging
import threading
import functools
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import random
import json
from pathlib import Path
import queue

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Available error recovery strategies."""
    RETRY = "retry"                    # Simple retry with backoff
    FALLBACK = "fallback"             # Switch to fallback method
    GRACEFUL_DEGRADATION = "degrade"  # Reduce functionality but continue
    CIRCUIT_BREAKER = "circuit"       # Temporarily disable failing component
    RESTART = "restart"               # Restart component/service
    FAILOVER = "failover"             # Switch to backup system


class HealthStatus(Enum):
    """Component health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class RecoveryConfig:
    """Configuration for error recovery behavior."""
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    exponential_backoff: bool = True
    jitter: bool = True
    
    # Circuit breaker settings
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 60.0
    success_threshold: int = 3
    
    # Health monitoring
    health_check_interval_seconds: float = 30.0
    enable_self_healing: bool = True
    
    # Fallback behavior
    enable_graceful_degradation: bool = True
    fallback_quality_threshold: float = 0.7


@dataclass
class ErrorContext:
    """Context information about an error."""
    error: Exception
    function_name: str
    attempt: int
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_strategy: Optional[RecoveryStrategy] = None


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, config: RecoveryConfig):
        """Initialize circuit breaker."""
        self.config = config
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "open":
                if time.time() - self.last_failure_time < self.config.recovery_timeout_seconds:
                    raise RuntimeError("Circuit breaker is OPEN")
                else:
                    self.state = "half-open"
                    logger.info(f"Circuit breaker {func.__name__} entering HALF-OPEN state")
        
        try:
            result = func(*args, **kwargs)
            
            with self._lock:
                if self.state == "half-open":
                    self.success_count += 1
                    if self.success_count >= self.config.success_threshold:
                        self.state = "closed"
                        self.failure_count = 0
                        self.success_count = 0
                        logger.info(f"Circuit breaker {func.__name__} CLOSED - recovered")
                elif self.state == "closed":
                    self.failure_count = max(0, self.failure_count - 1)
            
            return result
            
        except Exception as e:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.config.failure_threshold:
                    self.state = "open"
                    self.success_count = 0
                    logger.error(f"Circuit breaker {func.__name__} OPENED due to {self.failure_count} failures")
                elif self.state == "half-open":
                    self.state = "open"
                    logger.error(f"Circuit breaker {func.__name__} failed during recovery, back to OPEN")
            
            raise e
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        with self._lock:
            return {
                "state": self.state,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": self.last_failure_time
            }


class RetryManager:
    """Manages retry logic with exponential backoff."""
    
    def __init__(self, config: RecoveryConfig):
        """Initialize retry manager."""
        self.config = config
    
    def execute_with_retry(self, 
                          func: Callable, 
                          *args, 
                          retry_exceptions: Tuple[Exception, ...] = (Exception,),
                          **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if attempt > 0:
                    delay = self._calculate_delay(attempt)
                    logger.info(f"Retrying {func.__name__} (attempt {attempt + 1}) after {delay:.2f}s delay")
                    time.sleep(delay)
                
                return func(*args, **kwargs)
                
            except retry_exceptions as e:
                last_exception = e
                error_context = ErrorContext(
                    error=e,
                    function_name=func.__name__,
                    attempt=attempt + 1,
                    context={"args_count": len(args), "kwargs_keys": list(kwargs.keys())}
                )
                
                logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                
                if attempt == self.config.max_retries:
                    logger.error(f"All {self.config.max_retries + 1} attempts failed for {func.__name__}")
                    break
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        if self.config.exponential_backoff:
            delay = self.config.base_delay_seconds * (2 ** (attempt - 1))
        else:
            delay = self.config.base_delay_seconds
        
        delay = min(delay, self.config.max_delay_seconds)
        
        if self.config.jitter:
            # Add random jitter (±20%)
            jitter_range = delay * 0.2
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0.1, delay)  # Minimum 100ms delay


class FallbackManager:
    """Manages fallback strategies and graceful degradation."""
    
    def __init__(self, config: RecoveryConfig):
        """Initialize fallback manager."""
        self.config = config
        self.fallback_methods = {}
        self._lock = threading.Lock()
    
    def register_fallback(self, primary_method: str, fallback_func: Callable, quality_score: float = 1.0):
        """Register a fallback method."""
        with self._lock:
            if primary_method not in self.fallback_methods:
                self.fallback_methods[primary_method] = []
            
            self.fallback_methods[primary_method].append({
                "function": fallback_func,
                "quality_score": quality_score,
                "usage_count": 0,
                "success_rate": 1.0
            })
            
            # Sort by quality score descending
            self.fallback_methods[primary_method].sort(key=lambda x: x["quality_score"], reverse=True)
    
    def execute_with_fallback(self, 
                            primary_method: str, 
                            primary_func: Callable, 
                            *args, 
                            **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Execute with automatic fallback on failure."""
        execution_info = {
            "method_used": primary_method,
            "fallback_used": False,
            "quality_degradation": 0.0
        }
        
        try:
            result = primary_func(*args, **kwargs)
            return result, execution_info
            
        except Exception as primary_error:
            logger.warning(f"Primary method {primary_method} failed: {primary_error}")
            
            with self._lock:
                fallbacks = self.fallback_methods.get(primary_method, [])
            
            if not fallbacks:
                raise primary_error
            
            # Try fallbacks in order of quality
            for fallback_info in fallbacks:
                fallback_func = fallback_info["function"]
                quality_score = fallback_info["quality_score"]
                
                # Skip fallbacks below quality threshold
                if quality_score < self.config.fallback_quality_threshold:
                    continue
                
                try:
                    logger.info(f"Trying fallback method for {primary_method} (quality: {quality_score:.2f})")
                    
                    result = fallback_func(*args, **kwargs)
                    
                    # Update fallback statistics
                    with self._lock:
                        fallback_info["usage_count"] += 1
                        # Update success rate (simple moving average)
                        fallback_info["success_rate"] = (fallback_info["success_rate"] * 0.9) + (1.0 * 0.1)
                    
                    execution_info.update({
                        "method_used": f"{primary_method}_fallback",
                        "fallback_used": True,
                        "quality_degradation": 1.0 - quality_score,
                        "fallback_quality": quality_score
                    })
                    
                    return result, execution_info
                    
                except Exception as fallback_error:
                    logger.warning(f"Fallback method failed: {fallback_error}")
                    
                    # Update failure statistics
                    with self._lock:
                        fallback_info["usage_count"] += 1
                        fallback_info["success_rate"] = (fallback_info["success_rate"] * 0.9) + (0.0 * 0.1)
            
            # All methods failed
            raise RuntimeError(f"Primary method and all fallbacks failed for {primary_method}")
    
    def get_fallback_stats(self) -> Dict[str, Any]:
        """Get fallback usage statistics."""
        with self._lock:
            stats = {}
            for method, fallbacks in self.fallback_methods.items():
                stats[method] = [
                    {
                        "quality_score": fb["quality_score"],
                        "usage_count": fb["usage_count"],
                        "success_rate": fb["success_rate"]
                    }
                    for fb in fallbacks
                ]
            return stats


class HealthMonitor:
    """Monitors system health and triggers recovery actions."""
    
    def __init__(self, config: RecoveryConfig):
        """Initialize health monitor."""
        self.config = config
        self.component_health = {}
        self.health_history = []
        self.monitoring_active = False
        self.monitor_thread = None
        self._lock = threading.Lock()
        self.recovery_actions = {}
    
    def register_component(self, 
                         component_name: str, 
                         health_check_func: Callable[[], bool],
                         recovery_func: Optional[Callable] = None):
        """Register a component for health monitoring."""
        with self._lock:
            self.component_health[component_name] = {
                "status": HealthStatus.HEALTHY,
                "health_check": health_check_func,
                "recovery_action": recovery_func,
                "last_check": 0,
                "failure_count": 0,
                "recovery_attempts": 0
            }
    
    def start_monitoring(self):
        """Start health monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._check_all_components()
                time.sleep(self.config.health_check_interval_seconds)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(5.0)  # Brief pause on error
    
    def _check_all_components(self):
        """Check health of all registered components."""
        with self._lock:
            components = list(self.component_health.items())
        
        for component_name, component_info in components:
            try:
                is_healthy = component_info["health_check"]()
                current_time = time.time()
                
                with self._lock:
                    self.component_health[component_name]["last_check"] = current_time
                    
                    if is_healthy:
                        if component_info["status"] != HealthStatus.HEALTHY:
                            logger.info(f"Component {component_name} recovered to healthy state")
                            self.component_health[component_name]["status"] = HealthStatus.HEALTHY
                            self.component_health[component_name]["failure_count"] = 0
                    else:
                        self.component_health[component_name]["failure_count"] += 1
                        failure_count = self.component_health[component_name]["failure_count"]
                        
                        # Update status based on failure count
                        if failure_count == 1:
                            self.component_health[component_name]["status"] = HealthStatus.DEGRADED
                            logger.warning(f"Component {component_name} degraded")
                        elif failure_count >= 3:
                            self.component_health[component_name]["status"] = HealthStatus.FAILED
                            logger.error(f"Component {component_name} failed")
                            
                            # Attempt recovery if enabled
                            if self.config.enable_self_healing:
                                self._attempt_recovery(component_name)
                
            except Exception as e:
                logger.error(f"Health check failed for {component_name}: {e}")
                with self._lock:
                    self.component_health[component_name]["status"] = HealthStatus.FAILING
    
    def _attempt_recovery(self, component_name: str):
        """Attempt to recover a failed component."""
        with self._lock:
            component_info = self.component_health[component_name]
            recovery_func = component_info.get("recovery_action")
            
            if not recovery_func:
                logger.warning(f"No recovery action defined for {component_name}")
                return
            
            component_info["status"] = HealthStatus.RECOVERING
            component_info["recovery_attempts"] += 1
        
        try:
            logger.info(f"Attempting recovery for component {component_name}")
            recovery_func()
            
            # Wait a bit then check if recovery was successful
            time.sleep(2.0)
            is_healthy = component_info["health_check"]()
            
            with self._lock:
                if is_healthy:
                    self.component_health[component_name]["status"] = HealthStatus.HEALTHY
                    self.component_health[component_name]["failure_count"] = 0
                    logger.info(f"Successfully recovered component {component_name}")
                else:
                    self.component_health[component_name]["status"] = HealthStatus.FAILED
                    logger.error(f"Recovery failed for component {component_name}")
                    
        except Exception as e:
            logger.error(f"Recovery attempt failed for {component_name}: {e}")
            with self._lock:
                self.component_health[component_name]["status"] = HealthStatus.FAILED
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        with self._lock:
            health_summary = {
                "overall_status": "healthy",
                "components": {},
                "healthy_count": 0,
                "total_count": len(self.component_health)
            }
            
            for name, info in self.component_health.items():
                status = info["status"]
                health_summary["components"][name] = {
                    "status": status.value,
                    "last_check": info["last_check"],
                    "failure_count": info["failure_count"],
                    "recovery_attempts": info["recovery_attempts"]
                }
                
                if status == HealthStatus.HEALTHY:
                    health_summary["healthy_count"] += 1
                elif status in [HealthStatus.FAILED, HealthStatus.FAILING]:
                    health_summary["overall_status"] = "degraded"
            
            # Determine overall status
            health_ratio = health_summary["healthy_count"] / max(1, health_summary["total_count"])
            if health_ratio < 0.5:
                health_summary["overall_status"] = "critical"
            elif health_ratio < 0.8:
                health_summary["overall_status"] = "degraded"
            
            return health_summary


class ErrorRecoveryManager:
    """Main error recovery and resilience manager."""
    
    def __init__(self, config: RecoveryConfig = None):
        """Initialize error recovery manager."""
        self.config = config or RecoveryConfig()
        
        self.circuit_breakers = {}
        self.retry_manager = RetryManager(self.config)
        self.fallback_manager = FallbackManager(self.config)
        self.health_monitor = HealthMonitor(self.config)
        
        self._lock = threading.Lock()
        
        logger.info("Error recovery manager initialized")
    
    def resilient_execute(self, 
                         func: Callable, 
                         method_name: str,
                         *args,
                         recovery_strategies: List[RecoveryStrategy] = None,
                         **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Execute function with comprehensive error recovery."""
        
        if recovery_strategies is None:
            recovery_strategies = [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK]
        
        execution_info = {
            "strategies_used": [],
            "attempts": 0,
            "success": False,
            "fallback_used": False
        }
        
        last_error = None
        
        for strategy in recovery_strategies:
            try:
                execution_info["attempts"] += 1
                execution_info["strategies_used"].append(strategy.value)
                
                if strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                    result = self._execute_with_circuit_breaker(func, method_name, *args, **kwargs)
                    
                elif strategy == RecoveryStrategy.RETRY:
                    result = self.retry_manager.execute_with_retry(func, *args, **kwargs)
                    
                elif strategy == RecoveryStrategy.FALLBACK:
                    result, fallback_info = self.fallback_manager.execute_with_fallback(
                        method_name, func, *args, **kwargs
                    )
                    execution_info.update(fallback_info)
                    
                else:
                    # Direct execution
                    result = func(*args, **kwargs)
                
                execution_info["success"] = True
                return result, execution_info
                
            except Exception as e:
                last_error = e
                logger.warning(f"Recovery strategy {strategy.value} failed for {method_name}: {e}")
        
        # All strategies failed
        execution_info["final_error"] = str(last_error)
        raise last_error
    
    def _execute_with_circuit_breaker(self, func: Callable, method_name: str, *args, **kwargs) -> Any:
        """Execute with circuit breaker protection."""
        with self._lock:
            if method_name not in self.circuit_breakers:
                self.circuit_breakers[method_name] = CircuitBreaker(self.config)
            
            circuit_breaker = self.circuit_breakers[method_name]
        
        return circuit_breaker.call(func, *args, **kwargs)
    
    def register_fallback_method(self, 
                               primary_method: str, 
                               fallback_func: Callable, 
                               quality_score: float = 0.8):
        """Register a fallback method."""
        self.fallback_manager.register_fallback(primary_method, fallback_func, quality_score)
        logger.info(f"Registered fallback for {primary_method} with quality {quality_score}")
    
    def register_health_check(self, 
                            component_name: str, 
                            health_check_func: Callable[[], bool],
                            recovery_func: Optional[Callable] = None):
        """Register component for health monitoring."""
        self.health_monitor.register_component(component_name, health_check_func, recovery_func)
        logger.info(f"Registered health monitoring for {component_name}")
    
    def start_monitoring(self):
        """Start health monitoring."""
        self.health_monitor.start_monitoring()
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.health_monitor.stop_monitoring()
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get comprehensive recovery statistics."""
        with self._lock:
            circuit_breaker_stats = {
                name: cb.get_status() 
                for name, cb in self.circuit_breakers.items()
            }
        
        return {
            "circuit_breakers": circuit_breaker_stats,
            "fallback_methods": self.fallback_manager.get_fallback_stats(),
            "system_health": self.health_monitor.get_system_health(),
            "config": {
                "max_retries": self.config.max_retries,
                "failure_threshold": self.config.failure_threshold,
                "enable_self_healing": self.config.enable_self_healing
            }
        }


def resilient(method_name: str = None, 
             recovery_strategies: List[RecoveryStrategy] = None,
             recovery_manager: ErrorRecoveryManager = None) -> Callable:
    """Decorator to make functions resilient with error recovery."""
    
    def decorator(func: Callable) -> Callable:
        nonlocal method_name
        if method_name is None:
            method_name = func.__name__
        
        manager = recovery_manager or ErrorRecoveryManager()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result, execution_info = manager.resilient_execute(
                func, method_name, *args, 
                recovery_strategies=recovery_strategies, 
                **kwargs
            )
            return result
        
        # Store execution info on wrapper for debugging
        wrapper._last_execution_info = None
        
        return wrapper
    
    return decorator


# Example usage and factory functions
def create_error_recovery_manager(
    max_retries: int = 3,
    enable_circuit_breaker: bool = True,
    enable_self_healing: bool = True
) -> ErrorRecoveryManager:
    """Create error recovery manager with common configuration."""
    
    config = RecoveryConfig(
        max_retries=max_retries,
        failure_threshold=5 if enable_circuit_breaker else 999,
        enable_self_healing=enable_self_healing
    )
    
    return ErrorRecoveryManager(config)