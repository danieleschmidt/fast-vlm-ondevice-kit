"""
Advanced error recovery and resilience framework for FastVLM.

Provides comprehensive error handling, circuit breakers, bulkheads,
and self-healing capabilities for production deployment.
"""

import logging
import time
import threading
import traceback
import asyncio
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
import queue
import functools
import weakref

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    BULKHEAD = "bulkhead"
    FAIL_FAST = "fail_fast"
    GRACEFUL_DEGRADATION = "graceful_degradation"


@dataclass
class ErrorEvent:
    """Error event data structure."""
    error_id: str
    timestamp: float
    error_type: str
    error_message: str
    severity: ErrorSeverity
    context: Dict[str, Any]
    recovery_strategy: Optional[RecoveryStrategy]
    resolved: bool = False
    resolution_time: Optional[float] = None


@dataclass
class RecoveryConfig:
    """Configuration for error recovery mechanisms."""
    max_retries: int = 3
    retry_delay_ms: int = 1000
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout_ms: int = 30000
    bulkhead_max_concurrent: int = 10
    enable_self_healing: bool = True
    enable_metrics_collection: bool = True


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, failure_threshold: int = 5, timeout_ms: int = 30000):
        self.failure_threshold = failure_threshold
        self.timeout_ms = timeout_ms
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.success_count = 0
        self.lock = threading.Lock()
        
        logger.info(f"Circuit breaker initialized - threshold: {failure_threshold}, timeout: {timeout_ms}ms")
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == "OPEN":
                if time.time() * 1000 - self.last_failure_time > self.timeout_ms:
                    self.state = "HALF_OPEN"
                    self.success_count = 0
                    logger.info("Circuit breaker moving to HALF_OPEN state")
                else:
                    raise CircuitBreakerOpenError("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            
            with self.lock:
                if self.state == "HALF_OPEN":
                    self.success_count += 1
                    if self.success_count >= 3:  # Require 3 successes to close
                        self.state = "CLOSED"
                        self.failure_count = 0
                        logger.info("Circuit breaker CLOSED after recovery")
                elif self.state == "CLOSED":
                    self.failure_count = 0  # Reset on success
            
            return result
            
        except Exception as e:
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time() * 1000
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")
                elif self.state == "HALF_OPEN":
                    self.state = "OPEN"
                    logger.warning("Circuit breaker reopened during half-open state")
            
            raise e


class Bulkhead:
    """Bulkhead pattern for resource isolation."""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.semaphore = threading.Semaphore(max_concurrent)
        self.active_operations = 0
        self.total_operations = 0
        self.rejected_operations = 0
        self.lock = threading.Lock()
        
        logger.info(f"Bulkhead initialized with max concurrent operations: {max_concurrent}")
    
    def execute(self, func: Callable, timeout_ms: Optional[int] = None, *args, **kwargs):
        """Execute function with bulkhead isolation."""
        timeout_s = timeout_ms / 1000 if timeout_ms else None
        
        if not self.semaphore.acquire(timeout=timeout_s):
            with self.lock:
                self.rejected_operations += 1
            raise BulkheadFullError("Bulkhead capacity exceeded")
        
        try:
            with self.lock:
                self.active_operations += 1
                self.total_operations += 1
            
            return func(*args, **kwargs)
            
        finally:
            with self.lock:
                self.active_operations -= 1
            self.semaphore.release()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics."""
        with self.lock:
            return {
                "max_concurrent": self.max_concurrent,
                "active_operations": self.active_operations,
                "total_operations": self.total_operations,
                "rejected_operations": self.rejected_operations,
                "success_rate": ((self.total_operations - self.rejected_operations) / 
                               max(self.total_operations, 1)) * 100
            }


class SelfHealingManager:
    """Self-healing system for automatic recovery."""
    
    def __init__(self, config: RecoveryConfig):
        self.config = config
        self.healing_strategies = {}
        self.error_patterns = {}
        self.healing_history = []
        self.monitoring_active = False
        self.monitor_thread = None
        
    def register_healing_strategy(self, error_pattern: str, healing_func: Callable):
        """Register a self-healing strategy for specific error patterns."""
        self.healing_strategies[error_pattern] = healing_func
        logger.info(f"Registered healing strategy for pattern: {error_pattern}")
    
    def attempt_healing(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Attempt automatic healing for the given error."""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Find matching healing strategy
        for pattern, healing_func in self.healing_strategies.items():
            if pattern in error_type or pattern in error_message:
                try:
                    logger.info(f"Attempting self-healing for error: {error_type}")
                    success = healing_func(error, context)
                    
                    self.healing_history.append({
                        "timestamp": time.time(),
                        "error_type": error_type,
                        "pattern": pattern,
                        "success": success,
                        "context": context
                    })
                    
                    if success:
                        logger.info(f"Self-healing successful for {error_type}")
                        return True
                    
                except Exception as healing_error:
                    logger.error(f"Self-healing failed: {healing_error}")
        
        return False
    
    def start_monitoring(self):
        """Start background monitoring for self-healing opportunities."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Self-healing monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Self-healing monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Analyze error patterns and trends
                self._analyze_error_patterns()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    def _analyze_error_patterns(self):
        """Analyze error patterns for proactive healing."""
        # Implement pattern analysis logic
        pass


class AdvancedErrorRecoveryManager:
    """Advanced error recovery manager with multiple strategies."""
    
    def __init__(self, config: RecoveryConfig = None):
        self.config = config or RecoveryConfig()
        self.error_events = []
        self.circuit_breakers = {}
        self.bulkheads = {}
        self.self_healing = SelfHealingManager(self.config)
        self.fallback_methods = {}
        self.retry_policies = {}
        self.error_metrics = {
            "total_errors": 0,
            "recovered_errors": 0,
            "failed_recoveries": 0,
            "recovery_strategies_used": {}
        }
        self.session_id = str(uuid.uuid4())
        
        # Start self-healing if enabled
        if self.config.enable_self_healing:
            self.self_healing.start_monitoring()
        
        logger.info(f"Advanced error recovery manager initialized - session: {self.session_id}")
    
    def register_circuit_breaker(self, operation_name: str, 
                                threshold: int = None, timeout_ms: int = None):
        """Register a circuit breaker for specific operation."""
        threshold = threshold or self.config.circuit_breaker_threshold
        timeout_ms = timeout_ms or self.config.circuit_breaker_timeout_ms
        
        self.circuit_breakers[operation_name] = CircuitBreaker(threshold, timeout_ms)
        logger.info(f"Circuit breaker registered for operation: {operation_name}")
    
    def register_bulkhead(self, resource_name: str, max_concurrent: int = None):
        """Register a bulkhead for resource isolation."""
        max_concurrent = max_concurrent or self.config.bulkhead_max_concurrent
        self.bulkheads[resource_name] = Bulkhead(max_concurrent)
        logger.info(f"Bulkhead registered for resource: {resource_name}")
    
    def register_fallback_method(self, operation_name: str, fallback_func: Callable, 
                                quality_score: float = 0.5):
        """Register a fallback method for operation."""
        self.fallback_methods[operation_name] = {
            "function": fallback_func,
            "quality_score": quality_score,
            "usage_count": 0
        }
        logger.info(f"Fallback method registered for operation: {operation_name}")
    
    def with_recovery(self, operation_name: str, strategies: List[RecoveryStrategy] = None):
        """Decorator for adding recovery strategies to functions."""
        strategies = strategies or [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK]
        
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self.execute_with_recovery(
                    operation_name, func, strategies, *args, **kwargs
                )
            return wrapper
        return decorator
    
    def execute_with_recovery(self, operation_name: str, func: Callable, 
                            strategies: List[RecoveryStrategy], *args, **kwargs):
        """Execute function with comprehensive recovery strategies."""
        error_context = {
            "operation_name": operation_name,
            "args_count": len(args),
            "kwargs_keys": list(kwargs.keys()),
            "timestamp": time.time()
        }
        
        for strategy in strategies:
            try:
                if strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                    return self._execute_with_circuit_breaker(
                        operation_name, func, *args, **kwargs
                    )
                    
                elif strategy == RecoveryStrategy.BULKHEAD:
                    return self._execute_with_bulkhead(
                        operation_name, func, *args, **kwargs
                    )
                    
                elif strategy == RecoveryStrategy.RETRY:
                    return self._execute_with_retry(
                        operation_name, func, *args, **kwargs
                    )
                    
                elif strategy == RecoveryStrategy.FALLBACK:
                    return self._execute_with_fallback(
                        operation_name, func, *args, **kwargs
                    )
                    
                elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                    return self._execute_with_graceful_degradation(
                        operation_name, func, *args, **kwargs
                    )
                    
                else:
                    # Default execution
                    return func(*args, **kwargs)
                    
            except Exception as e:
                error_event = self._create_error_event(e, error_context, strategy)
                self.error_events.append(error_event)
                
                # Try self-healing
                if self.config.enable_self_healing:
                    if self.self_healing.attempt_healing(e, error_context):
                        # Retry after successful healing
                        try:
                            result = func(*args, **kwargs)
                            error_event.resolved = True
                            error_event.resolution_time = time.time()
                            self.error_metrics["recovered_errors"] += 1
                            return result
                        except Exception:
                            pass  # Continue to next strategy
                
                # Continue to next strategy if current one failed
                logger.warning(f"Strategy {strategy.value} failed for {operation_name}: {e}")
                continue
        
        # All strategies failed
        self.error_metrics["failed_recoveries"] += 1
        raise AllRecoveryStrategiesFailedError(
            f"All recovery strategies failed for operation: {operation_name}"
        )
    
    def _execute_with_circuit_breaker(self, operation_name: str, func: Callable, 
                                    *args, **kwargs):
        """Execute with circuit breaker protection."""
        if operation_name not in self.circuit_breakers:
            self.register_circuit_breaker(operation_name)
        
        circuit_breaker = self.circuit_breakers[operation_name]
        return circuit_breaker.call(func, *args, **kwargs)
    
    def _execute_with_bulkhead(self, operation_name: str, func: Callable, 
                              *args, **kwargs):
        """Execute with bulkhead isolation."""
        resource_name = f"{operation_name}_resource"
        if resource_name not in self.bulkheads:
            self.register_bulkhead(resource_name)
        
        bulkhead = self.bulkheads[resource_name]
        return bulkhead.execute(func, 5000, *args, **kwargs)  # 5s timeout
    
    def _execute_with_retry(self, operation_name: str, func: Callable, 
                           *args, **kwargs):
        """Execute with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    delay_ms = self.config.retry_delay_ms * (2 ** attempt)  # Exponential backoff
                    time.sleep(delay_ms / 1000)
                    logger.info(f"Retrying {operation_name} (attempt {attempt + 2})")
        
        raise last_exception
    
    def _execute_with_fallback(self, operation_name: str, func: Callable, 
                              *args, **kwargs):
        """Execute with fallback method."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if operation_name in self.fallback_methods:
                fallback = self.fallback_methods[operation_name]
                fallback["usage_count"] += 1
                logger.info(f"Using fallback method for {operation_name}")
                return fallback["function"](*args, **kwargs)
            raise e
    
    def _execute_with_graceful_degradation(self, operation_name: str, func: Callable, 
                                         *args, **kwargs):
        """Execute with graceful degradation."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Return degraded but functional result
            logger.warning(f"Graceful degradation for {operation_name}: {e}")
            return {
                "result": f"Degraded mode response for {operation_name}",
                "quality": "degraded",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _create_error_event(self, error: Exception, context: Dict[str, Any], 
                          strategy: RecoveryStrategy) -> ErrorEvent:
        """Create error event for tracking."""
        self.error_metrics["total_errors"] += 1
        self.error_metrics["recovery_strategies_used"][strategy.value] = \
            self.error_metrics["recovery_strategies_used"].get(strategy.value, 0) + 1
        
        return ErrorEvent(
            error_id=str(uuid.uuid4()),
            timestamp=time.time(),
            error_type=type(error).__name__,
            error_message=str(error),
            severity=self._classify_error_severity(error),
            context=context,
            recovery_strategy=strategy
        )
    
    def _classify_error_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity based on error type."""
        error_type = type(error).__name__
        
        if error_type in ["MemoryError", "SystemExit", "KeyboardInterrupt"]:
            return ErrorSeverity.CRITICAL
        elif error_type in ["ConnectionError", "TimeoutError", "IOError"]:
            return ErrorSeverity.HIGH
        elif error_type in ["ValueError", "TypeError", "AttributeError"]:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def get_error_metrics(self) -> Dict[str, Any]:
        """Get comprehensive error metrics."""
        total_errors = self.error_metrics["total_errors"]
        recovery_rate = (self.error_metrics["recovered_errors"] / max(total_errors, 1)) * 100
        
        return {
            "session_id": self.session_id,
            "total_errors": total_errors,
            "recovered_errors": self.error_metrics["recovered_errors"],
            "failed_recoveries": self.error_metrics["failed_recoveries"],
            "recovery_rate_percent": recovery_rate,
            "strategies_used": self.error_metrics["recovery_strategies_used"],
            "circuit_breaker_stats": {name: cb.state for name, cb in self.circuit_breakers.items()},
            "bulkhead_stats": {name: bh.get_stats() for name, bh in self.bulkheads.items()},
            "recent_errors": [asdict(event) for event in self.error_events[-10:]]
        }
    
    def reset_metrics(self):
        """Reset error tracking metrics."""
        self.error_events.clear()
        self.error_metrics = {
            "total_errors": 0,
            "recovered_errors": 0,
            "failed_recoveries": 0,
            "recovery_strategies_used": {}
        }
        logger.info("Error metrics reset")


# Custom Exception Classes
class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class BulkheadFullError(Exception):
    """Raised when bulkhead capacity is exceeded."""
    pass


class AllRecoveryStrategiesFailedError(Exception):
    """Raised when all recovery strategies have failed."""
    pass


# Convenience Functions
def create_error_recovery_manager(
    max_retries: int = 3,
    circuit_breaker_threshold: int = 5,
    enable_self_healing: bool = True
) -> AdvancedErrorRecoveryManager:
    """Create error recovery manager with common configurations."""
    config = RecoveryConfig(
        max_retries=max_retries,
        circuit_breaker_threshold=circuit_breaker_threshold,
        enable_self_healing=enable_self_healing
    )
    return AdvancedErrorRecoveryManager(config)


def resilient(operation_name: str, strategies: List[RecoveryStrategy] = None):
    """Decorator for making functions resilient with error recovery."""
    recovery_manager = create_error_recovery_manager()
    return recovery_manager.with_recovery(operation_name, strategies)


# Example usage and demonstration
if __name__ == "__main__":
    # Demo error recovery
    recovery_manager = create_error_recovery_manager()
    
    # Register fallback method
    def fallback_inference():
        return {"result": "Fallback response", "confidence": 0.5}
    
    recovery_manager.register_fallback_method("inference", fallback_inference)
    
    # Demo function with potential failures
    @recovery_manager.with_recovery("inference", [
        RecoveryStrategy.RETRY,
        RecoveryStrategy.CIRCUIT_BREAKER,
        RecoveryStrategy.FALLBACK
    ])
    def risky_inference():
        import random
        if random.random() < 0.7:  # 70% failure rate
            raise ConnectionError("Simulated connection failure")
        return {"result": "Success", "confidence": 0.9}
    
    # Test recovery
    for i in range(5):
        try:
            result = risky_inference()
            print(f"Attempt {i+1}: {result}")
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
    
    # Print recovery metrics
    metrics = recovery_manager.get_error_metrics()
    print(f"Recovery rate: {metrics['recovery_rate_percent']:.1f}%")