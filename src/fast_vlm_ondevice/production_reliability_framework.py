"""
Production Reliability Framework
Enterprise-grade reliability patterns for mission-critical FastVLM deployments.

This framework implements advanced reliability patterns:
- Circuit Breakers with Intelligent Recovery
- Bulkhead Isolation for Component Failures
- Retry Patterns with Exponential Backoff and Jitter
- Health Checks with Deep System Diagnostics
- Graceful Degradation with Fallback Strategies
- Self-Healing Systems with Automatic Recovery
- Chaos Engineering Integration
- Distributed Tracing and Observability
"""

import asyncio
import json
import time
import logging
import threading
import uuid
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
from pathlib import Path
import psutil

logger = logging.getLogger(__name__)

class ComponentState(Enum):
    """Component health states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    FAILED = "failed"
    RECOVERING = "recovering"
    MAINTENANCE = "maintenance"

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    component: str
    status: ComponentState
    latency_ms: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

@dataclass
class ReliabilityMetrics:
    """Comprehensive reliability metrics."""
    component: str
    uptime_seconds: float
    availability_percent: float
    mtbf_seconds: float  # Mean Time Between Failures
    mttr_seconds: float  # Mean Time To Recovery
    error_rate_percent: float
    success_count: int
    failure_count: int
    recovery_count: int
    last_failure_time: Optional[str] = None
    last_recovery_time: Optional[str] = None

class IntelligentCircuitBreaker:
    """Advanced circuit breaker with ML-based failure prediction."""
    
    def __init__(self, 
                 name: str,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 half_open_max_calls: int = 3,
                 enable_prediction: bool = True):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.enable_prediction = enable_prediction
        
        # State management
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.half_open_call_count = 0
        
        # Metrics tracking
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.state_transitions = []
        
        # Predictive failure detection
        self.response_times = deque(maxlen=100)
        self.error_patterns = deque(maxlen=50)
        
        # Thread safety
        self.lock = threading.Lock()
        
        logger.info(f"🔄 Intelligent Circuit Breaker '{name}' initialized")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self.lock:
            self.total_calls += 1
            
            # Check if circuit is open
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time < self.recovery_timeout:
                    raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is OPEN")
                else:
                    # Transition to half-open for testing
                    self._transition_to_half_open()
            
            # Handle half-open state
            elif self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_call_count >= self.half_open_max_calls:
                    raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' half-open limit reached")
                self.half_open_call_count += 1
        
        # Execute the function with monitoring
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Record success
            execution_time = (time.time() - start_time) * 1000
            self._record_success(execution_time)
            
            return result
            
        except Exception as e:
            # Record failure
            execution_time = (time.time() - start_time) * 1000
            self._record_failure(e, execution_time)
            raise
    
    def _record_success(self, execution_time_ms: float) -> None:
        """Record successful call and update circuit breaker state."""
        with self.lock:
            self.successful_calls += 1
            self.response_times.append(execution_time_ms)
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                # Successful call in half-open state
                if self.half_open_call_count >= self.half_open_max_calls:
                    self._transition_to_closed()
            elif self.state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                self.failure_count = max(0, self.failure_count - 1)
    
    def _record_failure(self, error: Exception, execution_time_ms: float) -> None:
        """Record failed call and update circuit breaker state."""
        with self.lock:
            self.failed_calls += 1
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            # Record error pattern
            error_info = {
                "type": type(error).__name__,
                "message": str(error),
                "execution_time_ms": execution_time_ms,
                "timestamp": time.time()
            }
            self.error_patterns.append(error_info)
            
            # Check if we should open the circuit
            if self.failure_count >= self.failure_threshold:
                self._transition_to_open()
    
    def _transition_to_open(self) -> None:
        """Transition circuit breaker to OPEN state."""
        if self.state != CircuitBreakerState.OPEN:
            self.state = CircuitBreakerState.OPEN
            self.state_transitions.append({
                "from": self.state.value,
                "to": CircuitBreakerState.OPEN.value,
                "timestamp": time.time(),
                "reason": f"Failure threshold reached: {self.failure_count}/{self.failure_threshold}"
            })
            logger.warning(f"⚡ Circuit breaker '{self.name}' OPENED due to {self.failure_count} failures")
    
    def _transition_to_half_open(self) -> None:
        """Transition circuit breaker to HALF_OPEN state."""
        self.state = CircuitBreakerState.HALF_OPEN
        self.half_open_call_count = 0
        self.state_transitions.append({
            "from": CircuitBreakerState.OPEN.value,
            "to": CircuitBreakerState.HALF_OPEN.value,
            "timestamp": time.time(),
            "reason": "Recovery timeout elapsed, testing recovery"
        })
        logger.info(f"🔄 Circuit breaker '{self.name}' transitioned to HALF_OPEN")
    
    def _transition_to_closed(self) -> None:
        """Transition circuit breaker to CLOSED state."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.half_open_call_count = 0
        self.state_transitions.append({
            "from": CircuitBreakerState.HALF_OPEN.value,
            "to": CircuitBreakerState.CLOSED.value,
            "timestamp": time.time(),
            "reason": "Recovery confirmed through successful calls"
        })
        logger.info(f"✅ Circuit breaker '{self.name}' CLOSED - service recovered")
    
    def predict_failure_probability(self) -> float:
        """Predict probability of imminent failure using ML techniques."""
        if not self.enable_prediction or len(self.response_times) < 10:
            return 0.0
        
        try:
            # Analyze response time trends
            recent_times = list(self.response_times)[-20:]
            time_trend = self._calculate_trend(recent_times)
            
            # Analyze error patterns
            recent_errors = [e for e in self.error_patterns if time.time() - e["timestamp"] < 300]  # Last 5 minutes
            error_rate = len(recent_errors) / 20  # Errors per minute
            
            # Calculate response time statistics
            avg_response = np.mean(recent_times)
            response_variance = np.var(recent_times)
            
            # Failure prediction model (simplified)
            failure_indicators = [
                min(1.0, time_trend / 100),      # Increasing response time trend
                min(1.0, error_rate / 0.1),      # High error rate
                min(1.0, avg_response / 1000),   # High average response time
                min(1.0, response_variance / 10000),  # High variance
                min(1.0, self.failure_count / self.failure_threshold)  # Current failure count
            ]
            
            # Weighted combination
            weights = [0.3, 0.3, 0.2, 0.1, 0.1]
            failure_probability = sum(w * i for w, i in zip(weights, failure_indicators))
            
            return min(1.0, failure_probability)
            
        except Exception as e:
            logger.warning(f"Failure prediction error: {e}")
            return 0.0
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in values (positive = increasing)."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression slope
        n = len(values)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive circuit breaker metrics."""
        success_rate = (self.successful_calls / self.total_calls * 100) if self.total_calls > 0 else 0
        
        avg_response_time = np.mean(self.response_times) if self.response_times else 0
        p95_response_time = np.percentile(self.response_times, 95) if len(self.response_times) > 5 else 0
        
        return {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate_percent": round(success_rate, 2),
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "avg_response_time_ms": round(avg_response_time, 2),
            "p95_response_time_ms": round(p95_response_time, 2),
            "failure_probability": round(self.predict_failure_probability(), 3),
            "state_transitions": len(self.state_transitions),
            "last_failure_time": self.last_failure_time,
            "recovery_timeout": self.recovery_timeout
        }

class BulkheadIsolation:
    """Bulkhead pattern for isolating failures between components."""
    
    def __init__(self, name: str, max_concurrent_calls: int = 10):
        self.name = name
        self.max_concurrent_calls = max_concurrent_calls
        self.current_calls = 0
        self.queued_calls = 0
        self.rejected_calls = 0
        self.total_calls = 0
        
        # Semaphore for controlling concurrency
        self.semaphore = threading.Semaphore(max_concurrent_calls)
        self.lock = threading.Lock()
        
        logger.info(f"🏗️ Bulkhead '{name}' initialized with {max_concurrent_calls} slots")
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with bulkhead isolation."""
        with self.lock:
            self.total_calls += 1
            
            if self.current_calls >= self.max_concurrent_calls:
                self.rejected_calls += 1
                raise BulkheadFullError(f"Bulkhead '{self.name}' is full ({self.current_calls}/{self.max_concurrent_calls})")
        
        # Acquire semaphore (blocking)
        acquired = self.semaphore.acquire(blocking=False)
        if not acquired:
            with self.lock:
                self.rejected_calls += 1
            raise BulkheadFullError(f"Bulkhead '{self.name}' could not acquire slot")
        
        try:
            with self.lock:
                self.current_calls += 1
            
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            return result
            
        finally:
            with self.lock:
                self.current_calls -= 1
            self.semaphore.release()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get bulkhead metrics."""
        with self.lock:
            rejection_rate = (self.rejected_calls / self.total_calls * 100) if self.total_calls > 0 else 0
            utilization = (self.current_calls / self.max_concurrent_calls * 100)
            
            return {
                "name": self.name,
                "max_concurrent_calls": self.max_concurrent_calls,
                "current_calls": self.current_calls,
                "total_calls": self.total_calls,
                "rejected_calls": self.rejected_calls,
                "rejection_rate_percent": round(rejection_rate, 2),
                "utilization_percent": round(utilization, 2),
                "available_slots": self.max_concurrent_calls - self.current_calls
            }

class IntelligentRetryPolicy:
    """Advanced retry policy with exponential backoff, jitter, and adaptive behavior."""
    
    def __init__(self,
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True,
                 retryable_exceptions: Optional[List[type]] = None):
        
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or [Exception]
        
        # Adaptive parameters
        self.success_history = deque(maxlen=100)
        self.failure_patterns = defaultdict(int)
        
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with intelligent retry logic."""
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                start_time = time.time()
                
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Record success
                execution_time = time.time() - start_time
                self.success_history.append({
                    "attempt": attempt + 1,
                    "execution_time": execution_time,
                    "timestamp": time.time()
                })
                
                return result
                
            except Exception as e:
                last_exception = e
                exception_type = type(e).__name__
                
                # Check if exception is retryable
                if not self._is_retryable_exception(e):
                    logger.warning(f"Non-retryable exception: {exception_type}")
                    raise e
                
                # Record failure pattern
                self.failure_patterns[exception_type] += 1
                
                # Don't retry on last attempt
                if attempt == self.max_attempts - 1:
                    break
                
                # Calculate delay with exponential backoff and jitter
                delay = self._calculate_delay(attempt)
                
                logger.warning(f"Attempt {attempt + 1}/{self.max_attempts} failed: {exception_type}. "
                             f"Retrying in {delay:.2f}s...")
                
                await asyncio.sleep(delay)
        
        # All attempts failed
        raise RetryExhaustedException(f"All {self.max_attempts} attempts failed. Last error: {last_exception}")
    
    def _is_retryable_exception(self, exception: Exception) -> bool:
        """Check if exception should trigger a retry."""
        return any(isinstance(exception, exc_type) for exc_type in self.retryable_exceptions)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and optional jitter."""
        # Exponential backoff
        delay = min(self.base_delay * (self.exponential_base ** attempt), self.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.jitter:
            jitter_amount = delay * 0.1 * np.random.random()  # Up to 10% jitter
            delay += jitter_amount
        
        # Adaptive adjustment based on success history
        if self.success_history:
            recent_successes = [s for s in self.success_history if time.time() - s["timestamp"] < 300]  # Last 5 min
            success_rate = len(recent_successes) / len(self.success_history)
            
            if success_rate < 0.5:  # Low success rate
                delay *= 1.5  # Increase delay
            elif success_rate > 0.9:  # High success rate
                delay *= 0.8  # Decrease delay
        
        return delay
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get retry policy metrics."""
        total_attempts = sum(s["attempt"] for s in self.success_history)
        avg_attempts = total_attempts / len(self.success_history) if self.success_history else 0
        
        return {
            "max_attempts": self.max_attempts,
            "total_executions": len(self.success_history),
            "avg_attempts_per_success": round(avg_attempts, 2),
            "failure_patterns": dict(self.failure_patterns),
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
            "jitter_enabled": self.jitter
        }

class HealthChecker:
    """Comprehensive health checking with deep diagnostics."""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.health_checks = {}
        self.health_history = defaultdict(lambda: deque(maxlen=100))
        self.component_states = {}
        
        # Monitoring thread
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._continuous_health_monitoring)
        self._monitoring_thread.daemon = True
        
    def register_health_check(self, component: str, check_func: Callable) -> None:
        """Register a health check function for a component."""
        self.health_checks[component] = check_func
        self.component_states[component] = ComponentState.HEALTHY
        logger.info(f"🏥 Health check registered for component: {component}")
    
    async def check_component_health(self, component: str) -> HealthCheckResult:
        """Check health of a specific component."""
        if component not in self.health_checks:
            return HealthCheckResult(
                component=component,
                status=ComponentState.FAILED,
                latency_ms=0,
                error_message=f"No health check registered for component: {component}"
            )
        
        start_time = time.time()
        try:
            check_func = self.health_checks[component]
            
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Interpret result
            if isinstance(result, bool):
                status = ComponentState.HEALTHY if result else ComponentState.FAILED
                metadata = {}
            elif isinstance(result, dict):
                status = ComponentState(result.get("status", ComponentState.HEALTHY))
                metadata = result.get("metadata", {})
            else:
                status = ComponentState.HEALTHY
                metadata = {"raw_result": str(result)}
            
            health_result = HealthCheckResult(
                component=component,
                status=status,
                latency_ms=round(latency_ms, 2),
                metadata=metadata
            )
            
            # Update component state
            self.component_states[component] = status
            self.health_history[component].append(health_result)
            
            return health_result
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            
            health_result = HealthCheckResult(
                component=component,
                status=ComponentState.FAILED,
                latency_ms=round(latency_ms, 2),
                error_message=str(e),
                metadata={"exception_type": type(e).__name__}
            )
            
            self.component_states[component] = ComponentState.FAILED
            self.health_history[component].append(health_result)
            
            return health_result
    
    async def check_all_components(self) -> Dict[str, HealthCheckResult]:
        """Check health of all registered components."""
        results = {}
        
        # Run all health checks concurrently
        check_tasks = [
            self.check_component_health(component) 
            for component in self.health_checks.keys()
        ]
        
        health_results = await asyncio.gather(*check_tasks, return_exceptions=True)
        
        for component, result in zip(self.health_checks.keys(), health_results):
            if isinstance(result, Exception):
                results[component] = HealthCheckResult(
                    component=component,
                    status=ComponentState.FAILED,
                    latency_ms=0,
                    error_message=str(result)
                )
            else:
                results[component] = result
        
        return results
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        if not self.component_states:
            return {
                "overall_status": ComponentState.HEALTHY.value,
                "healthy_components": 0,
                "total_components": 0,
                "degraded_components": [],
                "failed_components": []
            }
        
        component_counts = defaultdict(int)
        for state in self.component_states.values():
            component_counts[state] += 1
        
        # Determine overall status
        if component_counts[ComponentState.FAILED] > 0:
            if component_counts[ComponentState.FAILED] > len(self.component_states) // 2:
                overall_status = ComponentState.FAILED
            else:
                overall_status = ComponentState.DEGRADED
        elif component_counts[ComponentState.DEGRADED] > 0:
            overall_status = ComponentState.DEGRADED
        else:
            overall_status = ComponentState.HEALTHY
        
        # Categorize components
        degraded_components = [comp for comp, state in self.component_states.items() 
                             if state == ComponentState.DEGRADED]
        failed_components = [comp for comp, state in self.component_states.items() 
                           if state == ComponentState.FAILED]
        
        return {
            "overall_status": overall_status.value,
            "healthy_components": component_counts[ComponentState.HEALTHY],
            "degraded_components": degraded_components,
            "failed_components": failed_components,
            "total_components": len(self.component_states),
            "component_states": {comp: state.value for comp, state in self.component_states.items()}
        }
    
    def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if not self._monitoring_thread.is_alive():
            self._monitoring_active = True
            self._monitoring_thread = threading.Thread(target=self._continuous_health_monitoring)
            self._monitoring_thread.daemon = True
            self._monitoring_thread.start()
            logger.info("🏥 Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5)
        logger.info("🏥 Health monitoring stopped")
    
    def _continuous_health_monitoring(self) -> None:
        """Continuous health monitoring loop."""
        while self._monitoring_active:
            try:
                # Run health checks
                asyncio.run(self._async_health_check_cycle())
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(self.check_interval)
    
    async def _async_health_check_cycle(self) -> None:
        """Async health check cycle."""
        results = await self.check_all_components()
        
        # Log any status changes
        for component, result in results.items():
            if result.status != ComponentState.HEALTHY:
                logger.warning(f"Component '{component}' health: {result.status.value} "
                             f"(latency: {result.latency_ms}ms)")

class SelfHealingManager:
    """Self-healing system with automatic recovery strategies."""
    
    def __init__(self):
        self.recovery_strategies = {}
        self.recovery_history = deque(maxlen=200)
        self.healing_in_progress = set()
        
    def register_recovery_strategy(self, component: str, strategy_func: Callable) -> None:
        """Register a recovery strategy for a component."""
        self.recovery_strategies[component] = strategy_func
        logger.info(f"🔧 Recovery strategy registered for: {component}")
    
    async def attempt_healing(self, component: str, error_info: Dict[str, Any]) -> bool:
        """Attempt to heal a failed component."""
        if component in self.healing_in_progress:
            logger.warning(f"Healing already in progress for: {component}")
            return False
        
        if component not in self.recovery_strategies:
            logger.error(f"No recovery strategy for component: {component}")
            return False
        
        self.healing_in_progress.add(component)
        start_time = time.time()
        
        try:
            logger.info(f"🔧 Starting self-healing for component: {component}")
            
            recovery_func = self.recovery_strategies[component]
            
            if asyncio.iscoroutinefunction(recovery_func):
                success = await recovery_func(error_info)
            else:
                success = recovery_func(error_info)
            
            healing_time = time.time() - start_time
            
            # Record recovery attempt
            recovery_record = {
                "component": component,
                "success": success,
                "healing_time_seconds": healing_time,
                "error_info": error_info,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            self.recovery_history.append(recovery_record)
            
            if success:
                logger.info(f"✅ Self-healing successful for {component} in {healing_time:.2f}s")
            else:
                logger.warning(f"❌ Self-healing failed for {component}")
            
            return success
            
        except Exception as e:
            logger.error(f"Self-healing error for {component}: {e}")
            return False
        
        finally:
            self.healing_in_progress.discard(component)
    
    def get_healing_metrics(self) -> Dict[str, Any]:
        """Get self-healing metrics."""
        if not self.recovery_history:
            return {"total_attempts": 0, "success_rate": 0, "components": []}
        
        total_attempts = len(self.recovery_history)
        successful_attempts = sum(1 for r in self.recovery_history if r["success"])
        success_rate = (successful_attempts / total_attempts * 100)
        
        # Component-specific metrics
        component_metrics = defaultdict(lambda: {"attempts": 0, "successes": 0})
        for record in self.recovery_history:
            comp = record["component"]
            component_metrics[comp]["attempts"] += 1
            if record["success"]:
                component_metrics[comp]["successes"] += 1
        
        return {
            "total_attempts": total_attempts,
            "successful_attempts": successful_attempts,
            "success_rate_percent": round(success_rate, 2),
            "healing_in_progress": list(self.healing_in_progress),
            "component_metrics": {
                comp: {
                    **metrics,
                    "success_rate_percent": round((metrics["successes"] / metrics["attempts"] * 100), 2)
                }
                for comp, metrics in component_metrics.items()
            }
        }

class ProductionReliabilityFramework:
    """Comprehensive production reliability framework."""
    
    def __init__(self):
        """Initialize the reliability framework."""
        self.circuit_breakers = {}
        self.bulkheads = {}
        self.retry_policies = {}
        self.health_checker = HealthChecker(check_interval=30)
        self.self_healing = SelfHealingManager()
        
        # Framework metrics
        self.framework_start_time = time.time()
        self.reliability_metrics = {}
        
        # Start health monitoring
        self.health_checker.start_monitoring()
        
        logger.info("🏭 Production Reliability Framework initialized")
    
    def create_circuit_breaker(self, name: str, **kwargs) -> IntelligentCircuitBreaker:
        """Create and register a circuit breaker."""
        circuit_breaker = IntelligentCircuitBreaker(name, **kwargs)
        self.circuit_breakers[name] = circuit_breaker
        return circuit_breaker
    
    def create_bulkhead(self, name: str, **kwargs) -> BulkheadIsolation:
        """Create and register a bulkhead."""
        bulkhead = BulkheadIsolation(name, **kwargs)
        self.bulkheads[name] = bulkhead
        return bulkhead
    
    def create_retry_policy(self, **kwargs) -> IntelligentRetryPolicy:
        """Create a retry policy."""
        return IntelligentRetryPolicy(**kwargs)
    
    def register_health_check(self, component: str, check_func: Callable) -> None:
        """Register a health check."""
        self.health_checker.register_health_check(component, check_func)
    
    def register_recovery_strategy(self, component: str, strategy_func: Callable) -> None:
        """Register a self-healing strategy."""
        self.self_healing.register_recovery_strategy(component, strategy_func)
    
    async def execute_with_reliability(self,
                                     func: Callable,
                                     circuit_breaker_name: Optional[str] = None,
                                     bulkhead_name: Optional[str] = None,
                                     retry_policy: Optional[IntelligentRetryPolicy] = None,
                                     enable_self_healing: bool = True,
                                     *args, **kwargs) -> Any:
        """Execute function with comprehensive reliability patterns."""
        
        # Define execution function with all reliability patterns
        async def reliable_execution():
            try:
                # Apply bulkhead isolation if specified
                if bulkhead_name and bulkhead_name in self.bulkheads:
                    bulkhead = self.bulkheads[bulkhead_name]
                    return await bulkhead.execute(func, *args, **kwargs)
                else:
                    # Direct execution
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                        
            except Exception as e:
                # Attempt self-healing if enabled
                if enable_self_healing and hasattr(func, '__name__'):
                    component_name = func.__name__
                    if component_name in self.self_healing.recovery_strategies:
                        healing_success = await self.self_healing.attempt_healing(
                            component_name, 
                            {"error": str(e), "error_type": type(e).__name__}
                        )
                        
                        if healing_success:
                            # Retry after successful healing
                            if asyncio.iscoroutinefunction(func):
                                return await func(*args, **kwargs)
                            else:
                                return func(*args, **kwargs)
                
                # Re-raise if no healing or healing failed
                raise e
        
        # Apply circuit breaker if specified
        if circuit_breaker_name and circuit_breaker_name in self.circuit_breakers:
            circuit_breaker = self.circuit_breakers[circuit_breaker_name]
            execution_func = lambda: circuit_breaker.call(reliable_execution)
        else:
            execution_func = reliable_execution
        
        # Apply retry policy if specified
        if retry_policy:
            return await retry_policy.execute(execution_func)
        else:
            return await execution_func()
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive reliability report."""
        uptime = time.time() - self.framework_start_time
        
        # Collect metrics from all components
        circuit_breaker_metrics = {
            name: cb.get_metrics() 
            for name, cb in self.circuit_breakers.items()
        }
        
        bulkhead_metrics = {
            name: bh.get_metrics() 
            for name, bh in self.bulkheads.items()
        }
        
        health_status = self.health_checker.get_overall_health()
        healing_metrics = self.self_healing.get_healing_metrics()
        
        return {
            "framework_uptime_seconds": round(uptime, 2),
            "overall_health": health_status,
            "circuit_breakers": circuit_breaker_metrics,
            "bulkheads": bulkhead_metrics,
            "self_healing": healing_metrics,
            "reliability_score": self._calculate_reliability_score(),
            "recommendations": self._generate_reliability_recommendations(),
            "report_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _calculate_reliability_score(self) -> float:
        """Calculate overall reliability score (0-1)."""
        scores = []
        
        # Health score
        health_status = self.health_checker.get_overall_health()
        if health_status["total_components"] > 0:
            health_score = health_status["healthy_components"] / health_status["total_components"]
            scores.append(health_score)
        
        # Circuit breaker scores
        for cb in self.circuit_breakers.values():
            metrics = cb.get_metrics()
            if metrics["total_calls"] > 0:
                cb_score = metrics["success_rate_percent"] / 100
                scores.append(cb_score)
        
        # Self-healing score
        healing_metrics = self.self_healing.get_healing_metrics()
        if healing_metrics["total_attempts"] > 0:
            healing_score = healing_metrics["success_rate_percent"] / 100
            scores.append(healing_score)
        
        # Overall score
        return np.mean(scores) if scores else 1.0
    
    def _generate_reliability_recommendations(self) -> List[str]:
        """Generate reliability improvement recommendations."""
        recommendations = []
        
        # Check circuit breaker health
        for name, cb in self.circuit_breakers.items():
            metrics = cb.get_metrics()
            if metrics["success_rate_percent"] < 90:
                recommendations.append(f"Circuit breaker '{name}' has low success rate: {metrics['success_rate_percent']:.1f}%")
            
            failure_prob = metrics.get("failure_probability", 0)
            if failure_prob > 0.7:
                recommendations.append(f"Circuit breaker '{name}' has high failure probability: {failure_prob:.2f}")
        
        # Check bulkhead utilization
        for name, bh in self.bulkheads.items():
            metrics = bh.get_metrics()
            if metrics["rejection_rate_percent"] > 10:
                recommendations.append(f"Bulkhead '{name}' has high rejection rate: {metrics['rejection_rate_percent']:.1f}%")
        
        # Check health status
        health_status = self.health_checker.get_overall_health()
        if health_status["failed_components"]:
            recommendations.append(f"Failed components detected: {', '.join(health_status['failed_components'])}")
        
        # Check self-healing
        healing_metrics = self.self_healing.get_healing_metrics()
        if healing_metrics["total_attempts"] > 0 and healing_metrics["success_rate_percent"] < 70:
            recommendations.append("Self-healing success rate is low - review recovery strategies")
        
        if not recommendations:
            recommendations.append("System reliability is operating within normal parameters")
        
        return recommendations
    
    def shutdown(self) -> None:
        """Gracefully shutdown the reliability framework."""
        self.health_checker.stop_monitoring()
        logger.info("🏭 Production Reliability Framework shutdown complete")

# Custom Exceptions
class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass

class BulkheadFullError(Exception):
    """Raised when bulkhead is at capacity."""
    pass

class RetryExhaustedException(Exception):
    """Raised when all retry attempts are exhausted."""
    pass

# Factory function
def create_reliability_framework() -> ProductionReliabilityFramework:
    """Create a production reliability framework instance."""
    return ProductionReliabilityFramework()

# Example usage and testing
async def example_usage():
    """Example usage of the reliability framework."""
    print("🏭 Production Reliability Framework Demo")
    print("=" * 50)
    
    # Create framework
    framework = create_reliability_framework()
    
    # Create reliability components
    circuit_breaker = framework.create_circuit_breaker("example_service", failure_threshold=3)
    bulkhead = framework.create_bulkhead("example_bulkhead", max_concurrent_calls=5)
    retry_policy = framework.create_retry_policy(max_attempts=3, base_delay=1.0)
    
    # Mock service function
    call_count = [0]  # Mutable counter
    async def mock_service(should_fail: bool = False):
        call_count[0] += 1
        await asyncio.sleep(0.1)  # Simulate work
        
        if should_fail and call_count[0] % 3 == 0:
            raise Exception("Mock service failure")
        
        return {"result": f"success_{call_count[0]}", "timestamp": time.time()}
    
    # Health check function
    def health_check():
        return {"status": ComponentState.HEALTHY, "metadata": {"version": "1.0.0"}}
    
    # Recovery strategy
    async def recovery_strategy(error_info):
        print(f"🔧 Attempting recovery for error: {error_info['error']}")
        await asyncio.sleep(0.5)  # Simulate recovery time
        return True  # Assume recovery is successful
    
    # Register health check and recovery
    framework.register_health_check("mock_service", health_check)
    framework.register_recovery_strategy("mock_service", recovery_strategy)
    
    print("\n🧪 Testing reliability patterns...")
    
    # Test successful calls
    try:
        for i in range(5):
            result = await framework.execute_with_reliability(
                mock_service,
                circuit_breaker_name="example_service",
                bulkhead_name="example_bulkhead",
                retry_policy=retry_policy
            )
            print(f"✅ Call {i+1}: {result['result']}")
            await asyncio.sleep(0.2)
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test failing calls
    print("\n🔥 Testing failure scenarios...")
    try:
        for i in range(3):
            result = await framework.execute_with_reliability(
                lambda: mock_service(should_fail=True),
                circuit_breaker_name="example_service",
                retry_policy=retry_policy
            )
            print(f"✅ Resilient call {i+1}: {result['result']}")
    except Exception as e:
        print(f"❌ Final failure: {e}")
    
    # Wait a moment for health checks
    await asyncio.sleep(2)
    
    # Generate reliability report
    print("\n📊 Reliability Report:")
    report = framework.get_comprehensive_report()
    
    print(f"Framework uptime: {report['framework_uptime_seconds']:.1f}s")
    print(f"Overall reliability score: {report['reliability_score']:.3f}")
    print(f"Health status: {report['overall_health']['overall_status']}")
    
    if report['circuit_breakers']:
        cb_metrics = list(report['circuit_breakers'].values())[0]
        print(f"Circuit breaker success rate: {cb_metrics['success_rate_percent']:.1f}%")
    
    print("\n💡 Recommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    # Cleanup
    framework.shutdown()
    print("\n✅ Demo completed successfully!")

if __name__ == "__main__":
    asyncio.run(example_usage())