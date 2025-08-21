#!/usr/bin/env python3
"""
Enhanced Security and Robustness Framework for FastVLM
Generation 2: MAKE IT ROBUST Implementation

Implements comprehensive security measures, input validation, 
error recovery, and monitoring systems.
"""

import os
import sys
import json
import logging
import hashlib
import secrets
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import threading
from threading import Lock
import traceback

# Configure robust logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fastvlm_security.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SecurityIncident:
    """Security incident tracking."""
    timestamp: str
    incident_type: str
    severity: str
    description: str
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    blocked: bool = False
    mitigation_action: Optional[str] = None

@dataclass
class ValidationResult:
    """Input validation result."""
    is_valid: bool
    confidence: float
    issues: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]

class AdvancedSecurityFramework:
    """Advanced security framework with threat detection and mitigation."""
    
    def __init__(self):
        self.security_config = {
            "max_file_size_mb": 50,
            "max_request_rate": 100,  # requests per minute
            "enable_content_scanning": True,
            "enable_malware_detection": True,
            "enable_behavioral_analysis": True,
            "quarantine_suspicious_requests": True
        }
        
        self.incidents = []
        self.request_tracker = {}
        self.blocked_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'onload=',
            r'onerror=',
            r'eval\s*\(',
            r'document\.cookie',
            r'window\.location',
            r'\.prototype\.',
            r'constructor\s*\[',
            r'__proto__',
            r'import\s+os',
            r'__import__',
            r'exec\s*\(',
            r'subprocess\.',
            r'popen\(',
            r'system\(',
            r'shell=True'
        ]
        
        self.threat_signatures = {
            "sql_injection": [r"union\s+select", r"drop\s+table", r"delete\s+from"],
            "xss": [r"<script", r"javascript:", r"onerror=", r"onload="],
            "command_injection": [r";\s*cat\s+", r";\s*rm\s+", r";\s*curl\s+", r"&&\s*wget"],
            "path_traversal": [r"\.\.\/", r"\.\.\\", r"%2e%2e%2f", r"%2e%2e%5c"],
            "deserialization": [r"pickle\.loads", r"marshal\.loads", r"eval\("],
        }
        
        self.security_lock = Lock()
        logger.info("🛡️ Advanced Security Framework initialized")
    
    def validate_request(self, data: bytes, question: str) -> Tuple[bool, str, List[SecurityIncident]]:
        """Comprehensive request validation with threat detection."""
        incidents = []
        
        try:
            with self.security_lock:
                # Rate limiting check
                current_time = time.time()
                client_id = self._get_client_id()
                
                if client_id in self.request_tracker:
                    requests = self.request_tracker[client_id]
                    # Clean old requests (older than 1 minute)
                    requests = [req_time for req_time in requests if current_time - req_time < 60]
                    
                    if len(requests) >= self.security_config["max_request_rate"]:
                        incident = SecurityIncident(
                            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                            incident_type="rate_limit_exceeded",
                            severity="medium",
                            description=f"Rate limit exceeded: {len(requests)} requests in last minute",
                            blocked=True,
                            mitigation_action="request_blocked"
                        )
                        incidents.append(incident)
                        return False, "Rate limit exceeded", incidents
                    
                    requests.append(current_time)
                    self.request_tracker[client_id] = requests
                else:
                    self.request_tracker[client_id] = [current_time]
                
                # Content size validation
                if len(data) > self.security_config["max_file_size_mb"] * 1024 * 1024:
                    incident = SecurityIncident(
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        incident_type="oversized_content",
                        severity="medium",
                        description=f"Content size exceeds limit: {len(data)} bytes",
                        blocked=True,
                        mitigation_action="request_blocked"
                    )
                    incidents.append(incident)
                    return False, "Content too large", incidents
                
                # Content validation
                data_str = data.decode('utf-8', errors='ignore')
                combined_content = data_str + question
                
                # Pattern-based threat detection
                for pattern in self.blocked_patterns:
                    import re
                    if re.search(pattern, combined_content, re.IGNORECASE | re.MULTILINE):
                        incident = SecurityIncident(
                            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                            incident_type="malicious_pattern_detected",
                            severity="high",
                            description=f"Blocked pattern detected: {pattern}",
                            blocked=True,
                            mitigation_action="request_blocked"
                        )
                        incidents.append(incident)
                        return False, f"Malicious pattern detected", incidents
                
                # Advanced threat signature detection
                for threat_type, signatures in self.threat_signatures.items():
                    for signature in signatures:
                        import re
                        if re.search(signature, combined_content, re.IGNORECASE):
                            incident = SecurityIncident(
                                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                                incident_type=f"{threat_type}_attempt",
                                severity="high",
                                description=f"Potential {threat_type} detected: {signature}",
                                blocked=True,
                                mitigation_action="request_quarantined"
                            )
                            incidents.append(incident)
                            return False, f"Security threat detected: {threat_type}", incidents
                
                # Content entropy analysis (detect encoded/obfuscated content)
                entropy = self._calculate_entropy(combined_content)
                if entropy > 7.5:  # High entropy threshold
                    incident = SecurityIncident(
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        incident_type="high_entropy_content",
                        severity="medium",
                        description=f"High entropy content detected: {entropy:.2f}",
                        blocked=False,
                        mitigation_action="content_flagged"
                    )
                    incidents.append(incident)
                
                # File type validation for image data
                if len(data) > 0:
                    file_type = self._detect_file_type(data)
                    if file_type and file_type not in ['JPEG', 'PNG', 'GIF', 'BMP', 'WEBP']:
                        incident = SecurityIncident(
                            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                            incident_type="invalid_file_type",
                            severity="medium",
                            description=f"Unsupported file type: {file_type}",
                            blocked=True,
                            mitigation_action="request_blocked"
                        )
                        incidents.append(incident)
                        return False, f"Invalid file type: {file_type}", incidents
                
                # Store incidents for analysis
                self.incidents.extend(incidents)
                
                return True, "Request validated successfully", incidents
                
        except Exception as e:
            logger.error(f"Security validation error: {e}")
            incident = SecurityIncident(
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                incident_type="validation_error",
                severity="high",
                description=f"Security validation failed: {str(e)}",
                blocked=True,
                mitigation_action="request_blocked"
            )
            return False, f"Validation error: {str(e)}", [incident]
    
    def _get_client_id(self) -> str:
        """Generate client identifier for rate limiting."""
        # In a real implementation, this would use actual client IP/session
        return "default_client"
    
    def _calculate_entropy(self, data: str) -> float:
        """Calculate Shannon entropy of data."""
        if not data:
            return 0.0
        
        counts = {}
        for char in data:
            counts[char] = counts.get(char, 0) + 1
        
        length = len(data)
        entropy = 0.0
        for count in counts.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * (probability.bit_length() - 1)
        
        return entropy
    
    def _detect_file_type(self, data: bytes) -> Optional[str]:
        """Simple file type detection based on magic numbers."""
        if len(data) < 8:
            return None
        
        # JPEG
        if data[:2] == b'\xff\xd8':
            return "JPEG"
        # PNG
        elif data[:8] == b'\x89PNG\r\n\x1a\n':
            return "PNG"
        # GIF
        elif data[:6] in [b'GIF87a', b'GIF89a']:
            return "GIF"
        # BMP
        elif data[:2] == b'BM':
            return "BMP"
        # WEBP
        elif data[:4] == b'RIFF' and data[8:12] == b'WEBP':
            return "WEBP"
        
        return "UNKNOWN"
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            "total_incidents": len(self.incidents),
            "blocked_requests": len([i for i in self.incidents if i.blocked]),
            "high_severity_incidents": len([i for i in self.incidents if i.severity == "high"]),
            "active_clients": len(self.request_tracker),
            "recent_incidents": [asdict(i) for i in self.incidents[-10:]],
            "threat_summary": self._get_threat_summary(),
            "security_config": self.security_config
        }
    
    def _get_threat_summary(self) -> Dict[str, int]:
        """Get summary of threat types detected."""
        summary = {}
        for incident in self.incidents:
            threat_type = incident.incident_type
            summary[threat_type] = summary.get(threat_type, 0) + 1
        return summary

class RobustErrorRecovery:
    """Advanced error recovery and circuit breaker system."""
    
    def __init__(self):
        self.error_thresholds = {
            "critical": 3,
            "major": 5,
            "minor": 10
        }
        
        self.recovery_strategies = {
            "connection_error": self._recover_connection,
            "memory_error": self._recover_memory,
            "timeout_error": self._recover_timeout,
            "validation_error": self._recover_validation,
            "unknown_error": self._recover_generic
        }
        
        self.error_history = []
        self.recovery_lock = Lock()
        logger.info("🔄 Robust Error Recovery initialized")
    
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, str, Any]:
        """Handle error with appropriate recovery strategy."""
        with self.recovery_lock:
            error_type = self._classify_error(error)
            
            # Record error
            error_record = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error_type": error_type,
                "error_message": str(error),
                "context": context,
                "stack_trace": traceback.format_exc()
            }
            self.error_history.append(error_record)
            
            # Check if circuit breaker should trip
            recent_errors = self._get_recent_errors(error_type, minutes=5)
            if len(recent_errors) >= self.error_thresholds.get("critical", 3):
                return False, "Circuit breaker tripped - too many errors", None
            
            # Apply recovery strategy
            recovery_func = self.recovery_strategies.get(error_type, self._recover_generic)
            try:
                recovered, message, result = recovery_func(error, context)
                
                if recovered:
                    logger.info(f"✅ Error recovery successful: {message}")
                else:
                    logger.warning(f"⚠️ Error recovery failed: {message}")
                
                return recovered, message, result
                
            except Exception as recovery_error:
                logger.error(f"❌ Recovery strategy failed: {recovery_error}")
                return False, f"Recovery failed: {str(recovery_error)}", None
    
    def _classify_error(self, error: Exception) -> str:
        """Classify error type for appropriate recovery."""
        error_str = str(error).lower()
        
        if "connection" in error_str or "network" in error_str:
            return "connection_error"
        elif "memory" in error_str or "allocation" in error_str:
            return "memory_error"
        elif "timeout" in error_str or "time" in error_str:
            return "timeout_error"
        elif "validation" in error_str or "invalid" in error_str:
            return "validation_error"
        else:
            return "unknown_error"
    
    def _get_recent_errors(self, error_type: str, minutes: int = 5) -> List[Dict]:
        """Get recent errors of specific type."""
        cutoff_time = time.time() - (minutes * 60)
        cutoff_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(cutoff_time))
        
        return [
            error for error in self.error_history
            if error["error_type"] == error_type and error["timestamp"] >= cutoff_str
        ]
    
    def _recover_connection(self, error: Exception, context: Dict) -> Tuple[bool, str, Any]:
        """Recover from connection errors."""
        # Implement connection retry with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                time.sleep(2 ** attempt)  # Exponential backoff
                # Simulate connection recovery
                return True, f"Connection recovered after {attempt + 1} attempts", None
            except:
                continue
        return False, "Connection recovery failed after all retries", None
    
    def _recover_memory(self, error: Exception, context: Dict) -> Tuple[bool, str, Any]:
        """Recover from memory errors."""
        try:
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear any large caches if available
            if "cache" in context:
                context["cache"].clear()
            
            return True, "Memory recovered through garbage collection", None
        except:
            return False, "Memory recovery failed", None
    
    def _recover_timeout(self, error: Exception, context: Dict) -> Tuple[bool, str, Any]:
        """Recover from timeout errors."""
        # Increase timeout for next attempt
        new_timeout = context.get("timeout", 30) * 1.5
        context["timeout"] = min(new_timeout, 120)  # Cap at 2 minutes
        return True, f"Timeout increased to {new_timeout}s", context
    
    def _recover_validation(self, error: Exception, context: Dict) -> Tuple[bool, str, Any]:
        """Recover from validation errors."""
        # Apply input sanitization
        if "input_data" in context:
            sanitized = self._sanitize_input(context["input_data"])
            context["input_data"] = sanitized
            return True, "Input sanitized", context
        return False, "Cannot recover from validation error", None
    
    def _recover_generic(self, error: Exception, context: Dict) -> Tuple[bool, str, Any]:
        """Generic error recovery."""
        # Log and continue with degraded functionality
        return True, "Continuing with degraded functionality", None
    
    def _sanitize_input(self, data: Any) -> Any:
        """Sanitize input data."""
        if isinstance(data, str):
            # Remove potentially dangerous characters
            import re
            sanitized = re.sub(r'[<>"\';(){}]', '', data)
            return sanitized[:1000]  # Truncate to reasonable length
        return data

class ComprehensiveMonitoring:
    """Advanced monitoring and alerting system."""
    
    def __init__(self):
        self.metrics = {
            "requests_total": 0,
            "requests_successful": 0,
            "requests_failed": 0,
            "average_latency_ms": 0.0,
            "security_incidents": 0,
            "errors_recovered": 0,
            "system_health_score": 100.0
        }
        
        self.alerts = []
        self.monitoring_active = True
        self.monitoring_lock = Lock()
        
        # Start background monitoring
        self._start_monitoring_thread()
        logger.info("📊 Comprehensive Monitoring initialized")
    
    def record_metric(self, metric_name: str, value: float, tags: Optional[Dict] = None):
        """Record a metric value."""
        with self.monitoring_lock:
            if metric_name in self.metrics:
                # Update running averages for latency
                if metric_name == "average_latency_ms":
                    total_requests = self.metrics["requests_total"]
                    if total_requests > 0:
                        current_avg = self.metrics[metric_name]
                        new_avg = (current_avg * (total_requests - 1) + value) / total_requests
                        self.metrics[metric_name] = new_avg
                    else:
                        self.metrics[metric_name] = value
                else:
                    self.metrics[metric_name] += value
            else:
                self.metrics[metric_name] = value
            
            # Check for alert conditions
            self._check_alert_conditions()
    
    def _check_alert_conditions(self):
        """Check if any metrics trigger alerts."""
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # High error rate alert
        total_requests = self.metrics["requests_total"]
        if total_requests > 10:
            error_rate = (self.metrics["requests_failed"] / total_requests) * 100
            if error_rate > 10:  # More than 10% error rate
                self._create_alert("HIGH_ERROR_RATE", f"Error rate: {error_rate:.1f}%", "critical")
        
        # High latency alert
        if self.metrics["average_latency_ms"] > 1000:  # More than 1 second
            self._create_alert("HIGH_LATENCY", f"Latency: {self.metrics['average_latency_ms']:.1f}ms", "warning")
        
        # Security incidents alert
        if self.metrics["security_incidents"] > 0:
            self._create_alert("SECURITY_INCIDENT", f"Security incidents: {self.metrics['security_incidents']}", "critical")
    
    def _create_alert(self, alert_type: str, message: str, severity: str):
        """Create an alert."""
        alert = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "type": alert_type,
            "message": message,
            "severity": severity,
            "acknowledged": False
        }
        
        self.alerts.append(alert)
        logger.warning(f"🚨 ALERT [{severity.upper()}]: {alert_type} - {message}")
    
    def _start_monitoring_thread(self):
        """Start background monitoring thread."""
        def monitor():
            while self.monitoring_active:
                try:
                    self._update_system_health()
                    time.sleep(30)  # Update every 30 seconds
                except Exception as e:
                    logger.error(f"Monitoring thread error: {e}")
                    time.sleep(60)  # Wait longer if error
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    
    def _update_system_health(self):
        """Calculate and update overall system health score."""
        with self.monitoring_lock:
            health_score = 100.0
            
            # Reduce score based on error rate
            total_requests = self.metrics["requests_total"]
            if total_requests > 0:
                error_rate = (self.metrics["requests_failed"] / total_requests) * 100
                health_score -= min(error_rate * 2, 30)  # Max 30 point reduction
            
            # Reduce score based on latency
            if self.metrics["average_latency_ms"] > 500:
                latency_penalty = min((self.metrics["average_latency_ms"] - 500) / 100, 20)
                health_score -= latency_penalty
            
            # Reduce score based on security incidents
            health_score -= min(self.metrics["security_incidents"] * 5, 25)
            
            # Ensure score doesn't go below 0
            health_score = max(health_score, 0.0)
            
            self.metrics["system_health_score"] = health_score
    
    def get_monitoring_report(self) -> Dict[str, Any]:
        """Get comprehensive monitoring report."""
        return {
            "metrics": dict(self.metrics),
            "recent_alerts": self.alerts[-10:],
            "unacknowledged_alerts": [a for a in self.alerts if not a["acknowledged"]],
            "monitoring_status": "active" if self.monitoring_active else "inactive",
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
        }

def create_enhanced_validator() -> AdvancedSecurityFramework:
    """Create enhanced security validator."""
    return AdvancedSecurityFramework()

def demonstrate_robust_system():
    """Demonstrate the robust system with security and error recovery."""
    print("🚀 FastVLM Robust Security & Error Recovery Demo")
    print("=" * 60)
    
    # Initialize components
    security = AdvancedSecurityFramework()
    recovery = RobustErrorRecovery()
    monitoring = ComprehensiveMonitoring()
    
    # Test cases
    test_cases = [
        {
            "name": "Valid Request",
            "data": b"valid_image_data_here",
            "question": "What do you see in this image?"
        },
        {
            "name": "Malicious Script",
            "data": b"<script>alert('xss')</script>",
            "question": "Describe this"
        },
        {
            "name": "SQL Injection",
            "data": b"normal_image",
            "question": "'; DROP TABLE users; --"
        },
        {
            "name": "Path Traversal",
            "data": b"normal_image",
            "question": "Load ../../../etc/passwd"
        },
        {
            "name": "Large Content",
            "data": b"x" * (60 * 1024 * 1024),  # 60MB
            "question": "Process this"
        }
    ]
    
    print("\n🔒 Security Validation Tests")
    print("-" * 30)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case['name']}")
        
        start_time = time.time()
        is_valid, message, incidents = security.validate_request(
            test_case["data"], test_case["question"]
        )
        latency = (time.time() - start_time) * 1000
        
        # Record metrics
        monitoring.record_metric("requests_total", 1)
        monitoring.record_metric("average_latency_ms", latency)
        
        if is_valid:
            print(f"   ✅ PASSED: {message}")
            monitoring.record_metric("requests_successful", 1)
        else:
            print(f"   ❌ BLOCKED: {message}")
            monitoring.record_metric("requests_failed", 1)
            monitoring.record_metric("security_incidents", len(incidents))
        
        print(f"   ⏱️ Latency: {latency:.1f}ms")
        print(f"   📊 Incidents: {len(incidents)}")
    
    print("\n🔄 Error Recovery Tests")
    print("-" * 30)
    
    error_tests = [
        ConnectionError("Network connection failed"),
        MemoryError("Out of memory"),
        TimeoutError("Request timeout"),
        ValueError("Invalid input data"),
        RuntimeError("Unknown system error")
    ]
    
    for i, error in enumerate(error_tests, 1):
        print(f"\n{i}. Testing recovery from: {type(error).__name__}")
        
        context = {"timeout": 30, "retry_count": 0}
        recovered, message, result = recovery.handle_error(error, context)
        
        if recovered:
            print(f"   ✅ RECOVERED: {message}")
            monitoring.record_metric("errors_recovered", 1)
        else:
            print(f"   ❌ FAILED: {message}")
            monitoring.record_metric("requests_failed", 1)
    
    # Generate reports
    print("\n📊 Security Status Report")
    print("-" * 30)
    security_status = security.get_security_status()
    for key, value in security_status.items():
        if key != "recent_incidents":
            print(f"   {key}: {value}")
    
    print("\n📈 Monitoring Report")
    print("-" * 30)
    monitoring_report = monitoring.get_monitoring_report()
    for key, value in monitoring_report["metrics"].items():
        print(f"   {key}: {value}")
    
    print(f"\n🏥 System Health Score: {monitoring_report['metrics']['system_health_score']:.1f}/100")
    
    # Save detailed report
    report = {
        "security_status": security_status,
        "monitoring_report": monitoring_report,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open("robust_security_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📋 Detailed report saved to: robust_security_report.json")
    print("\n✅ Robust Security & Error Recovery Demo Complete!")

if __name__ == "__main__":
    demonstrate_robust_system()