"""
Enhanced Security Framework for FastVLM Generation 2.
Provides comprehensive security, validation, and threat detection.
"""

import re
import hashlib
import time
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import threading
from threading import Lock

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityIncident:
    """Security incident record."""
    timestamp: str
    threat_type: str
    threat_level: ThreatLevel
    description: str
    source_ip: Optional[str] = None
    blocked: bool = False
    metadata: Dict[str, Any] = None


class ContentSecurityPolicy:
    """Content Security Policy for input validation."""
    
    def __init__(self):
        self.blocked_patterns = [
            # Script injection patterns
            r'<script[^>]*>.*?</script>',
            r'javascript:\s*[^;]+',
            r'data:\s*text/html',
            r'vbscript:\s*[^;]+',
            
            # Code execution patterns
            r'eval\s*\(',
            r'setTimeout\s*\(',
            r'setInterval\s*\(',
            r'Function\s*\(',
            
            # DOM manipulation patterns
            r'document\.',
            r'window\.',
            r'location\.',
            r'navigator\.',
            
            # SQL injection patterns
            r"(?:'|\"|`|;).*(?:union|select|insert|update|delete|drop|create|alter).*(?:'|\"|`|;)",
            r"(?:'|\").*(?:or|and).*\d+.*=.*\d+.*(?:'|\")",
            
            # Command injection patterns
            r'[;&|`$].*(?:rm|cat|ls|ps|kill|sudo|su).*[;&|`$]',
            r'(?:exec|system|shell_exec|passthru)\s*\(',
            
            # Path traversal patterns
            r'\.\.\/+',
            r'\.\.\\+',
            r'%2e%2e%2f',
            r'%2e%2e\\',
            
            # Binary/executable patterns
            r'\x00',  # Null bytes
            r'[\x01-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]',  # Control characters
        ]
        
        # Compile patterns for performance
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) 
                                for pattern in self.blocked_patterns]
        
        # File type restrictions
        self.allowed_file_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        self.blocked_file_signatures = [
            b'\x7fELF',  # ELF executables
            b'MZ',       # Windows PE
            b'\x00\x00\x01\x00',  # ICO files (can contain malicious code)
            b'PK\x03\x04',  # ZIP files
        ]
    
    def scan_content(self, content: str, content_type: str = "text") -> Tuple[bool, List[str]]:
        """Scan content for security threats."""
        threats = []
        
        # Pattern-based scanning
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(content):
                threat_desc = f"Suspicious pattern detected (rule {i+1})"
                threats.append(threat_desc)
        
        # Content-specific checks
        if content_type == "question":
            threats.extend(self._scan_question_content(content))
        elif content_type == "filename":
            threats.extend(self._scan_filename(content))
        
        is_safe = len(threats) == 0
        return is_safe, threats
    
    def _scan_question_content(self, question: str) -> List[str]:
        """Scan question text for specific threats."""
        threats = []
        
        # Check for excessive repetition (potential DoS)
        if len(set(question.lower().split())) < len(question.split()) / 10:
            threats.append("Excessive content repetition detected")
        
        # Check for encoding attacks
        try:
            # Multiple encoding attempts
            for encoding in ['utf-8', 'utf-16', 'utf-32']:
                try:
                    decoded = question.encode('latin1').decode(encoding, errors='ignore')
                    if decoded != question and len(decoded) > len(question) * 1.5:
                        threats.append("Potential encoding attack detected")
                        break
                except:
                    pass
        except:
            pass
        
        return threats
    
    def _scan_filename(self, filename: str) -> List[str]:
        """Scan filename for security issues."""
        threats = []
        
        # Check extension
        if '.' in filename:
            ext = '.' + filename.split('.')[-1].lower()
            if ext not in self.allowed_file_extensions:
                threats.append(f"Disallowed file extension: {ext}")
        
        # Check for path traversal in filename
        if '..' in filename or '/' in filename or '\\' in filename:
            threats.append("Path traversal attempt in filename")
        
        return threats
    
    def scan_binary_content(self, data: bytes) -> Tuple[bool, List[str]]:
        """Scan binary content for threats."""
        threats = []
        
        # Check file signatures
        for signature in self.blocked_file_signatures:
            if data.startswith(signature):
                threats.append(f"Blocked file signature detected: {signature.hex()}")
        
        # Check for embedded scripts in images
        data_str = data.decode('utf-8', errors='ignore')
        if len(data_str) > 100:  # Only check if we got meaningful text
            is_safe, text_threats = self.scan_content(data_str, "embedded")
            threats.extend(text_threats)
        
        # Check for excessive size patterns (potential zip bombs)
        if len(data) > 100 * 1024 * 1024:  # 100MB
            compression_ratio = len(set(data)) / len(data)
            if compression_ratio < 0.01:  # Very low entropy
                threats.append("Potential compression bomb detected")
        
        is_safe = len(threats) == 0
        return is_safe, threats


class ThreatDetectionEngine:
    """Real-time threat detection and response."""
    
    def __init__(self):
        self.csp = ContentSecurityPolicy()
        self.incident_log: List[SecurityIncident] = []
        self.blocked_sources: Dict[str, int] = {}  # IP -> block_count
        self.rate_limiter = {}  # source -> (count, window_start)
        self.lock = Lock()
        
        # Configuration
        self.max_requests_per_minute = 100
        self.max_incidents_per_source = 10
        self.auto_block_threshold = 5
        
    def scan_request(self, 
                    image_data: bytes, 
                    question: str, 
                    source_ip: Optional[str] = None) -> Tuple[bool, List[SecurityIncident]]:
        """Comprehensive request scanning."""
        incidents = []
        
        with self.lock:
            # Rate limiting check
            if source_ip and not self._check_rate_limit(source_ip):
                incident = SecurityIncident(
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    threat_type="rate_limit_exceeded",
                    threat_level=ThreatLevel.MEDIUM,
                    description="Rate limit exceeded",
                    source_ip=source_ip,
                    blocked=True
                )
                incidents.append(incident)
                self.incident_log.append(incident)
                return False, incidents
            
            # Check if source is already blocked
            if source_ip and source_ip in self.blocked_sources:
                if self.blocked_sources[source_ip] >= self.max_incidents_per_source:
                    incident = SecurityIncident(
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        threat_type="blocked_source",
                        threat_level=ThreatLevel.HIGH,
                        description="Request from blocked source",
                        source_ip=source_ip,
                        blocked=True
                    )
                    incidents.append(incident)
                    self.incident_log.append(incident)
                    return False, incidents
        
        # Image content scanning
        try:
            image_safe, image_threats = self.csp.scan_binary_content(image_data)
            if not image_safe:
                for threat in image_threats:
                    incident = SecurityIncident(
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        threat_type="malicious_image_content",
                        threat_level=ThreatLevel.HIGH,
                        description=f"Image threat: {threat}",
                        source_ip=source_ip,
                        blocked=True
                    )
                    incidents.append(incident)
        except Exception as e:
            # If scanning fails, be conservative and block
            incident = SecurityIncident(
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                threat_type="image_scanning_error",
                threat_level=ThreatLevel.MEDIUM,
                description=f"Image scanning failed: {str(e)}",
                source_ip=source_ip,
                blocked=True
            )
            incidents.append(incident)
        
        # Question content scanning
        question_safe, question_threats = self.csp.scan_content(question, "question")
        if not question_safe:
            for threat in question_threats:
                incident = SecurityIncident(
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    threat_type="malicious_question_content",
                    threat_level=ThreatLevel.HIGH,
                    description=f"Question threat: {threat}",
                    source_ip=source_ip,
                    blocked=True
                )
                incidents.append(incident)
        
        # Log incidents and update blocking
        with self.lock:
            for incident in incidents:
                self.incident_log.append(incident)
                if source_ip and incident.blocked:
                    self.blocked_sources[source_ip] = self.blocked_sources.get(source_ip, 0) + 1
        
        # Determine if request should be blocked
        blocked_incidents = [i for i in incidents if i.blocked]
        is_safe = len(blocked_incidents) == 0
        
        return is_safe, incidents
    
    def _check_rate_limit(self, source_ip: str) -> bool:
        """Check rate limiting for source IP."""
        current_time = time.time()
        window_start = current_time - 60  # 1 minute window
        
        if source_ip not in self.rate_limiter:
            self.rate_limiter[source_ip] = (1, current_time)
            return True
        
        count, last_window_start = self.rate_limiter[source_ip]
        
        # Reset window if it's been more than a minute
        if last_window_start < window_start:
            self.rate_limiter[source_ip] = (1, current_time)
            return True
        
        # Check if under limit
        if count < self.max_requests_per_minute:
            self.rate_limiter[source_ip] = (count + 1, last_window_start)
            return True
        
        return False
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        with self.lock:
            total_incidents = len(self.incident_log)
            blocked_incidents = len([i for i in self.incident_log if i.blocked])
            
            # Recent incidents (last hour)
            recent_time = time.time() - 3600
            recent_incidents = [
                i for i in self.incident_log 
                if time.mktime(time.strptime(i.timestamp, "%Y-%m-%d %H:%M:%S")) > recent_time
            ]
            
            # Threat level distribution
            threat_levels = {}
            for incident in self.incident_log:
                level = incident.threat_level.value
                threat_levels[level] = threat_levels.get(level, 0) + 1
            
            return {
                "total_incidents": total_incidents,
                "blocked_incidents": blocked_incidents,
                "recent_incidents_1h": len(recent_incidents),
                "blocked_sources": len(self.blocked_sources),
                "threat_level_distribution": threat_levels,
                "rate_limited_sources": len(self.rate_limiter),
                "status": "secure" if blocked_incidents == 0 else "under_attack" if blocked_incidents > 10 else "monitoring"
            }
    
    def clear_incidents(self, older_than_hours: int = 24) -> int:
        """Clear old incidents to prevent memory growth."""
        cutoff_time = time.time() - (older_than_hours * 3600)
        
        with self.lock:
            initial_count = len(self.incident_log)
            self.incident_log = [
                i for i in self.incident_log 
                if time.mktime(time.strptime(i.timestamp, "%Y-%m-%d %H:%M:%S")) > cutoff_time
            ]
            cleared_count = initial_count - len(self.incident_log)
            
            logger.info(f"Cleared {cleared_count} old security incidents")
            return cleared_count


class EnhancedInputValidator:
    """Enhanced input validator with security integration."""
    
    def __init__(self):
        self.threat_detector = ThreatDetectionEngine()
        self.max_image_size_mb = 50
        self.max_question_length = 1000
        self.min_image_size_bytes = 100
        
    def validate_request(self, 
                        image_data: bytes, 
                        question: str, 
                        source_ip: Optional[str] = None) -> Tuple[bool, str, List[SecurityIncident]]:
        """Comprehensive request validation."""
        
        # Basic size validations
        if len(image_data) < self.min_image_size_bytes:
            return False, "Image data too small to be valid", []
        
        image_size_mb = len(image_data) / (1024 * 1024)
        if image_size_mb > self.max_image_size_mb:
            return False, f"Image too large: {image_size_mb:.1f}MB (max: {self.max_image_size_mb}MB)", []
        
        if len(question.strip()) == 0:
            return False, "Question cannot be empty", []
        
        if len(question) > self.max_question_length:
            return False, f"Question too long: {len(question)} chars (max: {self.max_question_length})", []
        
        # Security scanning
        is_safe, incidents = self.threat_detector.scan_request(image_data, question, source_ip)
        
        if not is_safe:
            blocked_threats = [i.description for i in incidents if i.blocked]
            return False, f"Security threats detected: {'; '.join(blocked_threats)}", incidents
        
        return True, "Valid request", incidents
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get security status from threat detector."""
        return self.threat_detector.get_security_status()
    
    def clear_old_incidents(self, hours: int = 24) -> int:
        """Clear old security incidents."""
        return self.threat_detector.clear_incidents(hours)


# Factory function for creating enhanced validators
def create_enhanced_validator() -> EnhancedInputValidator:
    """Create an enhanced input validator with security features."""
    return EnhancedInputValidator()


# Factory function for threat detection
def create_threat_detector() -> ThreatDetectionEngine:
    """Create a threat detection engine."""
    return ThreatDetectionEngine()


if __name__ == "__main__":
    # Demo the enhanced security framework
    print("Enhanced Security Framework Demo")
    print("=" * 40)
    
    validator = create_enhanced_validator()
    
    # Test cases
    test_cases = [
        # (image_data, question, expected_safe, description)
        (b"valid_image_data" * 100, "What is in this image?", True, "Normal request"),
        (b"<script>alert('xss')</script>" + b"x" * 1000, "Normal question", False, "Malicious image"),
        (b"valid_image_data" * 100, "<script>alert('xss')</script>", False, "Malicious question"),
        (b"x" * (60 * 1024 * 1024), "Normal question", False, "Oversized image"),
        (b"valid_image_data" * 100, "x" * 2000, False, "Oversized question"),
        (b"", "Normal question", False, "Empty image"),
        (b"valid_image_data" * 100, "", False, "Empty question"),
    ]
    
    for i, (image_data, question, expected_safe, description) in enumerate(test_cases, 1):
        print(f"\nTest {i}: {description}")
        is_safe, message, incidents = validator.validate_request(image_data, question)
        
        status = "✅ SAFE" if is_safe else "🚫 BLOCKED"
        print(f"  Result: {status}")
        print(f"  Message: {message}")
        
        if incidents:
            print(f"  Security incidents: {len(incidents)}")
            for incident in incidents:
                print(f"    - {incident.threat_level.value.upper()}: {incident.description}")
        
        if is_safe == expected_safe:
            print("  ✅ Test result matches expectation")
        else:
            print("  ❌ Test result does not match expectation")
    
    # Show security status
    print(f"\nSecurity Status:")
    status = validator.get_security_status()
    for key, value in status.items():
        print(f"  {key}: {value}")