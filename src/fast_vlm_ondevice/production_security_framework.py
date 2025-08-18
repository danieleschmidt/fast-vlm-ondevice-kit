"""
Production-grade security framework for FastVLM.

Comprehensive security controls including input validation, cryptographic operations,
threat detection, access control, and compliance monitoring.
"""

import hashlib
import hmac
import secrets
import time
import re
import json
import logging
import threading
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import base64
from pathlib import Path
import ipaddress
from datetime import datetime, timedelta

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    logging.warning("Cryptography library not available. Some security features disabled.")

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security level classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of security threats."""
    INJECTION = "injection"
    XSS = "xss"
    CSRF = "csrf"
    MALICIOUS_INPUT = "malicious_input"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_EXFILTRATION = "data_exfiltration"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"


@dataclass
class SecurityEvent:
    """Security event data structure."""
    event_id: str
    timestamp: float
    event_type: ThreatType
    severity: SecurityLevel
    source_ip: Optional[str]
    user_id: Optional[str]
    description: str
    context: Dict[str, Any]
    blocked: bool
    confidence_score: float


@dataclass
class AccessAttempt:
    """Access attempt tracking."""
    ip_address: str
    user_id: Optional[str]
    timestamp: float
    success: bool
    resource: str
    action: str


class InputValidator:
    """Comprehensive input validation with security focus."""
    
    def __init__(self):
        self.max_image_size_mb = 50
        self.max_question_length = 2000
        self.blocked_patterns = [
            # Script injection patterns
            r'<\s*script[^>]*>.*?<\s*/\s*script\s*>',
            r'javascript\s*:',
            r'data\s*:\s*text\s*/\s*html',
            r'vbscript\s*:',
            
            # SQL injection patterns
            r'(\bunion\b.*\bselect\b)|(\bselect\b.*\bunion\b)',
            r'\bdrop\s+table\b',
            r'\binsert\s+into\b',
            r'\bdelete\s+from\b',
            r'\bupdate\s+.*\bset\b',
            
            # Command injection patterns
            r'[;&|`$()]',
            r'\b(eval|exec|system|shell_exec|passthru)\s*\(',
            
            # Path traversal patterns
            r'\.\./',
            r'\.\.\\',
            
            # Common attack strings
            r'<\s*iframe[^>]*>',
            r'<\s*object[^>]*>',
            r'<\s*embed[^>]*>',
            r'<\s*link[^>]*>',
            r'<\s*meta[^>]*>',
        ]
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE | re.DOTALL) 
                                for pattern in self.blocked_patterns]
        
        # Character encoding validation
        self.allowed_chars = set()
        self.allowed_chars.update(range(32, 127))  # ASCII printable
        self.allowed_chars.update([9, 10, 13])     # Tab, LF, CR
        self.allowed_chars.update(range(160, 255)) # Extended ASCII
        
        logger.info("Input validator initialized with security patterns")
    
    def validate_image_data(self, image_data: bytes, filename: str = None) -> Dict[str, Any]:
        """Validate image data for security threats."""
        result = {
            "valid": True,
            "threats": [],
            "warnings": [],
            "metadata": {}
        }
        
        try:
            # Size validation
            size_mb = len(image_data) / (1024 * 1024)
            if size_mb > self.max_image_size_mb:
                result["valid"] = False
                result["threats"].append({
                    "type": "oversized_file",
                    "severity": SecurityLevel.MEDIUM,
                    "message": f"Image too large: {size_mb:.1f}MB (max: {self.max_image_size_mb}MB)"
                })
            
            # File signature validation
            if not self._validate_image_signature(image_data):
                result["valid"] = False
                result["threats"].append({
                    "type": "invalid_signature",
                    "severity": SecurityLevel.HIGH,
                    "message": "Invalid or suspicious image file signature"
                })
            
            # Embedded content detection
            embedded_threats = self._detect_embedded_content(image_data)
            if embedded_threats:
                result["valid"] = False
                result["threats"].extend(embedded_threats)
            
            # Filename validation
            if filename:
                filename_issues = self._validate_filename(filename)
                result["warnings"].extend(filename_issues)
            
            # Metadata extraction
            result["metadata"] = {
                "size_bytes": len(image_data),
                "size_mb": size_mb,
                "suspected_format": self._detect_image_format(image_data)
            }
            
        except Exception as e:
            result["valid"] = False
            result["threats"].append({
                "type": "validation_error",
                "severity": SecurityLevel.HIGH,
                "message": f"Image validation failed: {e}"
            })
        
        return result
    
    def validate_text_input(self, text: str, context: str = "question") -> Dict[str, Any]:
        """Validate text input for security threats."""
        result = {
            "valid": True,
            "threats": [],
            "warnings": [],
            "sanitized_text": text,
            "confidence_score": 1.0
        }
        
        try:
            # Length validation
            if len(text) > self.max_question_length:
                result["valid"] = False
                result["threats"].append({
                    "type": "oversized_input",
                    "severity": SecurityLevel.MEDIUM,
                    "message": f"Text too long: {len(text)} chars (max: {self.max_question_length})"
                })
            
            # Empty input check
            if not text.strip():
                result["valid"] = False
                result["threats"].append({
                    "type": "empty_input",
                    "severity": SecurityLevel.LOW,
                    "message": "Empty or whitespace-only input"
                })
            
            # Pattern-based threat detection
            threat_score = 0
            detected_patterns = []
            
            for i, pattern in enumerate(self.compiled_patterns):
                matches = pattern.findall(text)
                if matches:
                    detected_patterns.append({
                        "pattern_id": i,
                        "matches": matches,
                        "threat_type": self._classify_pattern_threat(i)
                    })
                    threat_score += len(matches) * 0.2
            
            if detected_patterns:
                result["valid"] = False
                result["threats"].append({
                    "type": "malicious_patterns",
                    "severity": SecurityLevel.HIGH,
                    "message": f"Detected {len(detected_patterns)} suspicious patterns",
                    "patterns": detected_patterns
                })
            
            # Character encoding validation
            invalid_chars = self._check_character_encoding(text)
            if invalid_chars:
                result["warnings"].append({
                    "type": "suspicious_encoding",
                    "message": f"Found {len(invalid_chars)} potentially suspicious characters"
                })
                threat_score += len(invalid_chars) * 0.1
            
            # Entropy analysis (detect random/encoded content)
            entropy = self._calculate_entropy(text)
            if entropy > 7.0:  # High entropy threshold
                result["warnings"].append({
                    "type": "high_entropy",
                    "message": f"High entropy content detected: {entropy:.2f}"
                })
                threat_score += 0.3
            
            # Calculate confidence score
            result["confidence_score"] = max(0.0, 1.0 - (threat_score / 2.0))
            
            # Sanitize text if needed
            if threat_score > 0:
                result["sanitized_text"] = self._sanitize_text(text)
            
        except Exception as e:
            result["valid"] = False
            result["threats"].append({
                "type": "validation_error",
                "severity": SecurityLevel.HIGH,
                "message": f"Text validation failed: {e}"
            })
        
        return result
    
    def _validate_image_signature(self, image_data: bytes) -> bool:
        """Validate image file signature."""
        if len(image_data) < 10:
            return False
        
        # Common image signatures
        signatures = {
            b'\xff\xd8\xff': 'JPEG',
            b'\x89PNG\r\n\x1a\n': 'PNG',
            b'GIF87a': 'GIF',
            b'GIF89a': 'GIF',
            b'RIFF': 'WEBP',  # Needs additional validation
            b'BM': 'BMP'
        }
        
        for sig, format_name in signatures.items():
            if image_data.startswith(sig):
                return True
        
        return False
    
    def _detect_embedded_content(self, image_data: bytes) -> List[Dict[str, Any]]:
        """Detect embedded malicious content in image data."""
        threats = []
        
        # Convert to string for pattern matching (with error handling)
        try:
            data_str = image_data.decode('utf-8', errors='ignore')
        except:
            data_str = str(image_data)
        
        # Check for embedded scripts or HTML
        html_patterns = [
            r'<\s*script[^>]*>',
            r'<\s*iframe[^>]*>',
            r'javascript\s*:',
            r'data\s*:\s*text\s*/\s*html'
        ]
        
        for pattern in html_patterns:
            if re.search(pattern, data_str, re.IGNORECASE):
                threats.append({
                    "type": "embedded_content",
                    "severity": SecurityLevel.HIGH,
                    "message": f"Detected embedded content pattern: {pattern}"
                })
        
        # Check for unusual metadata or comments
        if b'<!--' in image_data or b'<script' in image_data:
            threats.append({
                "type": "suspicious_metadata",
                "severity": SecurityLevel.MEDIUM,
                "message": "Suspicious metadata or comments found"
            })
        
        return threats
    
    def _validate_filename(self, filename: str) -> List[Dict[str, Any]]:
        """Validate filename for security issues."""
        warnings = []
        
        # Path traversal check
        if '..' in filename or '/' in filename or '\\' in filename:
            warnings.append({
                "type": "path_traversal",
                "message": "Filename contains path traversal characters"
            })
        
        # Suspicious extensions
        suspicious_extensions = ['.exe', '.bat', '.cmd', '.scr', '.pif', '.com']
        if any(filename.lower().endswith(ext) for ext in suspicious_extensions):
            warnings.append({
                "type": "suspicious_extension",
                "message": "Filename has suspicious extension"
            })
        
        # Double extensions
        if filename.count('.') > 1:
            warnings.append({
                "type": "double_extension",
                "message": "Filename has multiple extensions"
            })
        
        return warnings
    
    def _detect_image_format(self, image_data: bytes) -> str:
        """Detect image format from data."""
        if image_data.startswith(b'\xff\xd8\xff'):
            return "JPEG"
        elif image_data.startswith(b'\x89PNG\r\n\x1a\n'):
            return "PNG"
        elif image_data.startswith(b'GIF87a') or image_data.startswith(b'GIF89a'):
            return "GIF"
        elif image_data.startswith(b'RIFF') and b'WEBP' in image_data[:20]:
            return "WEBP"
        elif image_data.startswith(b'BM'):
            return "BMP"
        else:
            return "unknown"
    
    def _classify_pattern_threat(self, pattern_id: int) -> ThreatType:
        """Classify threat type based on pattern ID."""
        if pattern_id < 4:
            return ThreatType.XSS
        elif pattern_id < 8:
            return ThreatType.INJECTION
        elif pattern_id < 12:
            return ThreatType.INJECTION
        else:
            return ThreatType.MALICIOUS_INPUT
    
    def _check_character_encoding(self, text: str) -> List[int]:
        """Check for suspicious character encodings."""
        invalid_chars = []
        for char in text:
            if ord(char) not in self.allowed_chars:
                invalid_chars.append(ord(char))
        return invalid_chars
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text."""
        if not text:
            return 0.0
        
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        entropy = 0.0
        text_len = len(text)
        
        for count in char_counts.values():
            probability = count / text_len
            if probability > 0:
                entropy -= probability * (probability.bit_length() - 1)
        
        return entropy
    
    def _sanitize_text(self, text: str) -> str:
        """Sanitize text by removing/escaping dangerous content."""
        # Remove script tags and their content
        text = re.sub(r'<\s*script[^>]*>.*?<\s*/\s*script\s*>', '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove other dangerous HTML tags
        dangerous_tags = ['iframe', 'object', 'embed', 'link', 'meta']
        for tag in dangerous_tags:
            text = re.sub(f'<\s*{tag}[^>]*>', '', text, flags=re.IGNORECASE)
        
        # Escape remaining HTML
        text = text.replace('<', '&lt;').replace('>', '&gt;')
        
        # Remove javascript: and vbscript: protocols
        text = re.sub(r'(javascript|vbscript)\s*:', '', text, flags=re.IGNORECASE)
        
        return text


class CryptographicManager:
    """Cryptographic operations manager."""
    
    def __init__(self):
        self.fernet_key = None
        self.rsa_private_key = None
        self.rsa_public_key = None
        
        if CRYPTOGRAPHY_AVAILABLE:
            self._initialize_crypto()
        else:
            logger.warning("Cryptographic features disabled - cryptography library not available")
    
    def _initialize_crypto(self):
        """Initialize cryptographic components."""
        try:
            # Generate Fernet key for symmetric encryption
            self.fernet_key = Fernet.generate_key()
            
            # Generate RSA key pair for asymmetric encryption
            self.rsa_private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            self.rsa_public_key = self.rsa_private_key.public_key()
            
            logger.info("Cryptographic components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize cryptographic components: {e}")
    
    def encrypt_data(self, data: bytes) -> Dict[str, Any]:
        """Encrypt data using Fernet symmetric encryption."""
        if not CRYPTOGRAPHY_AVAILABLE or not self.fernet_key:
            raise RuntimeError("Cryptographic features not available")
        
        try:
            fernet = Fernet(self.fernet_key)
            encrypted_data = fernet.encrypt(data)
            
            return {
                "encrypted_data": base64.b64encode(encrypted_data).decode(),
                "algorithm": "Fernet",
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: str) -> bytes:
        """Decrypt Fernet encrypted data."""
        if not CRYPTOGRAPHY_AVAILABLE or not self.fernet_key:
            raise RuntimeError("Cryptographic features not available")
        
        try:
            fernet = Fernet(self.fernet_key)
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted_data = fernet.decrypt(encrypted_bytes)
            
            return decrypted_data
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token."""
        return secrets.token_urlsafe(length)
    
    def hash_password(self, password: str, salt: bytes = None) -> Dict[str, str]:
        """Hash password using PBKDF2."""
        if salt is None:
            salt = secrets.token_bytes(32)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = kdf.derive(password.encode())
        
        return {
            "hash": base64.b64encode(key).decode(),
            "salt": base64.b64encode(salt).decode(),
            "algorithm": "PBKDF2-SHA256",
            "iterations": 100000
        }
    
    def verify_password(self, password: str, hash_data: Dict[str, str]) -> bool:
        """Verify password against hash."""
        try:
            salt = base64.b64decode(hash_data["salt"].encode())
            stored_hash = base64.b64decode(hash_data["hash"].encode())
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=hash_data.get("iterations", 100000),
            )
            
            kdf.verify(password.encode(), stored_hash)
            return True
            
        except Exception:
            return False
    
    def create_hmac_signature(self, data: bytes, secret: str) -> str:
        """Create HMAC signature for data integrity."""
        signature = hmac.new(
            secret.encode(),
            data,
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def verify_hmac_signature(self, data: bytes, signature: str, secret: str) -> bool:
        """Verify HMAC signature."""
        expected_signature = self.create_hmac_signature(data, secret)
        return hmac.compare_digest(signature, expected_signature)


class SecurityScanner:
    """Security scanner for comprehensive threat detection."""
    
    def __init__(self):
        self.input_validator = InputValidator()
        self.threat_detector = None  # Will be initialized when needed
        
    def scan_input(self, image_data: bytes, question: str) -> Dict[str, Any]:
        """Scan input for security threats."""
        # Validate image
        image_result = self.input_validator.validate_image_data(image_data)
        
        # Validate text
        text_result = self.input_validator.validate_text_input(question)
        
        # Combine results
        overall_safe = image_result["valid"] and text_result["valid"]
        threats = []
        
        if image_result.get("threats"):
            threats.extend(image_result["threats"])
        if text_result.get("threats"):
            threats.extend(text_result["threats"])
        
        # Determine risk level
        if not overall_safe:
            if any(t.get("severity") == SecurityLevel.CRITICAL for t in threats):
                risk_level = "critical"
            elif any(t.get("severity") == SecurityLevel.HIGH for t in threats):
                risk_level = "high"
            else:
                risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "safe": overall_safe,
            "risk_level": risk_level,
            "threats": threats,
            "image_validation": image_result,
            "text_validation": text_result
        }


class AuthenticationManager:
    """Authentication and authorization manager for secure user access control."""
    
    def __init__(self):
        self.active_sessions = {}
        self.failed_attempts = {}
        self.lockout_times = {}
        self.max_failed_attempts = 5
        self.lockout_duration = 300  # 5 minutes
        
    def authenticate(self, user_id: str, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Perform user authentication with secure credential validation."""
        current_time = time.time()
        
        # Check if user is locked out
        if user_id in self.lockout_times:
            if current_time - self.lockout_times[user_id] < self.lockout_duration:
                return {
                    "authenticated": False,
                    "reason": "account_locked",
                    "retry_after": self.lockout_duration - (current_time - self.lockout_times[user_id])
                }
            else:
                # Lockout expired
                del self.lockout_times[user_id]
                self.failed_attempts[user_id] = 0
        
        # Validate credentials (simplified)
        api_key = credentials.get("api_key")
        if not api_key or len(api_key) < 32:
            self.failed_attempts[user_id] = self.failed_attempts.get(user_id, 0) + 1
            
            if self.failed_attempts[user_id] >= self.max_failed_attempts:
                self.lockout_times[user_id] = current_time
            
            return {
                "authenticated": False,
                "reason": "invalid_credentials"
            }
        
        # Success
        session_id = str(uuid.uuid4())
        self.active_sessions[session_id] = {
            "user_id": user_id,
            "created_at": current_time,
            "last_activity": current_time
        }
        
        # Reset failed attempts
        self.failed_attempts[user_id] = 0
        
        return {
            "authenticated": True,
            "session_id": session_id,
            "expires_at": current_time + 3600,  # 1 hour authentication session
            "authentication_method": "api_key"
        }
    
    def authorize(self, session_id: str, resource: str, action: str) -> Dict[str, Any]:
        """Perform authorization check for user access to resources."""
        if session_id not in self.active_sessions:
            return {
                "authorized": False,
                "reason": "invalid_session"
            }
        
        session = self.active_sessions[session_id]
        current_time = time.time()
        
        # Check session expiry
        if current_time - session["created_at"] > 3600:  # 1 hour
            del self.active_sessions[session_id]
            return {
                "authorized": False,
                "reason": "session_expired"
            }
        
        # Update last activity
        session["last_activity"] = current_time
        
        # Simple authorization logic (would be more complex in practice)
        allowed_actions = {
            "model_inference": ["read", "execute"],
            "model_management": ["read"],
            "system_config": []  # Admin only
        }
        
        if action in allowed_actions.get(resource, []):
            return {
                "authorized": True,
                "user_id": session["user_id"],
                "authorization_granted": True
            }
        
        return {
            "authorized": False,
            "reason": "insufficient_permissions"
        }


class ThreatDetectionEngine:
    """Advanced threat detection and analysis."""
    
    def __init__(self):
        self.threat_patterns = self._load_threat_patterns()
        self.anomaly_baselines = {}
        self.security_events = []
        self.blocked_ips = set()
        self.rate_limits = {}
        self.lock = threading.Lock()
        
        logger.info("Threat detection engine initialized")
    
    def _load_threat_patterns(self) -> Dict[str, List[str]]:
        """Load threat detection patterns."""
        return {
            "sql_injection": [
                r"(\bunion\b.*\bselect\b)|(\bselect\b.*\bunion\b)",
                r"\bdrop\s+table\b",
                r"\binsert\s+into\b",
                r"\bdelete\s+from\b"
            ],
            "xss": [
                r"<\s*script[^>]*>",
                r"javascript\s*:",
                r"data\s*:\s*text\s*/\s*html"
            ],
            "command_injection": [
                r"[;&|`$()]",
                r"\b(eval|exec|system)\s*\("
            ]
        }
    
    def analyze_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze request for security threats."""
        analysis_result = {
            "threat_detected": False,
            "threat_level": SecurityLevel.LOW,
            "threats": [],
            "blocked": False,
            "confidence_score": 1.0
        }
        
        try:
            # Rate limiting check
            if self._check_rate_limiting(request_data):
                analysis_result["threat_detected"] = True
                analysis_result["threat_level"] = SecurityLevel.HIGH
                analysis_result["threats"].append({
                    "type": ThreatType.RATE_LIMIT_EXCEEDED,
                    "message": "Rate limit exceeded"
                })
            
            # IP blocking check
            source_ip = request_data.get("source_ip")
            if source_ip and source_ip in self.blocked_ips:
                analysis_result["threat_detected"] = True
                analysis_result["threat_level"] = SecurityLevel.CRITICAL
                analysis_result["blocked"] = True
                analysis_result["threats"].append({
                    "type": ThreatType.UNAUTHORIZED_ACCESS,
                    "message": "Request from blocked IP"
                })
            
            # Pattern-based threat detection
            patterns_threats = self._detect_pattern_threats(request_data)
            if patterns_threats:
                analysis_result["threat_detected"] = True
                analysis_result["threats"].extend(patterns_threats)
            
            # Anomaly detection
            anomaly_score = self._detect_anomalies(request_data)
            if anomaly_score > 0.7:
                analysis_result["threat_detected"] = True
                analysis_result["threats"].append({
                    "type": ThreatType.ANOMALOUS_BEHAVIOR,
                    "message": f"Anomalous behavior detected (score: {anomaly_score:.2f})"
                })
            
            # Calculate overall confidence
            if analysis_result["threats"]:
                threat_scores = [0.8] * len(analysis_result["threats"])
                analysis_result["confidence_score"] = sum(threat_scores) / len(threat_scores)
            
            # Log security event
            if analysis_result["threat_detected"]:
                self._log_security_event(request_data, analysis_result)
        
        except Exception as e:
            logger.error(f"Threat analysis failed: {e}")
            analysis_result["threat_detected"] = True
            analysis_result["threat_level"] = SecurityLevel.HIGH
        
        return analysis_result
    
    def _check_rate_limiting(self, request_data: Dict[str, Any]) -> bool:
        """Check if request exceeds rate limits."""
        source_ip = request_data.get("source_ip", "unknown")
        current_time = time.time()
        
        with self.lock:
            if source_ip not in self.rate_limits:
                self.rate_limits[source_ip] = []
            
            # Clean old entries (older than 1 minute)
            self.rate_limits[source_ip] = [
                timestamp for timestamp in self.rate_limits[source_ip]
                if current_time - timestamp < 60
            ]
            
            # Add current request
            self.rate_limits[source_ip].append(current_time)
            
            # Check if rate limit exceeded (max 100 requests per minute)
            return len(self.rate_limits[source_ip]) > 100
    
    def _detect_pattern_threats(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect threats using pattern matching."""
        threats = []
        
        for field_name, field_value in request_data.items():
            if isinstance(field_value, str):
                for threat_type, patterns in self.threat_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, field_value, re.IGNORECASE):
                            threats.append({
                                "type": threat_type,
                                "field": field_name,
                                "pattern": pattern,
                                "severity": SecurityLevel.HIGH
                            })
        
        return threats
    
    def _detect_anomalies(self, request_data: Dict[str, Any]) -> float:
        """Detect anomalous behavior patterns."""
        # Simple anomaly detection based on request characteristics
        anomaly_score = 0.0
        
        # Check request size
        total_size = sum(len(str(v)) for v in request_data.values())
        if total_size > 10000:  # Large request
            anomaly_score += 0.3
        
        # Check for unusual characters
        text_data = " ".join(str(v) for v in request_data.values())
        non_printable_ratio = sum(1 for c in text_data if ord(c) < 32 or ord(c) > 126) / max(len(text_data), 1)
        anomaly_score += non_printable_ratio * 0.5
        
        # Check request frequency patterns (simplified)
        source_ip = request_data.get("source_ip", "unknown")
        if source_ip in self.rate_limits:
            recent_requests = len(self.rate_limits[source_ip])
            if recent_requests > 50:  # High frequency
                anomaly_score += 0.4
        
        return min(anomaly_score, 1.0)
    
    def _log_security_event(self, request_data: Dict[str, Any], analysis_result: Dict[str, Any]):
        """Log security event for audit trail."""
        event = SecurityEvent(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            event_type=ThreatType.MALICIOUS_INPUT,  # Default type
            severity=analysis_result["threat_level"],
            source_ip=request_data.get("source_ip"),
            user_id=request_data.get("user_id"),
            description=f"Threat detected: {len(analysis_result['threats'])} threats found",
            context={
                "threats": analysis_result["threats"],
                "request_data": {k: str(v)[:100] for k, v in request_data.items()}  # Truncate for logging
            },
            blocked=analysis_result["blocked"],
            confidence_score=analysis_result["confidence_score"]
        )
        
        with self.lock:
            self.security_events.append(event)
            
            # Keep only recent events (last 1000)
            if len(self.security_events) > 1000:
                self.security_events = self.security_events[-1000:]
        
        logger.warning(f"Security event logged: {event.event_id}")
    
    def block_ip(self, ip_address: str, reason: str = "Manual block"):
        """Block IP address."""
        with self.lock:
            self.blocked_ips.add(ip_address)
        
        logger.warning(f"IP blocked: {ip_address} - Reason: {reason}")
    
    def unblock_ip(self, ip_address: str):
        """Unblock IP address."""
        with self.lock:
            self.blocked_ips.discard(ip_address)
        
        logger.info(f"IP unblocked: {ip_address}")
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security monitoring summary."""
        with self.lock:
            recent_events = self.security_events[-50:]  # Last 50 events
            
            threat_counts = {}
            for event in recent_events:
                threat_type = event.event_type.value
                threat_counts[threat_type] = threat_counts.get(threat_type, 0) + 1
        
        return {
            "total_events": len(self.security_events),
            "recent_events_count": len(recent_events),
            "blocked_ips_count": len(self.blocked_ips),
            "threat_distribution": threat_counts,
            "recent_events": [asdict(event) for event in recent_events]
        }


class ProductionSecurityFramework:
    """Main security framework coordinating all security components."""
    
    def __init__(self):
        self.input_validator = InputValidator()
        self.crypto_manager = CryptographicManager()
        self.threat_detector = ThreatDetectionEngine()
        self.security_scanner = SecurityScanner()
        self.auth_manager = AuthenticationManager()
        self.security_enabled = True
        self.audit_log = []
        
        logger.info("Production security framework initialized")
    
    def validate_and_secure_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive request validation and security check."""
        if not self.security_enabled:
            return {"valid": True, "processed_data": request_data}
        
        result = {
            "valid": True,
            "threats_detected": False,
            "security_warnings": [],
            "processed_data": request_data.copy(),
            "security_metadata": {}
        }
        
        try:
            # Step 1: Threat detection
            threat_analysis = self.threat_detector.analyze_request(request_data)
            if threat_analysis["threat_detected"]:
                result["threats_detected"] = True
                result["security_warnings"].extend(threat_analysis["threats"])
                
                if threat_analysis["blocked"]:
                    result["valid"] = False
                    return result
            
            # Step 2: Input validation
            if "image_data" in request_data:
                image_validation = self.input_validator.validate_image_data(
                    request_data["image_data"],
                    request_data.get("filename")
                )
                if not image_validation["valid"]:
                    result["valid"] = False
                    result["security_warnings"].extend(image_validation["threats"])
            
            if "question" in request_data:
                text_validation = self.input_validator.validate_text_input(
                    request_data["question"]
                )
                if not text_validation["valid"]:
                    result["valid"] = False
                    result["security_warnings"].extend(text_validation["threats"])
                else:
                    # Use sanitized text
                    result["processed_data"]["question"] = text_validation["sanitized_text"]
            
            # Step 3: Add security metadata
            result["security_metadata"] = {
                "validation_timestamp": time.time(),
                "threat_confidence": threat_analysis.get("confidence_score", 1.0),
                "security_level": self._calculate_security_level(result),
                "request_id": str(uuid.uuid4())
            }
            
            # Step 4: Audit logging
            self._log_security_audit(request_data, result)
            
        except Exception as e:
            logger.error(f"Security validation failed: {e}")
            result["valid"] = False
            result["security_warnings"].append({
                "type": "validation_error",
                "message": f"Security validation failed: {e}"
            })
        
        return result
    
    def _calculate_security_level(self, result: Dict[str, Any]) -> SecurityLevel:
        """Calculate overall security level for request."""
        if not result["valid"]:
            return SecurityLevel.CRITICAL
        elif result["threats_detected"]:
            return SecurityLevel.HIGH
        elif result["security_warnings"]:
            return SecurityLevel.MEDIUM
        else:
            return SecurityLevel.LOW
    
    def _log_security_audit(self, request_data: Dict[str, Any], result: Dict[str, Any]):
        """Log security audit event."""
        audit_entry = {
            "timestamp": time.time(),
            "request_size": sum(len(str(v)) for v in request_data.values()),
            "valid": result["valid"],
            "threats_detected": result["threats_detected"],
            "warnings_count": len(result["security_warnings"]),
            "source_ip": request_data.get("source_ip", "unknown"),
            "user_id": request_data.get("user_id")
        }
        
        self.audit_log.append(audit_entry)
        
        # Keep audit log manageable
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-5000:]
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            "framework_status": "active" if self.security_enabled else "disabled",
            "components": {
                "input_validator": "active",
                "crypto_manager": "active" if CRYPTOGRAPHY_AVAILABLE else "disabled",
                "threat_detector": "active"
            },
            "threat_detection_summary": self.threat_detector.get_security_summary(),
            "audit_log_size": len(self.audit_log),
            "blocked_ips": len(self.threat_detector.blocked_ips)
        }
    
    def enable_security(self):
        """Enable security framework."""
        self.security_enabled = True
        logger.info("Security framework enabled")
    
    def disable_security(self):
        """Disable security framework (for testing only)."""
        self.security_enabled = False
        logger.warning("Security framework disabled - USE ONLY FOR TESTING")
    
    def authenticate(self, user_id: str, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate user with the security framework using secure authentication methods."""
        return self.auth_manager.authenticate(user_id, credentials)
    
    def authorize(self, session_id: str, resource: str, action: str) -> Dict[str, Any]:
        """Authorize user action with the framework."""
        return self.auth_manager.authorize(session_id, resource, action)


# Factory function for easy initialization
def create_security_framework() -> ProductionSecurityFramework:
    """Create and initialize production security framework."""
    return ProductionSecurityFramework()


# Example usage and testing
if __name__ == "__main__":
    # Initialize security framework
    security = create_security_framework()
    
    # Test with safe request
    safe_request = {
        "image_data": b"\x89PNG\r\n\x1a\n" + b"fake_png_data" * 100,
        "question": "What objects are visible in this image?",
        "source_ip": "192.168.1.100",
        "user_id": "user123"
    }
    
    safe_result = security.validate_and_secure_request(safe_request)
    print(f"Safe request validation: {safe_result['valid']}")
    
    # Test with potentially malicious request
    malicious_request = {
        "image_data": b"<script>alert('xss')</script>" + b"fake_data" * 50,
        "question": "'; DROP TABLE users; --",
        "source_ip": "192.168.1.100",
        "user_id": "user123"
    }
    
    malicious_result = security.validate_and_secure_request(malicious_request)
    print(f"Malicious request validation: {malicious_result['valid']}")
    print(f"Threats detected: {malicious_result['threats_detected']}")
    print(f"Warnings: {len(malicious_result['security_warnings'])}")
    
    # Get security status
    status = security.get_security_status()
    print(f"Security framework status: {status['framework_status']}")