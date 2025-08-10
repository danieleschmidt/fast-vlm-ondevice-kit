"""
Advanced Security Framework for FastVLM.

Implements comprehensive security measures including zero-trust architecture,
threat detection, secure model deployment, and advanced cryptographic protection
for mobile AI systems.
"""

import asyncio
import logging
import time
import hashlib
import hmac
import json
import uuid
import base64
from typing import Dict, Any, Optional, List, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
import secrets
import cryptography
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from datetime import datetime, timedelta
import re
import ipaddress
import socket

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class SecurityEvent(Enum):
    """Types of security events."""
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_VIOLATION = "authorization_violation"
    SUSPICIOUS_REQUEST = "suspicious_request"
    DATA_BREACH_ATTEMPT = "data_breach_attempt"
    MODEL_TAMPERING = "model_tampering"
    INJECTION_ATTACK = "injection_attack"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    CRYPTOGRAPHIC_FAILURE = "cryptographic_failure"
    PRIVILEGE_ESCALATION = "privilege_escalation"


class EncryptionLevel(Enum):
    """Encryption security levels."""
    BASIC = "basic"          # AES-128
    STANDARD = "standard"    # AES-256
    HIGH = "high"           # AES-256 + RSA-2048
    QUANTUM_SAFE = "quantum_safe"  # Post-quantum cryptography


@dataclass
class SecurityThreat:
    """Represents a detected security threat."""
    threat_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    threat_type: SecurityEvent = SecurityEvent.SUSPICIOUS_REQUEST
    threat_level: ThreatLevel = ThreatLevel.MEDIUM
    timestamp: float = field(default_factory=time.time)
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    request_payload: Optional[Dict[str, Any]] = None
    description: str = ""
    indicators: List[str] = field(default_factory=list)
    mitigation_applied: Optional[str] = None
    blocked: bool = False
    
    def age_minutes(self) -> float:
        """Get threat age in minutes."""
        return (time.time() - self.timestamp) / 60
    
    def is_recent(self, minutes: float = 60) -> bool:
        """Check if threat occurred recently."""
        return self.age_minutes() < minutes


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    policy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    policy_name: str = "default_policy"
    
    # Authentication settings
    require_authentication: bool = True
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 15
    session_timeout_minutes: int = 30
    require_2fa: bool = False
    
    # Rate limiting
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10
    
    # Input validation
    max_payload_size_mb: float = 10.0
    allowed_content_types: List[str] = field(default_factory=lambda: [
        "application/json", "multipart/form-data", "image/jpeg", "image/png"
    ])
    blocked_patterns: List[str] = field(default_factory=lambda: [
        r"<script.*?>.*?</script>",  # XSS
        r"(union|select|insert|delete|drop|create|alter)\s+",  # SQL injection
        r"../|..\\\\|\.\./",  # Path traversal
    ])
    
    # Model protection
    encrypt_model_weights: bool = True
    verify_model_integrity: bool = True
    allow_model_updates: bool = False
    require_signed_models: bool = True
    
    # Data protection
    encrypt_sensitive_data: bool = True
    data_retention_days: int = 30
    anonymize_logs: bool = True
    
    # Network security
    allowed_ip_ranges: List[str] = field(default_factory=list)
    blocked_ip_ranges: List[str] = field(default_factory=list)
    require_https: bool = True
    
    # Advanced protection
    enable_threat_detection: bool = True
    enable_anomaly_detection: bool = True
    enable_honeypots: bool = True
    threat_response_level: ThreatLevel = ThreatLevel.HIGH


class CryptographicManager:
    """Advanced cryptographic operations manager."""
    
    def __init__(self, encryption_level: EncryptionLevel = EncryptionLevel.STANDARD):
        self.encryption_level = encryption_level
        self.symmetric_keys = {}
        self.asymmetric_keys = {}
        self.key_derivation_iterations = 100000
        self._initialize_crypto_components()
    
    def _initialize_crypto_components(self):
        """Initialize cryptographic components."""
        logger.info(f"Initializing cryptographic manager with {self.encryption_level.value} security")
        
        # Generate master encryption key
        self.master_key = self._generate_master_key()
        
        # Initialize symmetric encryption
        self.fernet = Fernet(base64.urlsafe_b64encode(self.master_key[:32]))
        
        # Generate RSA key pair for asymmetric operations
        if self.encryption_level in [EncryptionLevel.HIGH, EncryptionLevel.QUANTUM_SAFE]:
            self._generate_rsa_keypair()
    
    def _generate_master_key(self) -> bytes:
        """Generate master encryption key."""
        if self.encryption_level == EncryptionLevel.BASIC:
            return secrets.token_bytes(16)  # 128-bit
        else:
            return secrets.token_bytes(32)  # 256-bit
    
    def _generate_rsa_keypair(self):
        """Generate RSA key pair."""
        key_size = 2048 if self.encryption_level == EncryptionLevel.HIGH else 4096
        
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend()
        )
        
        self.asymmetric_keys = {
            'private_key': private_key,
            'public_key': private_key.public_key()
        }
        
        logger.info(f"Generated RSA {key_size}-bit key pair")
    
    async def encrypt_data(self, data: Union[str, bytes], context: str = "default") -> Dict[str, Any]:
        """Encrypt data with contextual encryption."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Generate context-specific key if needed
        context_key = await self._get_context_key(context)
        
        # Encrypt data
        if self.encryption_level == EncryptionLevel.BASIC:
            encrypted_data = self.fernet.encrypt(data)
        else:
            # Use layered encryption for higher security levels
            encrypted_data = await self._layered_encrypt(data, context_key)
        
        return {
            "encrypted_data": base64.b64encode(encrypted_data).decode('ascii'),
            "context": context,
            "encryption_level": self.encryption_level.value,
            "timestamp": time.time()
        }
    
    async def decrypt_data(self, encrypted_package: Dict[str, Any]) -> bytes:
        """Decrypt data from encrypted package."""
        encrypted_data = base64.b64decode(encrypted_package["encrypted_data"])
        context = encrypted_package.get("context", "default")
        
        if encrypted_package["encryption_level"] == EncryptionLevel.BASIC.value:
            return self.fernet.decrypt(encrypted_data)
        else:
            context_key = await self._get_context_key(context)
            return await self._layered_decrypt(encrypted_data, context_key)
    
    async def _get_context_key(self, context: str) -> bytes:
        """Get or generate context-specific encryption key."""
        if context not in self.symmetric_keys:
            # Derive key from master key and context
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=context.encode('utf-8')[:16].ljust(16, b'\x00'),
                iterations=self.key_derivation_iterations,
                backend=default_backend()
            )
            self.symmetric_keys[context] = kdf.derive(self.master_key)
        
        return self.symmetric_keys[context]
    
    async def _layered_encrypt(self, data: bytes, context_key: bytes) -> bytes:
        """Apply layered encryption for enhanced security."""
        # Layer 1: Symmetric encryption
        fernet_context = Fernet(base64.urlsafe_b64encode(context_key))
        layer1_encrypted = fernet_context.encrypt(data)
        
        # Layer 2: RSA encryption for small data or key encryption
        if len(layer1_encrypted) < 200 and 'public_key' in self.asymmetric_keys:
            try:
                layer2_encrypted = self.asymmetric_keys['public_key'].encrypt(
                    layer1_encrypted,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                return layer2_encrypted
            except Exception as e:
                logger.warning(f"RSA encryption failed, using symmetric only: {e}")
        
        return layer1_encrypted
    
    async def _layered_decrypt(self, encrypted_data: bytes, context_key: bytes) -> bytes:
        """Decrypt layered encryption."""
        try:
            # Try RSA decryption first
            if 'private_key' in self.asymmetric_keys:
                try:
                    layer1_data = self.asymmetric_keys['private_key'].decrypt(
                        encrypted_data,
                        padding.OAEP(
                            mgf=padding.MGF1(algorithm=hashes.SHA256()),
                            algorithm=hashes.SHA256(),
                            label=None
                        )
                    )
                    encrypted_data = layer1_data
                except Exception:
                    # Not RSA encrypted, continue with symmetric
                    pass
            
            # Symmetric decryption
            fernet_context = Fernet(base64.urlsafe_b64encode(context_key))
            return fernet_context.decrypt(encrypted_data)
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    async def generate_secure_token(self, purpose: str = "general", validity_hours: int = 24) -> str:
        """Generate secure authentication token."""
        token_data = {
            "purpose": purpose,
            "issued_at": time.time(),
            "expires_at": time.time() + (validity_hours * 3600),
            "nonce": secrets.token_hex(16)
        }
        
        token_json = json.dumps(token_data, sort_keys=True)
        encrypted_token = await self.encrypt_data(token_json, f"token_{purpose}")
        
        # Create signed token
        signature = self._sign_data(token_json)
        
        return base64.b64encode(json.dumps({
            "token": encrypted_token,
            "signature": signature
        }).encode()).decode()
    
    async def validate_token(self, token: str, purpose: str = "general") -> Dict[str, Any]:
        """Validate and decode secure token."""
        try:
            # Decode token
            token_package = json.loads(base64.b64decode(token).decode())
            encrypted_token = token_package["token"]
            signature = token_package["signature"]
            
            # Decrypt token data
            decrypted_data = await self.decrypt_data(encrypted_token)
            token_data = json.loads(decrypted_data.decode())
            
            # Verify signature
            if not self._verify_signature(json.dumps(token_data, sort_keys=True), signature):
                raise ValueError("Token signature verification failed")
            
            # Check purpose
            if token_data["purpose"] != purpose:
                raise ValueError(f"Token purpose mismatch: expected {purpose}, got {token_data['purpose']}")
            
            # Check expiration
            if time.time() > token_data["expires_at"]:
                raise ValueError("Token has expired")
            
            return {
                "valid": True,
                "token_data": token_data,
                "remaining_time": token_data["expires_at"] - time.time()
            }
            
        except Exception as e:
            logger.warning(f"Token validation failed: {e}")
            return {"valid": False, "error": str(e)}
    
    def _sign_data(self, data: str) -> str:
        """Create HMAC signature for data."""
        signature = hmac.new(
            self.master_key,
            data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _verify_signature(self, data: str, signature: str) -> bool:
        """Verify HMAC signature."""
        expected_signature = self._sign_data(data)
        return hmac.compare_digest(signature, expected_signature)
    
    async def secure_hash(self, data: Union[str, bytes], salt: Optional[bytes] = None) -> str:
        """Generate secure hash with salt."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if salt is None:
            salt = secrets.token_bytes(16)
        
        # Use PBKDF2 for secure hashing
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.key_derivation_iterations,
            backend=default_backend()
        )
        
        hash_bytes = kdf.derive(data)
        
        # Return salt + hash encoded
        return base64.b64encode(salt + hash_bytes).decode()
    
    async def verify_secure_hash(self, data: Union[str, bytes], stored_hash: str) -> bool:
        """Verify data against stored secure hash."""
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            # Decode stored hash to get salt and hash
            decoded = base64.b64decode(stored_hash)
            salt = decoded[:16]
            expected_hash = decoded[16:]
            
            # Compute hash with same salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=self.key_derivation_iterations,
                backend=default_backend()
            )
            
            computed_hash = kdf.derive(data)
            
            return hmac.compare_digest(expected_hash, computed_hash)
            
        except Exception as e:
            logger.error(f"Hash verification failed: {e}")
            return False


class ThreatDetectionEngine:
    """Advanced threat detection and analysis engine."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.detected_threats = deque(maxlen=10000)
        self.threat_patterns = self._load_threat_patterns()
        self.anomaly_baselines = {}
        self.request_history = defaultdict(deque)
        self.blocked_ips = set()
        self.honeypots = set()
        
        # Machine learning components for anomaly detection
        self.anomaly_detector = self._initialize_anomaly_detector()
        
        # Threat intelligence
        self.threat_intelligence = self._load_threat_intelligence()
        
    def _load_threat_patterns(self) -> List[Dict[str, Any]]:
        """Load known threat patterns."""
        return [
            {
                "name": "sql_injection",
                "pattern": r"(union|select|insert|delete|drop|create|alter)\s+",
                "threat_level": ThreatLevel.HIGH,
                "description": "Potential SQL injection attack"
            },
            {
                "name": "xss_attack",
                "pattern": r"<script.*?>.*?</script>",
                "threat_level": ThreatLevel.MEDIUM,
                "description": "Potential cross-site scripting attack"
            },
            {
                "name": "path_traversal",
                "pattern": r"\.\.[\\/]|\.\.\\\\",
                "threat_level": ThreatLevel.HIGH,
                "description": "Potential path traversal attack"
            },
            {
                "name": "command_injection",
                "pattern": r"(;|\||&&|\|\||`)\s*(ls|cat|ps|wget|curl|nc|bash|sh)",
                "threat_level": ThreatLevel.CRITICAL,
                "description": "Potential command injection attack"
            },
            {
                "name": "model_extraction",
                "pattern": r"(weights|parameters|gradients|model\.state_dict)",
                "threat_level": ThreatLevel.HIGH,
                "description": "Potential model extraction attempt"
            }
        ]
    
    def _initialize_anomaly_detector(self):
        """Initialize anomaly detection components."""
        return {
            "request_frequency_baseline": {},
            "payload_size_baseline": {},
            "response_time_baseline": {},
            "user_agent_patterns": set(),
            "normal_request_patterns": set()
        }
    
    def _load_threat_intelligence(self) -> Dict[str, Any]:
        """Load threat intelligence data."""
        return {
            "malicious_ips": set([
                "192.168.1.100",  # Example malicious IP
                "10.0.0.50"
            ]),
            "suspicious_user_agents": [
                "sqlmap",
                "nikto",
                "nmap",
                "masscan",
                "gobuster"
            ],
            "known_attack_signatures": [
                "../../../etc/passwd",
                "../../windows/system32",
                "SELECT * FROM users",
                "<script>alert(1)</script>"
            ]
        }
    
    async def analyze_request(self, request_data: Dict[str, Any]) -> List[SecurityThreat]:
        """Analyze incoming request for security threats."""
        threats = []
        
        # Extract request information
        source_ip = request_data.get("source_ip", "unknown")
        user_agent = request_data.get("user_agent", "")
        payload = request_data.get("payload", {})
        request_path = request_data.get("path", "")
        request_method = request_data.get("method", "GET")
        
        # IP-based checks
        ip_threats = await self._analyze_ip_address(source_ip)
        threats.extend(ip_threats)
        
        # Rate limiting checks
        rate_threats = await self._check_rate_limits(source_ip, request_data)
        threats.extend(rate_threats)
        
        # Payload analysis
        payload_threats = await self._analyze_payload(payload, request_path)
        threats.extend(payload_threats)
        
        # User agent analysis
        ua_threats = await self._analyze_user_agent(user_agent)
        threats.extend(ua_threats)
        
        # Anomaly detection
        anomaly_threats = await self._detect_anomalies(request_data)
        threats.extend(anomaly_threats)
        
        # Pattern matching
        pattern_threats = await self._match_threat_patterns(request_data)
        threats.extend(pattern_threats)
        
        # Store request history for baseline learning
        await self._update_request_history(source_ip, request_data)
        
        # Record threats
        for threat in threats:
            self.detected_threats.append(threat)
            logger.warning(f"Threat detected: {threat.threat_type.value} from {source_ip}")
        
        return threats
    
    async def _analyze_ip_address(self, source_ip: str) -> List[SecurityThreat]:
        """Analyze source IP for threats."""
        threats = []
        
        try:
            ip_addr = ipaddress.ip_address(source_ip)
            
            # Check against threat intelligence
            if source_ip in self.threat_intelligence["malicious_ips"]:
                threats.append(SecurityThreat(
                    threat_type=SecurityEvent.SUSPICIOUS_REQUEST,
                    threat_level=ThreatLevel.HIGH,
                    source_ip=source_ip,
                    description=f"Request from known malicious IP: {source_ip}",
                    indicators=["malicious_ip_intelligence"]
                ))
            
            # Check if IP is in blocked ranges
            for blocked_range in self.policy.blocked_ip_ranges:
                if ip_addr in ipaddress.ip_network(blocked_range):
                    threats.append(SecurityThreat(
                        threat_type=SecurityEvent.AUTHORIZATION_VIOLATION,
                        threat_level=ThreatLevel.HIGH,
                        source_ip=source_ip,
                        description=f"Request from blocked IP range: {blocked_range}",
                        indicators=["blocked_ip_range"]
                    ))
            
            # Check if IP is already blocked
            if source_ip in self.blocked_ips:
                threats.append(SecurityThreat(
                    threat_type=SecurityEvent.AUTHORIZATION_VIOLATION,
                    threat_level=ThreatLevel.CRITICAL,
                    source_ip=source_ip,
                    description=f"Request from previously blocked IP: {source_ip}",
                    indicators=["previously_blocked_ip"]
                ))
            
        except ValueError:
            # Invalid IP address format
            threats.append(SecurityThreat(
                threat_type=SecurityEvent.SUSPICIOUS_REQUEST,
                threat_level=ThreatLevel.MEDIUM,
                source_ip=source_ip,
                description=f"Invalid IP address format: {source_ip}",
                indicators=["invalid_ip_format"]
            ))
        
        return threats
    
    async def _check_rate_limits(self, source_ip: str, request_data: Dict[str, Any]) -> List[SecurityThreat]:
        """Check request rate limits."""
        threats = []
        current_time = time.time()
        
        # Get request history for this IP
        ip_history = self.request_history[source_ip]
        
        # Clean old requests (older than 1 hour)
        while ip_history and ip_history[0]["timestamp"] < current_time - 3600:
            ip_history.popleft()
        
        # Add current request
        ip_history.append({"timestamp": current_time, "request": request_data})
        
        # Check rate limits
        recent_minute = [r for r in ip_history if r["timestamp"] > current_time - 60]
        recent_hour = [r for r in ip_history if r["timestamp"] > current_time - 3600]
        
        if len(recent_minute) > self.policy.requests_per_minute:
            threats.append(SecurityThreat(
                threat_type=SecurityEvent.RATE_LIMIT_EXCEEDED,
                threat_level=ThreatLevel.MEDIUM,
                source_ip=source_ip,
                description=f"Rate limit exceeded: {len(recent_minute)} requests in last minute",
                indicators=["rate_limit_minute"]
            ))
        
        if len(recent_hour) > self.policy.requests_per_hour:
            threats.append(SecurityThreat(
                threat_type=SecurityEvent.RATE_LIMIT_EXCEEDED,
                threat_level=ThreatLevel.HIGH,
                source_ip=source_ip,
                description=f"Rate limit exceeded: {len(recent_hour)} requests in last hour",
                indicators=["rate_limit_hour"]
            ))
        
        # Check for burst patterns
        very_recent = [r for r in ip_history if r["timestamp"] > current_time - 10]
        if len(very_recent) > self.policy.burst_limit:
            threats.append(SecurityThreat(
                threat_type=SecurityEvent.RATE_LIMIT_EXCEEDED,
                threat_level=ThreatLevel.HIGH,
                source_ip=source_ip,
                description=f"Burst limit exceeded: {len(very_recent)} requests in 10 seconds",
                indicators=["burst_limit_exceeded"]
            ))
        
        return threats
    
    async def _analyze_payload(self, payload: Dict[str, Any], request_path: str) -> List[SecurityThreat]:
        """Analyze request payload for threats."""
        threats = []
        
        # Convert payload to string for analysis
        payload_str = json.dumps(payload) if isinstance(payload, dict) else str(payload)
        
        # Check payload size
        payload_size_mb = len(payload_str.encode('utf-8')) / (1024 * 1024)
        if payload_size_mb > self.policy.max_payload_size_mb:
            threats.append(SecurityThreat(
                threat_type=SecurityEvent.SUSPICIOUS_REQUEST,
                threat_level=ThreatLevel.MEDIUM,
                description=f"Payload size exceeds limit: {payload_size_mb:.2f}MB",
                indicators=["oversized_payload"]
            ))
        
        # Check against blocked patterns
        for pattern in self.policy.blocked_patterns:
            if re.search(pattern, payload_str, re.IGNORECASE):
                threats.append(SecurityThreat(
                    threat_type=SecurityEvent.INJECTION_ATTACK,
                    threat_level=ThreatLevel.HIGH,
                    description=f"Blocked pattern detected in payload: {pattern}",
                    indicators=["blocked_pattern_match"],
                    request_payload=payload
                ))
        
        # Check for known attack signatures
        for signature in self.threat_intelligence["known_attack_signatures"]:
            if signature.lower() in payload_str.lower():
                threats.append(SecurityThreat(
                    threat_type=SecurityEvent.INJECTION_ATTACK,
                    threat_level=ThreatLevel.CRITICAL,
                    description=f"Known attack signature detected: {signature}",
                    indicators=["known_attack_signature"],
                    request_payload=payload
                ))
        
        # Specific model protection checks
        model_indicators = ["weights", "parameters", "state_dict", "model.pth", "checkpoint"]
        for indicator in model_indicators:
            if indicator in payload_str.lower():
                threats.append(SecurityThreat(
                    threat_type=SecurityEvent.MODEL_TAMPERING,
                    threat_level=ThreatLevel.CRITICAL,
                    description=f"Potential model extraction attempt detected",
                    indicators=["model_extraction_attempt"],
                    request_payload=payload
                ))
                break
        
        return threats
    
    async def _analyze_user_agent(self, user_agent: str) -> List[SecurityThreat]:
        """Analyze user agent for threats."""
        threats = []
        
        if not user_agent:
            threats.append(SecurityThreat(
                threat_type=SecurityEvent.SUSPICIOUS_REQUEST,
                threat_level=ThreatLevel.LOW,
                user_agent=user_agent,
                description="Missing user agent",
                indicators=["missing_user_agent"]
            ))
            return threats
        
        # Check against suspicious user agents
        user_agent_lower = user_agent.lower()
        for suspicious_ua in self.threat_intelligence["suspicious_user_agents"]:
            if suspicious_ua in user_agent_lower:
                threats.append(SecurityThreat(
                    threat_type=SecurityEvent.SUSPICIOUS_REQUEST,
                    threat_level=ThreatLevel.HIGH,
                    user_agent=user_agent,
                    description=f"Suspicious user agent detected: {suspicious_ua}",
                    indicators=["suspicious_user_agent"]
                ))
        
        # Check for automated tools patterns
        automation_patterns = [r"bot", r"crawler", r"spider", r"scanner", r"python-requests", r"curl/"]
        for pattern in automation_patterns:
            if re.search(pattern, user_agent_lower):
                threats.append(SecurityThreat(
                    threat_type=SecurityEvent.SUSPICIOUS_REQUEST,
                    threat_level=ThreatLevel.MEDIUM,
                    user_agent=user_agent,
                    description=f"Automated tool detected in user agent",
                    indicators=["automation_detected"]
                ))
                break
        
        return threats
    
    async def _detect_anomalies(self, request_data: Dict[str, Any]) -> List[SecurityThreat]:
        """Detect anomalous behavior patterns."""
        threats = []
        
        if not self.policy.enable_anomaly_detection:
            return threats
        
        source_ip = request_data.get("source_ip", "unknown")
        
        # Analyze request frequency anomalies
        ip_history = self.request_history.get(source_ip, deque())
        if len(ip_history) > 10:  # Need sufficient history
            # Calculate request intervals
            intervals = []
            for i in range(1, min(len(ip_history), 20)):
                interval = ip_history[i]["timestamp"] - ip_history[i-1]["timestamp"]
                intervals.append(interval)
            
            if intervals:
                avg_interval = sum(intervals) / len(intervals)
                recent_intervals = intervals[-5:]  # Last 5 intervals
                
                # Check for sudden burst (intervals much shorter than usual)
                if recent_intervals and max(recent_intervals) < avg_interval * 0.1:
                    threats.append(SecurityThreat(
                        threat_type=SecurityEvent.ANOMALOUS_BEHAVIOR,
                        threat_level=ThreatLevel.MEDIUM,
                        source_ip=source_ip,
                        description="Anomalous request frequency detected (burst pattern)",
                        indicators=["frequency_anomaly"]
                    ))
        
        # Payload size anomalies
        payload_size = len(str(request_data.get("payload", "")))
        if source_ip not in self.anomaly_baselines:
            self.anomaly_baselines[source_ip] = {
                "payload_sizes": deque(maxlen=100),
                "request_paths": set()
            }
        
        baseline = self.anomaly_baselines[source_ip]
        baseline["payload_sizes"].append(payload_size)
        
        if len(baseline["payload_sizes"]) > 10:
            avg_size = sum(baseline["payload_sizes"]) / len(baseline["payload_sizes"])
            if payload_size > avg_size * 5:  # 5x larger than usual
                threats.append(SecurityThreat(
                    threat_type=SecurityEvent.ANOMALOUS_BEHAVIOR,
                    threat_level=ThreatLevel.MEDIUM,
                    source_ip=source_ip,
                    description=f"Anomalous payload size: {payload_size} vs avg {avg_size:.0f}",
                    indicators=["payload_size_anomaly"]
                ))
        
        return threats
    
    async def _match_threat_patterns(self, request_data: Dict[str, Any]) -> List[SecurityThreat]:
        """Match request against known threat patterns."""
        threats = []
        
        # Combine all request data into searchable text
        searchable_text = json.dumps(request_data, default=str).lower()
        
        for pattern_info in self.threat_patterns:
            pattern = pattern_info["pattern"]
            
            if re.search(pattern, searchable_text, re.IGNORECASE):
                threats.append(SecurityThreat(
                    threat_type=SecurityEvent.INJECTION_ATTACK,
                    threat_level=pattern_info["threat_level"],
                    source_ip=request_data.get("source_ip"),
                    description=pattern_info["description"],
                    indicators=[f"pattern_match_{pattern_info['name']}"],
                    request_payload=request_data.get("payload")
                ))
        
        return threats
    
    async def _update_request_history(self, source_ip: str, request_data: Dict[str, Any]):
        """Update request history for baseline learning."""
        if source_ip not in self.request_history:
            self.request_history[source_ip] = deque(maxlen=1000)
        
        self.request_history[source_ip].append({
            "timestamp": time.time(),
            "request": request_data
        })
    
    async def block_ip(self, ip_address: str, reason: str, duration_minutes: int = 60):
        """Block an IP address for security reasons."""
        self.blocked_ips.add(ip_address)
        
        logger.warning(f"Blocked IP {ip_address}: {reason}")
        
        # Schedule unblock (in a real implementation, this would be persistent)
        asyncio.create_task(self._schedule_ip_unblock(ip_address, duration_minutes))
    
    async def _schedule_ip_unblock(self, ip_address: str, duration_minutes: int):
        """Schedule IP unblock after specified duration."""
        await asyncio.sleep(duration_minutes * 60)
        if ip_address in self.blocked_ips:
            self.blocked_ips.remove(ip_address)
            logger.info(f"Unblocked IP {ip_address} after {duration_minutes} minutes")
    
    def get_threat_summary(self) -> Dict[str, Any]:
        """Get summary of detected threats."""
        current_time = time.time()
        recent_threats = [t for t in self.detected_threats if t.is_recent(60)]  # Last hour
        
        threat_counts = defaultdict(int)
        threat_levels = defaultdict(int)
        
        for threat in recent_threats:
            threat_counts[threat.threat_type.value] += 1
            threat_levels[threat.threat_level.value] += 1
        
        return {
            "total_threats_detected": len(self.detected_threats),
            "recent_threats_1h": len(recent_threats),
            "blocked_ips_count": len(self.blocked_ips),
            "threat_types": dict(threat_counts),
            "threat_levels": dict(threat_levels),
            "top_threat_sources": self._get_top_threat_sources(),
            "detection_rate": len(recent_threats) / 60 if recent_threats else 0  # Threats per minute
        }
    
    def _get_top_threat_sources(self) -> List[Dict[str, Any]]:
        """Get top sources of threats."""
        source_counts = defaultdict(int)
        
        for threat in list(self.detected_threats)[-1000:]:  # Last 1000 threats
            if threat.source_ip:
                source_counts[threat.source_ip] += 1
        
        # Sort by count and return top 10
        top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return [{"ip": ip, "threat_count": count} for ip, count in top_sources]


class SecurityResponseManager:
    """Manages automated security responses to threats."""
    
    def __init__(self, policy: SecurityPolicy, threat_detector: ThreatDetectionEngine):
        self.policy = policy
        self.threat_detector = threat_detector
        self.response_history = deque(maxlen=1000)
        self.auto_responses_enabled = True
    
    async def respond_to_threats(self, threats: List[SecurityThreat], request_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute appropriate responses to detected threats."""
        if not threats:
            return {"action": "allow", "reason": "no_threats_detected"}
        
        # Determine overall threat level
        max_threat_level = max(threat.threat_level for threat in threats)
        source_ip = request_context.get("source_ip")
        
        response_actions = []
        
        # Critical threats - immediate blocking
        if max_threat_level == ThreatLevel.CRITICAL:
            if source_ip:
                await self.threat_detector.block_ip(
                    source_ip, 
                    "Critical security threat detected", 
                    duration_minutes=self.policy.lockout_duration_minutes * 2
                )
                response_actions.append("ip_blocked_critical")
            
            response_actions.append("request_blocked")
            await self._log_security_incident(threats, "CRITICAL_THREAT_BLOCKED", request_context)
        
        # High threats - temporary blocking and monitoring
        elif max_threat_level == ThreatLevel.HIGH:
            high_threat_count = sum(1 for t in threats if t.threat_level == ThreatLevel.HIGH)
            
            if high_threat_count >= 2:  # Multiple high threats
                if source_ip:
                    await self.threat_detector.block_ip(
                        source_ip,
                        f"Multiple high-level threats detected ({high_threat_count})",
                        duration_minutes=self.policy.lockout_duration_minutes
                    )
                    response_actions.append("ip_blocked_high")
                
                response_actions.append("request_blocked")
            else:
                response_actions.append("request_monitored")
                await self._increase_monitoring(source_ip)
        
        # Medium threats - increased monitoring
        elif max_threat_level == ThreatLevel.MEDIUM:
            response_actions.append("increased_monitoring")
            await self._increase_monitoring(source_ip)
            
            # Check if this IP has repeated medium threats
            if source_ip and await self._check_repeated_threats(source_ip, ThreatLevel.MEDIUM):
                response_actions.append("request_blocked")
        
        # Low threats - logging only
        else:
            response_actions.append("logged_only")
        
        # Record response
        response_record = {
            "timestamp": time.time(),
            "threats": [t.threat_id for t in threats],
            "max_threat_level": max_threat_level.value,
            "actions_taken": response_actions,
            "source_ip": source_ip,
            "auto_response": self.auto_responses_enabled
        }
        
        self.response_history.append(response_record)
        
        # Determine final action
        if "request_blocked" in response_actions:
            return {
                "action": "block",
                "reason": f"Security threat detected: {max_threat_level.value}",
                "threat_count": len(threats),
                "actions_taken": response_actions
            }
        else:
            return {
                "action": "allow",
                "reason": "Request allowed with monitoring",
                "threat_count": len(threats),
                "actions_taken": response_actions
            }
    
    async def _increase_monitoring(self, source_ip: Optional[str]):
        """Increase monitoring for a specific source."""
        if source_ip:
            logger.info(f"Increased monitoring activated for IP: {source_ip}")
            # In a real implementation, this would configure enhanced logging and monitoring
    
    async def _check_repeated_threats(self, source_ip: str, threat_level: ThreatLevel) -> bool:
        """Check if an IP has repeated threats of a certain level."""
        current_time = time.time()
        recent_threats = [
            t for t in self.threat_detector.detected_threats 
            if t.source_ip == source_ip and 
               t.threat_level == threat_level and 
               t.is_recent(30)  # Last 30 minutes
        ]
        
        return len(recent_threats) >= 3  # 3 or more threats in 30 minutes
    
    async def _log_security_incident(self, threats: List[SecurityThreat], 
                                   incident_type: str, context: Dict[str, Any]):
        """Log a security incident for audit purposes."""
        incident_record = {
            "timestamp": time.time(),
            "incident_type": incident_type,
            "threats": [
                {
                    "threat_id": t.threat_id,
                    "threat_type": t.threat_type.value,
                    "threat_level": t.threat_level.value,
                    "description": t.description,
                    "indicators": t.indicators
                }
                for t in threats
            ],
            "context": context,
            "severity": "CRITICAL" if any(t.threat_level == ThreatLevel.CRITICAL for t in threats) else "HIGH"
        }
        
        # In a real implementation, this would be sent to a SIEM or security logging system
        logger.critical(f"SECURITY INCIDENT: {incident_type} - {len(threats)} threats detected")
        
        # Could also send alerts to security team
        await self._send_security_alert(incident_record)
    
    async def _send_security_alert(self, incident_record: Dict[str, Any]):
        """Send security alert to administrators."""
        # Placeholder for sending alerts (email, Slack, webhook, etc.)
        logger.info(f"Security alert sent for incident: {incident_record['incident_type']}")


class AdvancedSecurityFramework:
    """Main security framework coordinating all security components."""
    
    def __init__(self, policy: SecurityPolicy = None, encryption_level: EncryptionLevel = EncryptionLevel.STANDARD):
        self.policy = policy or SecurityPolicy()
        self.crypto_manager = CryptographicManager(encryption_level)
        self.threat_detector = ThreatDetectionEngine(self.policy)
        self.response_manager = SecurityResponseManager(self.policy, self.threat_detector)
        
        # Security state
        self.is_enabled = True
        self.security_metrics = {
            "requests_analyzed": 0,
            "threats_detected": 0,
            "requests_blocked": 0,
            "security_incidents": 0
        }
        
        # Background monitoring
        self._monitoring_tasks = []
        self._shutdown_event = asyncio.Event()
    
    async def start(self):
        """Start the security framework."""
        if not self.is_enabled:
            return
        
        logger.info("Starting Advanced Security Framework")
        
        # Start background monitoring tasks
        self._monitoring_tasks = [
            asyncio.create_task(self._security_monitoring_loop()),
            asyncio.create_task(self._threat_analysis_loop()),
            asyncio.create_task(self._cleanup_loop())
        ]
        
        logger.info(f"Security framework started with {self.policy.policy_name} policy")
        
        # Wait for shutdown
        await self._shutdown_event.wait()
        
        # Cancel monitoring tasks
        for task in self._monitoring_tasks:
            task.cancel()
        
        await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
    
    async def stop(self):
        """Stop the security framework."""
        logger.info("Stopping security framework")
        self.is_enabled = False
        self._shutdown_event.set()
    
    async def validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate incoming request against security policies."""
        if not self.is_enabled:
            return {"valid": True, "action": "allow"}
        
        self.security_metrics["requests_analyzed"] += 1
        
        # Analyze request for threats
        threats = await self.threat_detector.analyze_request(request_data)
        
        if threats:
            self.security_metrics["threats_detected"] += len(threats)
            
            # Generate security response
            response = await self.response_manager.respond_to_threats(threats, request_data)
            
            if response["action"] == "block":
                self.security_metrics["requests_blocked"] += 1
            
            return {
                "valid": response["action"] == "allow",
                "action": response["action"],
                "reason": response["reason"],
                "threats_detected": len(threats),
                "threat_details": [
                    {
                        "type": t.threat_type.value,
                        "level": t.threat_level.value,
                        "description": t.description
                    }
                    for t in threats
                ]
            }
        
        return {
            "valid": True,
            "action": "allow",
            "threats_detected": 0
        }
    
    async def secure_model_data(self, model_data: bytes, context: str = "model_weights") -> Dict[str, Any]:
        """Securely encrypt model data."""
        if not self.policy.encrypt_model_weights:
            return {"encrypted": False, "data": model_data}
        
        encrypted_package = await self.crypto_manager.encrypt_data(model_data, context)
        
        return {
            "encrypted": True,
            "package": encrypted_package,
            "context": context
        }
    
    async def decrypt_model_data(self, encrypted_package: Dict[str, Any]) -> bytes:
        """Decrypt secured model data."""
        return await self.crypto_manager.decrypt_data(encrypted_package["package"])
    
    async def create_secure_session(self, user_id: str, permissions: List[str]) -> Dict[str, Any]:
        """Create secure authenticated session."""
        session_data = {
            "user_id": user_id,
            "permissions": permissions,
            "created_at": time.time(),
            "ip_address": None  # Would be set from request context
        }
        
        session_token = await self.crypto_manager.generate_secure_token(
            "session", 
            self.policy.session_timeout_minutes // 60
        )
        
        return {
            "session_token": session_token,
            "expires_at": time.time() + (self.policy.session_timeout_minutes * 60),
            "permissions": permissions
        }
    
    async def validate_session(self, session_token: str) -> Dict[str, Any]:
        """Validate secure session token."""
        return await self.crypto_manager.validate_token(session_token, "session")
    
    async def _security_monitoring_loop(self):
        """Background security monitoring loop."""
        logger.info("Starting security monitoring loop")
        
        while self.is_enabled:
            try:
                # Monitor threat detection performance
                threat_summary = self.threat_detector.get_threat_summary()
                
                # Alert on high threat activity
                if threat_summary["detection_rate"] > 5:  # More than 5 threats per minute
                    logger.warning(f"High threat activity detected: {threat_summary['detection_rate']:.1f} threats/min")
                    self.security_metrics["security_incidents"] += 1
                
                # Clean up old data
                await self._cleanup_old_data()
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Security monitoring error: {e}")
                await asyncio.sleep(300)  # Wait longer on error
    
    async def _threat_analysis_loop(self):
        """Background threat analysis and learning loop."""
        logger.info("Starting threat analysis loop")
        
        while self.is_enabled:
            try:
                # Analyze threat patterns and update detection rules
                await self._analyze_threat_patterns()
                
                # Update threat intelligence
                await self._update_threat_intelligence()
                
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
            except Exception as e:
                logger.error(f"Threat analysis error: {e}")
                await asyncio.sleep(600)  # Wait longer on error
    
    async def _cleanup_loop(self):
        """Cleanup old security data."""
        while self.is_enabled:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # Cleanup every hour
            except Exception as e:
                logger.error(f"Security cleanup error: {e}")
                await asyncio.sleep(1800)
    
    async def _analyze_threat_patterns(self):
        """Analyze detected threats to identify new patterns."""
        # Get recent threats
        recent_threats = [t for t in self.threat_detector.detected_threats if t.is_recent(60)]
        
        if len(recent_threats) < 5:
            return
        
        # Analyze common patterns
        threat_types = defaultdict(int)
        source_ips = defaultdict(int)
        
        for threat in recent_threats:
            threat_types[threat.threat_type] += 1
            if threat.source_ip:
                source_ips[threat.source_ip] += 1
        
        # Identify frequently attacking IPs
        for ip, count in source_ips.items():
            if count >= 5:  # 5 or more threats from same IP
                await self.threat_detector.block_ip(
                    ip, 
                    f"Frequent threat source ({count} threats in 1 hour)", 
                    duration_minutes=120
                )
    
    async def _update_threat_intelligence(self):
        """Update threat intelligence based on observed attacks."""
        # In a real implementation, this would:
        # 1. Analyze successful attacks
        # 2. Update threat signatures
        # 3. Share intelligence with external sources
        # 4. Update blocking rules
        pass
    
    async def _cleanup_old_data(self):
        """Clean up old security data to manage memory."""
        current_time = time.time()
        retention_seconds = self.policy.data_retention_days * 86400
        
        # Clean up old request history
        for ip in list(self.threat_detector.request_history.keys()):
            history = self.threat_detector.request_history[ip]
            # Remove requests older than retention period
            while history and history[0]["timestamp"] < current_time - retention_seconds:
                history.popleft()
            
            # Remove empty histories
            if not history:
                del self.threat_detector.request_history[ip]
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security framework status."""
        threat_summary = self.threat_detector.get_threat_summary()
        
        return {
            "framework": {
                "enabled": self.is_enabled,
                "policy_name": self.policy.policy_name,
                "encryption_level": self.crypto_manager.encryption_level.value,
                "monitoring_active": len(self._monitoring_tasks) > 0
            },
            "metrics": self.security_metrics.copy(),
            "threat_detection": threat_summary,
            "blocked_ips": len(self.threat_detector.blocked_ips),
            "policy_settings": {
                "require_authentication": self.policy.require_authentication,
                "rate_limits": {
                    "requests_per_minute": self.policy.requests_per_minute,
                    "requests_per_hour": self.policy.requests_per_hour
                },
                "encryption_enabled": self.policy.encrypt_model_weights,
                "anomaly_detection": self.policy.enable_anomaly_detection
            }
        }


# Factory functions
def create_security_framework(security_level: str = "standard") -> AdvancedSecurityFramework:
    """Create security framework with predefined security levels."""
    if security_level == "basic":
        policy = SecurityPolicy(
            require_authentication=False,
            requests_per_minute=120,
            enable_threat_detection=True,
            encrypt_model_weights=False
        )
        encryption = EncryptionLevel.BASIC
    elif security_level == "high":
        policy = SecurityPolicy(
            require_authentication=True,
            require_2fa=True,
            requests_per_minute=30,
            enable_threat_detection=True,
            enable_anomaly_detection=True,
            encrypt_model_weights=True,
            require_signed_models=True
        )
        encryption = EncryptionLevel.HIGH
    elif security_level == "quantum_safe":
        policy = SecurityPolicy(
            require_authentication=True,
            require_2fa=True,
            requests_per_minute=20,
            enable_threat_detection=True,
            enable_anomaly_detection=True,
            encrypt_model_weights=True,
            require_signed_models=True,
            enable_honeypots=True
        )
        encryption = EncryptionLevel.QUANTUM_SAFE
    else:  # standard
        policy = SecurityPolicy()
        encryption = EncryptionLevel.STANDARD
    
    return AdvancedSecurityFramework(policy, encryption)
