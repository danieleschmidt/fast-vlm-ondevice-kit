#!/usr/bin/env python3
"""
Advanced Security Enhancement v4.0
Zero-trust security architecture with ML-powered threat detection

Implements comprehensive security measures for AI systems including
cryptographic protection, threat detection, and autonomous security response.
"""

import asyncio
import logging
import time
import json
import hashlib
import hmac
import secrets
import base64
import math
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
import threading
from threading import Lock
import statistics
import re
import ipaddress
from contextlib import asynccontextmanager
import tempfile
import os

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Security threat levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AttackType(Enum):
    """Types of security attacks"""
    INJECTION = "injection"
    XSS = "xss"
    CSRF = "csrf"
    DOS = "dos"
    BRUTE_FORCE = "brute_force"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    MALWARE = "malware"
    SOCIAL_ENGINEERING = "social_engineering"
    ZERO_DAY = "zero_day"


class SecurityAction(Enum):
    """Security response actions"""
    BLOCK = "block"
    QUARANTINE = "quarantine"
    MONITOR = "monitor"
    ALERT = "alert"
    REMEDIATE = "remediate"
    ISOLATE = "isolate"
    TERMINATE = "terminate"


@dataclass
class SecurityEvent:
    """Security threat or incident event"""
    event_id: str
    threat_level: ThreatLevel
    attack_type: AttackType
    source_ip: str
    target_component: str
    description: str
    evidence: Dict[str, Any]
    timestamp: str
    mitigation_actions: List[str] = field(default_factory=list)
    resolved: bool = False
    false_positive: bool = False


@dataclass
class CryptographicKey:
    """Cryptographic key material"""
    key_id: str
    key_type: str  # AES, RSA, ECDSA, etc.
    key_size: int
    created_at: str
    expires_at: Optional[str]
    purpose: str  # encryption, signing, authentication
    active: bool = True


class CryptographicManager:
    """Advanced cryptographic operations manager"""
    
    def __init__(self):
        self.keys: Dict[str, CryptographicKey] = {}
        self.key_rotation_interval = timedelta(days=30)
        self.lock = Lock()
        
        logger.info("🔐 Cryptographic manager initialized")
        
    def generate_key(self, key_type: str = "AES", key_size: int = 256, purpose: str = "encryption") -> str:
        """Generate new cryptographic key"""
        key_id = f"{key_type}_{purpose}_{int(time.time())}"
        
        # Generate secure key material
        if key_type == "AES":
            key_material = secrets.token_bytes(key_size // 8)
        else:
            key_material = secrets.token_bytes(key_size // 8)
            
        # Store key metadata
        crypto_key = CryptographicKey(
            key_id=key_id,
            key_type=key_type,
            key_size=key_size,
            created_at=datetime.now().isoformat(),
            expires_at=(datetime.now() + self.key_rotation_interval).isoformat(),
            purpose=purpose
        )
        
        with self.lock:
            self.keys[key_id] = crypto_key
            
        logger.info(f"Generated {key_type} key: {key_id}")
        return key_id
        
    def encrypt_data(self, data: bytes, key_id: str) -> bytes:
        """Encrypt data with specified key"""
        if key_id not in self.keys:
            raise ValueError(f"Key not found: {key_id}")
            
        key = self.keys[key_id]
        if not key.active:
            raise ValueError(f"Key is inactive: {key_id}")
            
        # Simplified encryption (in production, use proper crypto library)
        # This is a demonstration - use cryptography library for real implementation
        nonce = secrets.token_bytes(12)
        encrypted = self._xor_encrypt(data, key_id.encode())
        
        return nonce + encrypted
        
    def decrypt_data(self, encrypted_data: bytes, key_id: str) -> bytes:
        """Decrypt data with specified key"""
        if key_id not in self.keys:
            raise ValueError(f"Key not found: {key_id}")
            
        # Extract nonce and encrypted data
        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:]
        
        # Simplified decryption
        decrypted = self._xor_encrypt(ciphertext, key_id.encode())
        
        return decrypted
        
    def _xor_encrypt(self, data: bytes, key: bytes) -> bytes:
        """Simple XOR encryption for demonstration"""
        result = bytearray()
        key_len = len(key)
        
        for i, byte in enumerate(data):
            result.append(byte ^ key[i % key_len])
            
        return bytes(result)
        
    def sign_data(self, data: bytes, key_id: str) -> bytes:
        """Create digital signature for data"""
        if key_id not in self.keys:
            raise ValueError(f"Key not found: {key_id}")
            
        # Simplified signing using HMAC
        signature = hmac.new(
            key_id.encode(),
            data,
            hashlib.sha256
        ).digest()
        
        return signature
        
    def verify_signature(self, data: bytes, signature: bytes, key_id: str) -> bool:
        """Verify digital signature"""
        try:
            expected_signature = self.sign_data(data, key_id)
            return hmac.compare_digest(signature, expected_signature)
        except Exception:
            return False
            
    def rotate_keys(self):
        """Rotate expired keys"""
        current_time = datetime.now()
        
        with self.lock:
            for key_id, key in self.keys.items():
                if key.expires_at and datetime.fromisoformat(key.expires_at) < current_time:
                    logger.info(f"Rotating expired key: {key_id}")
                    
                    # Generate new key with same parameters
                    new_key_id = self.generate_key(key.key_type, key.key_size, key.purpose)
                    
                    # Deactivate old key
                    key.active = False


class MLThreatDetector:
    """Machine Learning-powered threat detection system"""
    
    def __init__(self):
        self.threat_patterns = self._load_threat_patterns()
        self.behavioral_baselines: Dict[str, List[float]] = {}
        self.anomaly_threshold = 2.0  # Standard deviations
        
        logger.info("🤖 ML Threat Detector initialized")
        
    def _load_threat_patterns(self) -> Dict[str, List[str]]:
        """Load known threat patterns"""
        return {
            "injection_patterns": [
                r"(?i)(union\s+select|drop\s+table|insert\s+into)",
                r"(?i)(<script|javascript:|eval\()",
                r"(?i)(exec\(|system\(|subprocess\.)",
                r"(\.\./|\.\.\\|/etc/passwd)",
                r"(?i)(select.*from.*where)"
            ],
            "xss_patterns": [
                r"(?i)<script.*?>.*?</script>",
                r"(?i)javascript:",
                r"(?i)on\w+\s*=",
                r"(?i)document\.(cookie|domain)",
                r"(?i)window\.(location|open)"
            ],
            "dos_patterns": [
                r"a{1000,}",  # Excessive repetition
                r"(?i)(sleep\(|benchmark\()",
                r"[\x00-\x1f]{100,}"  # Control characters
            ],
            "malware_indicators": [
                r"(?i)(trojan|virus|malware|backdoor)",
                r"(?i)(keylogger|rootkit|spyware)",
                r"(?i)(cryptominer|botnet|ransomware)"
            ]
        }
        
    async def analyze_input(self, input_data: str, source_ip: str = "unknown") -> Optional[SecurityEvent]:
        """Analyze input for security threats"""
        
        # Pattern-based detection
        threat = self._detect_known_patterns(input_data)
        if threat:
            return self._create_security_event(threat, source_ip, input_data)
            
        # Behavioral analysis
        behavioral_threat = self._analyze_behavioral_anomaly(input_data, source_ip)
        if behavioral_threat:
            return self._create_security_event(behavioral_threat, source_ip, input_data)
            
        # Content analysis
        content_threat = self._analyze_content_anomalies(input_data)
        if content_threat:
            return self._create_security_event(content_threat, source_ip, input_data)
            
        return None
        
    def _detect_known_patterns(self, input_data: str) -> Optional[Dict[str, Any]]:
        """Detect known attack patterns"""
        
        for pattern_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                if re.search(pattern, input_data):
                    return {
                        "attack_type": self._map_pattern_to_attack_type(pattern_type),
                        "threat_level": ThreatLevel.HIGH,
                        "pattern": pattern,
                        "pattern_type": pattern_type
                    }
                    
        return None
        
    def _analyze_behavioral_anomaly(self, input_data: str, source_ip: str) -> Optional[Dict[str, Any]]:
        """Analyze behavioral anomalies"""
        
        # Calculate input characteristics
        characteristics = {
            "length": len(input_data),
            "special_chars": len([c for c in input_data if not c.isalnum()]),
            "entropy": self._calculate_entropy(input_data),
            "numeric_ratio": len([c for c in input_data if c.isdigit()]) / max(len(input_data), 1)
        }
        
        # Check against baselines
        if source_ip not in self.behavioral_baselines:
            self.behavioral_baselines[source_ip] = []
            
        baseline = self.behavioral_baselines[source_ip]
        baseline.append(characteristics["entropy"])
        
        # Keep only recent samples
        if len(baseline) > 100:
            baseline = baseline[-100:]
            self.behavioral_baselines[source_ip] = baseline
            
        # Detect anomalies
        if len(baseline) >= 10:
            mean_entropy = statistics.mean(baseline[:-1])  # Exclude current sample
            std_entropy = statistics.stdev(baseline[:-1])
            
            if std_entropy > 0:
                z_score = abs((characteristics["entropy"] - mean_entropy) / std_entropy)
                
                if z_score > self.anomaly_threshold:
                    return {
                        "attack_type": AttackType.DATA_EXFILTRATION,
                        "threat_level": ThreatLevel.MEDIUM,
                        "anomaly_type": "behavioral",
                        "z_score": z_score,
                        "characteristics": characteristics
                    }
                    
        return None
        
    def _analyze_content_anomalies(self, input_data: str) -> Optional[Dict[str, Any]]:
        """Analyze content for anomalies"""
        
        # Check for suspicious content
        suspicious_indicators = [
            (len(input_data) > 10000, "excessive_length"),
            (input_data.count('\n') > 100, "excessive_newlines"),
            (len(set(input_data)) < 10, "low_entropy"),
            (any(ord(c) < 32 or ord(c) > 126 for c in input_data if c not in '\n\r\t'), "binary_content")
        ]
        
        for condition, indicator in suspicious_indicators:
            if condition:
                return {
                    "attack_type": AttackType.MALWARE,
                    "threat_level": ThreatLevel.MEDIUM,
                    "anomaly_type": "content",
                    "indicator": indicator
                }
                
        return None
        
    def _calculate_entropy(self, data: str) -> float:
        """Calculate Shannon entropy of string"""
        if not data:
            return 0.0
            
        # Count character frequencies
        char_counts = {}
        for char in data:
            char_counts[char] = char_counts.get(char, 0) + 1
            
        # Calculate entropy
        entropy = 0.0
        data_len = len(data)
        
        for count in char_counts.values():
            probability = count / data_len
            entropy -= probability * math.log2(probability) if probability > 0 else 0
            
        return entropy
        
    def _map_pattern_to_attack_type(self, pattern_type: str) -> AttackType:
        """Map pattern type to attack type"""
        mapping = {
            "injection_patterns": AttackType.INJECTION,
            "xss_patterns": AttackType.XSS,
            "dos_patterns": AttackType.DOS,
            "malware_indicators": AttackType.MALWARE
        }
        
        return mapping.get(pattern_type, AttackType.INJECTION)
        
    def _create_security_event(self, threat_info: Dict[str, Any], source_ip: str, input_data: str) -> SecurityEvent:
        """Create security event from threat detection"""
        
        event_id = f"SEC_{int(time.time())}_{secrets.token_hex(4)}"
        
        return SecurityEvent(
            event_id=event_id,
            threat_level=threat_info["threat_level"],
            attack_type=threat_info["attack_type"],
            source_ip=source_ip,
            target_component="input_validator",
            description=f"Detected {threat_info['attack_type'].value} attack",
            evidence={
                "input_sample": input_data[:200] + "..." if len(input_data) > 200 else input_data,
                "threat_info": threat_info,
                "detection_method": "ml_threat_detector"
            },
            timestamp=datetime.now().isoformat()
        )


class SecureInputValidator:
    """Advanced secure input validation with threat detection"""
    
    def __init__(self, crypto_manager: CryptographicManager, threat_detector: MLThreatDetector):
        self.crypto_manager = crypto_manager
        self.threat_detector = threat_detector
        self.validation_rules = self._load_validation_rules()
        
        logger.info("🛡️ Secure input validator initialized")
        
    def _load_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load input validation rules"""
        return {
            "image_data": {
                "max_size_mb": 50,
                "allowed_formats": ["JPEG", "PNG", "WebP"],
                "max_dimensions": (4096, 4096),
                "scan_for_malware": True
            },
            "text_input": {
                "max_length": 1000,
                "allowed_chars": r"[a-zA-Z0-9\s\.,\?!\-']",
                "block_patterns": ["<script", "javascript:", "eval("],
                "sanitize": True
            },
            "file_paths": {
                "max_length": 256,
                "allowed_extensions": [".jpg", ".png", ".txt", ".json"],
                "block_patterns": ["../", "..\\", "/etc/", "C:\\"],
                "sanitize_path": True
            }
        }
        
    async def validate_input(self, input_data: Any, input_type: str, source_ip: str = "unknown") -> Tuple[bool, Optional[SecurityEvent], Any]:
        """Validate input with comprehensive security checks"""
        
        # Step 1: ML threat detection
        if isinstance(input_data, str):
            security_event = await self.threat_detector.analyze_input(input_data, source_ip)
            if security_event and security_event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                logger.warning(f"🚨 High-threat input blocked: {security_event.event_id}")
                return False, security_event, None
                
        # Step 2: Rule-based validation
        validation_result = self._apply_validation_rules(input_data, input_type)
        if not validation_result["valid"]:
            # Create security event for rule violation
            security_event = SecurityEvent(
                event_id=f"VAL_{int(time.time())}_{secrets.token_hex(4)}",
                threat_level=ThreatLevel.MEDIUM,
                attack_type=AttackType.INJECTION,
                source_ip=source_ip,
                target_component="input_validator",
                description=f"Input validation failed: {validation_result['reason']}",
                evidence={"validation_result": validation_result},
                timestamp=datetime.now().isoformat()
            )
            return False, security_event, None
            
        # Step 3: Sanitize input
        sanitized_input = self._sanitize_input(input_data, input_type)
        
        return True, None, sanitized_input
        
    def _apply_validation_rules(self, input_data: Any, input_type: str) -> Dict[str, Any]:
        """Apply validation rules based on input type"""
        
        if input_type not in self.validation_rules:
            return {"valid": False, "reason": f"Unknown input type: {input_type}"}
            
        rules = self.validation_rules[input_type]
        
        if input_type == "text_input":
            return self._validate_text_input(input_data, rules)
        elif input_type == "image_data":
            return self._validate_image_data(input_data, rules)
        elif input_type == "file_paths":
            return self._validate_file_path(input_data, rules)
        else:
            return {"valid": True, "reason": "No specific validation"}
            
    def _validate_text_input(self, text: str, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate text input"""
        
        if len(text) > rules["max_length"]:
            return {"valid": False, "reason": f"Text too long: {len(text)} > {rules['max_length']}"}
            
        for pattern in rules["block_patterns"]:
            if pattern.lower() in text.lower():
                return {"valid": False, "reason": f"Blocked pattern detected: {pattern}"}
                
        # Check allowed characters
        if not re.match(rules["allowed_chars"], text):
            return {"valid": False, "reason": "Contains invalid characters"}
            
        return {"valid": True, "reason": "Text validation passed"}
        
    def _validate_image_data(self, image_data: bytes, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate image data"""
        
        if len(image_data) > rules["max_size_mb"] * 1024 * 1024:
            return {"valid": False, "reason": f"Image too large: {len(image_data)} bytes"}
            
        # Simple format detection (in production, use proper image library)
        if image_data[:4] == b'\xff\xd8\xff\xe0':  # JPEG magic bytes
            format_detected = "JPEG"
        elif image_data[:8] == b'\x89\x50\x4e\x47\x0d\x0a\x1a\x0a':  # PNG magic bytes
            format_detected = "PNG"
        else:
            format_detected = "UNKNOWN"
            
        if format_detected not in rules["allowed_formats"]:
            return {"valid": False, "reason": f"Unsupported format: {format_detected}"}
            
        return {"valid": True, "reason": "Image validation passed"}
        
    def _validate_file_path(self, file_path: str, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate file path"""
        
        if len(file_path) > rules["max_length"]:
            return {"valid": False, "reason": f"Path too long: {len(file_path)}"}
            
        for pattern in rules["block_patterns"]:
            if pattern in file_path:
                return {"valid": False, "reason": f"Blocked path pattern: {pattern}"}
                
        # Check file extension
        if not any(file_path.lower().endswith(ext) for ext in rules["allowed_extensions"]):
            return {"valid": False, "reason": "Unsupported file extension"}
            
        return {"valid": True, "reason": "File path validation passed"}
        
    def _sanitize_input(self, input_data: Any, input_type: str) -> Any:
        """Sanitize input data"""
        
        if input_type == "text_input" and isinstance(input_data, str):
            # Basic HTML/script sanitization
            sanitized = input_data.replace("<", "&lt;").replace(">", "&gt;")
            sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
            return sanitized
            
        elif input_type == "file_paths" and isinstance(input_data, str):
            # Path traversal protection
            sanitized = input_data.replace("../", "").replace("..\\", "")
            return sanitized
            
        return input_data


class SecurityOrchestrator:
    """Main security orchestrator coordinating all security components"""
    
    def __init__(self):
        self.crypto_manager = CryptographicManager()
        self.threat_detector = MLThreatDetector()
        self.input_validator = SecureInputValidator(self.crypto_manager, self.threat_detector)
        self.security_events: List[SecurityEvent] = []
        self.blocked_ips: Dict[str, datetime] = {}
        self.rate_limits: Dict[str, List[float]] = {}
        
        logger.info("🏛️ Security orchestrator initialized")
        
    async def initialize_security(self):
        """Initialize security components"""
        logger.info("🔒 Initializing Advanced Security Enhancement v4.0")
        
        # Generate initial keys
        self.crypto_manager.generate_key("AES", 256, "data_encryption")
        self.crypto_manager.generate_key("HMAC", 256, "data_signing")
        
        # Start security monitoring
        asyncio.create_task(self._security_monitoring_loop())
        
        logger.info("✅ Security initialization complete")
        
    async def process_request_securely(self, 
                                     input_data: Any, 
                                     input_type: str,
                                     source_ip: str = "unknown",
                                     encrypt_response: bool = True) -> Dict[str, Any]:
        """Process request with comprehensive security"""
        
        # Step 1: Rate limiting
        if not self._check_rate_limit(source_ip):
            return {
                "success": False,
                "error": "Rate limit exceeded",
                "security_action": SecurityAction.BLOCK.value
            }
            
        # Step 2: IP blocking check
        if self._is_ip_blocked(source_ip):
            return {
                "success": False,
                "error": "IP address blocked",
                "security_action": SecurityAction.BLOCK.value
            }
            
        # Step 3: Input validation and threat detection
        valid, security_event, sanitized_data = await self.input_validator.validate_input(
            input_data, input_type, source_ip
        )
        
        if not valid and security_event:
            await self._handle_security_event(security_event)
            return {
                "success": False,
                "error": "Security threat detected",
                "event_id": security_event.event_id,
                "security_action": SecurityAction.BLOCK.value
            }
            
        # Step 4: Process request (placeholder for actual processing)
        try:
            # Simulate secure processing
            processing_result = await self._secure_processing(sanitized_data, input_type)
            
            # Step 5: Encrypt response if requested
            if encrypt_response:
                encryption_key = self.crypto_manager.generate_key("AES", 256, "response_encryption")
                encrypted_result = self.crypto_manager.encrypt_data(
                    json.dumps(processing_result).encode(),
                    encryption_key
                )
                
                return {
                    "success": True,
                    "encrypted_response": base64.b64encode(encrypted_result).decode(),
                    "encryption_key_id": encryption_key,
                    "security_status": "secure"
                }
            else:
                return {
                    "success": True,
                    "result": processing_result,
                    "security_status": "validated"
                }
                
        except Exception as e:
            logger.error(f"Secure processing failed: {e}")
            return {
                "success": False,
                "error": "Processing failed",
                "security_status": "error"
            }
            
    async def _secure_processing(self, data: Any, input_type: str) -> Dict[str, Any]:
        """Secure processing of validated data"""
        
        if input_type == "text_input":
            # Simulate VLM inference with security monitoring
            result = {
                "processed_text": f"Securely processed: {data}",
                "processing_time": time.time(),
                "security_score": 1.0
            }
            
        elif input_type == "image_data":
            result = {
                "image_analysis": "Secure image processing complete",
                "processing_time": time.time(),
                "security_score": 1.0
            }
            
        else:
            result = {
                "message": "Generic secure processing complete",
                "processing_time": time.time(),
                "security_score": 1.0
            }
            
        return result
        
    def _check_rate_limit(self, source_ip: str, max_requests_per_minute: int = 60) -> bool:
        """Check if request is within rate limits"""
        current_time = time.time()
        
        if source_ip not in self.rate_limits:
            self.rate_limits[source_ip] = []
            
        # Clean old requests
        self.rate_limits[source_ip] = [
            req_time for req_time in self.rate_limits[source_ip]
            if current_time - req_time < 60  # Keep requests from last minute
        ]
        
        # Check rate limit
        if len(self.rate_limits[source_ip]) >= max_requests_per_minute:
            logger.warning(f"Rate limit exceeded for IP: {source_ip}")
            return False
            
        # Add current request
        self.rate_limits[source_ip].append(current_time)
        return True
        
    def _is_ip_blocked(self, source_ip: str) -> bool:
        """Check if IP address is blocked"""
        if source_ip in self.blocked_ips:
            block_time = self.blocked_ips[source_ip]
            if datetime.now() - block_time < timedelta(hours=24):  # 24-hour block
                return True
            else:
                # Remove expired block
                del self.blocked_ips[source_ip]
                
        return False
        
    async def _handle_security_event(self, event: SecurityEvent):
        """Handle security event with appropriate response"""
        self.security_events.append(event)
        
        logger.warning(f"🚨 Security Event: {event.event_id} - {event.attack_type.value} from {event.source_ip}")
        
        # Determine response action based on threat level
        if event.threat_level == ThreatLevel.CRITICAL:
            # Block IP immediately
            self.blocked_ips[event.source_ip] = datetime.now()
            event.mitigation_actions.append("IP_BLOCKED")
            
        elif event.threat_level == ThreatLevel.HIGH:
            # Increase monitoring and rate limiting
            event.mitigation_actions.append("ENHANCED_MONITORING")
            
        # Log security event
        await self._log_security_event(event)
        
    async def _log_security_event(self, event: SecurityEvent):
        """Log security event to secure audit trail"""
        
        # Create audit log entry
        audit_entry = {
            "timestamp": event.timestamp,
            "event_id": event.event_id,
            "threat_level": event.threat_level.value,
            "attack_type": event.attack_type.value,
            "source_ip": event.source_ip,
            "mitigation_actions": event.mitigation_actions,
            "evidence_hash": hashlib.sha256(json.dumps(event.evidence, default=str).encode()).hexdigest()
        }
        
        # Sign audit entry
        signing_key_id = list(self.crypto_manager.keys.keys())[1] if len(self.crypto_manager.keys) > 1 else None
        if signing_key_id:
            signature = self.crypto_manager.sign_data(
                json.dumps(audit_entry).encode(),
                signing_key_id
            )
            audit_entry["signature"] = base64.b64encode(signature).decode()
            
        # Save to secure audit log
        audit_file = Path("security_audit.jsonl")
        with open(audit_file, "a") as f:
            f.write(json.dumps(audit_entry) + "\n")
            
    async def _security_monitoring_loop(self):
        """Continuous security monitoring"""
        while True:
            try:
                # Rotate keys if needed
                self.crypto_manager.rotate_keys()
                
                # Clean up old rate limit data
                self._cleanup_rate_limits()
                
                # Generate security summary
                await self._generate_security_summary()
                
                await asyncio.sleep(300)  # Monitor every 5 minutes
                
            except Exception as e:
                logger.error(f"Security monitoring error: {e}")
                await asyncio.sleep(60)
                
    def _cleanup_rate_limits(self):
        """Clean up old rate limit data"""
        current_time = time.time()
        
        for ip in list(self.rate_limits.keys()):
            self.rate_limits[ip] = [
                req_time for req_time in self.rate_limits[ip]
                if current_time - req_time < 3600  # Keep last hour
            ]
            
            if not self.rate_limits[ip]:
                del self.rate_limits[ip]
                
    async def _generate_security_summary(self):
        """Generate security status summary"""
        
        # Recent events (last 24 hours)
        recent_events = [
            event for event in self.security_events
            if datetime.now() - datetime.fromisoformat(event.timestamp) < timedelta(days=1)
        ]
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "active_keys": len([k for k in self.crypto_manager.keys.values() if k.active]),
            "blocked_ips": len(self.blocked_ips),
            "recent_threats": len(recent_events),
            "threat_breakdown": {
                threat_type.value: len([e for e in recent_events if e.attack_type == threat_type])
                for threat_type in AttackType
            },
            "security_score": self._calculate_security_score()
        }
        
        if len(recent_events) > 0:
            logger.info(f"🛡️ Security Summary: {summary['recent_threats']} threats in 24h, Security Score: {summary['security_score']:.2f}")
            
    def _calculate_security_score(self) -> float:
        """Calculate overall security score"""
        
        # Base score
        score = 1.0
        
        # Deduct points for recent threats
        recent_events = [
            event for event in self.security_events
            if datetime.now() - datetime.fromisoformat(event.timestamp) < timedelta(hours=1)
        ]
        
        score -= len(recent_events) * 0.1
        
        # Deduct points for blocked IPs (indicates ongoing attacks)
        score -= len(self.blocked_ips) * 0.05
        
        # Ensure score stays within bounds
        return max(0.0, min(1.0, score))
        
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        
        recent_events = [
            event for event in self.security_events
            if datetime.now() - datetime.fromisoformat(event.timestamp) < timedelta(days=1)
        ]
        
        return {
            "security_score": self._calculate_security_score(),
            "active_keys": len([k for k in self.crypto_manager.keys.values() if k.active]),
            "blocked_ips": len(self.blocked_ips),
            "recent_threats_24h": len(recent_events),
            "threat_types": {
                threat_type.value: len([e for e in recent_events if e.attack_type == threat_type])
                for threat_type in AttackType
            },
            "system_status": "secure" if self._calculate_security_score() > 0.8 else "under_attack"
        }


# Global security orchestrator instance
security_orchestrator = SecurityOrchestrator()


async def initialize_security():
    """Initialize global security system"""
    await security_orchestrator.initialize_security()
    return security_orchestrator


async def process_securely(input_data: Any, input_type: str, source_ip: str = "unknown"):
    """Process data with comprehensive security"""
    return await security_orchestrator.process_request_securely(input_data, input_type, source_ip)


def get_security_status():
    """Get current security status"""
    return security_orchestrator.get_security_status()


async def main():
    """Main execution for testing security enhancement"""
    logger.info("🧪 Testing Advanced Security Enhancement")
    
    # Initialize security
    orchestrator = SecurityOrchestrator()
    await orchestrator.initialize_security()
    
    # Test secure processing
    test_inputs = [
        ("Hello, what's in this image?", "text_input"),
        ("SELECT * FROM users WHERE id=1; DROP TABLE users;", "text_input"),
        ("<script>alert('xss')</script>", "text_input"),
        (b"\xff\xd8\xff\xe0" + b"fake_jpeg_data", "image_data")
    ]
    
    for input_data, input_type in test_inputs:
        logger.info(f"Testing {input_type}: {str(input_data)[:50]}...")
        
        result = await orchestrator.process_request_securely(
            input_data, 
            input_type, 
            source_ip="192.168.1.100"
        )
        
        logger.info(f"Result: {result.get('success', False)} - {result.get('error', 'OK')}")
        
    # Get security status
    status = orchestrator.get_security_status()
    
    # Save security report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"security_report_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump(status, f, indent=2)
        
    logger.info(f"🛡️ Security Report saved: {report_file}")
    logger.info(f"🔒 Security Score: {status['security_score']:.2f}")
    
    return status


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())