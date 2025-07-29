# Security Implementation Guide

This document outlines comprehensive security measures for the Fast VLM On-Device Kit.

## Security Framework

### 1. Dependency Security

#### Automated Vulnerability Scanning

Create `.github/dependabot.yml`:
```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    commit-message:
      prefix: "security"
      include: "scope"
    reviewers:
      - "security-team"
    
  - package-ecosystem: "swift"
    directory: "/ios"
    schedule:
      interval: "weekly"
    commit-message:
      prefix: "security"
```

#### Security Scanning Configuration

Enhance `.pre-commit-config.yaml`:
```yaml
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: [--skip, B101, --format, json, --output, security-report.json]
        exclude: tests/
  
  - repo: https://github.com/PyCQA/safety
    rev: v2.3.4
    hooks:
      - id: safety
        args: [--json, --output, safety-report.json]
```

### 2. Code Security

#### Secure Coding Guidelines

Create `docs/security/secure-coding.md`:
```markdown
# Secure Coding Guidelines

## Input Validation
- Validate all external inputs (images, text, file paths)
- Sanitize file paths to prevent directory traversal
- Limit input sizes to prevent DoS attacks

## Model Security
- Verify model file integrity before loading
- Use secure model loading practices
- Implement model signature verification

## Data Handling
- Never log sensitive data (API keys, user content)
- Implement secure temporary file handling
- Clear sensitive data from memory after use

## Error Handling
- Don't expose internal information in error messages
- Log security events for monitoring
- Implement proper exception handling
```

#### Security Utilities

Create `src/fast_vlm_ondevice/security.py`:
```python
import hashlib
import hmac
import secrets
from pathlib import Path
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)

class SecurityValidator:
    """Security validation utilities."""
    
    @staticmethod
    def validate_file_path(file_path: Union[str, Path], allowed_extensions: set) -> bool:
        """Validate file path for security."""
        path = Path(file_path)
        
        # Check for directory traversal
        if '..' in str(path):
            logger.warning(f"Directory traversal attempt: {file_path}")
            return False
        
        # Check file extension
        if path.suffix.lower() not in allowed_extensions:
            logger.warning(f"Invalid file extension: {path.suffix}")
            return False
        
        # Check if path exists and is a file
        if not (path.exists() and path.is_file()):
            logger.warning(f"Invalid file path: {file_path}")
            return False
        
        return True
    
    @staticmethod
    def verify_model_integrity(model_path: Path, expected_hash: Optional[str] = None) -> bool:
        """Verify model file integrity."""
        if not model_path.exists():
            return False
        
        if expected_hash:
            actual_hash = SecurityValidator.calculate_file_hash(model_path)
            if actual_hash != expected_hash:
                logger.error(f"Model integrity check failed: {model_path}")
                return False
        
        return True
    
    @staticmethod
    def calculate_file_hash(file_path: Path, algorithm: str = 'sha256') -> str:
        """Calculate file hash for integrity checking."""
        hash_obj = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    
    @staticmethod
    def secure_random_string(length: int = 32) -> str:
        """Generate cryptographically secure random string."""
        return secrets.token_urlsafe(length)

class SecureTempFile:
    """Secure temporary file handling."""
    
    def __init__(self, suffix: str = '.tmp', delete_on_exit: bool = True):
        self.suffix = suffix
        self.delete_on_exit = delete_on_exit
        self.path = None
    
    def __enter__(self):
        import tempfile
        fd, self.path = tempfile.mkstemp(suffix=self.suffix)
        return Path(self.path)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.delete_on_exit and self.path:
            try:
                Path(self.path).unlink()
            except FileNotFoundError:
                pass
```

### 3. Container Security

#### Secure Dockerfile Practices

Update `docker/Dockerfile.converter`:
```dockerfile
# Use specific version tags, not 'latest'
FROM python:3.10.12-slim as base

# Run as non-root user
RUN groupadd -r converter && useradd -r -g converter converter

# Install security updates
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        git \
        wget \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Set secure permissions
WORKDIR /workspace
RUN chown converter:converter /workspace

# Copy and install dependencies as non-root
USER converter
COPY --chown=converter:converter requirements.txt ./
RUN pip install --user --no-cache-dir -r requirements.txt

# Use COPY instead of ADD for better security
COPY --chown=converter:converter . .

# Don't run as root
USER converter

# Use exec form for better signal handling
ENTRYPOINT ["python", "-m", "fast_vlm_ondevice.converter"]
```

#### Container Security Scanning

Create `docker/security-scan.sh`:
```bash
#!/bin/bash
set -e

echo "Scanning Docker image for vulnerabilities..."

# Scan with Trivy
trivy image --severity HIGH,CRITICAL fastvlm-converter:latest

# Scan with Docker Scout (if available)
if command -v docker &> /dev/null; then
    docker scout cves fastvlm-converter:latest
fi

echo "Security scan completed."
```

### 4. Secrets Management

#### Environment Configuration

Create `.env.example`:
```bash
# Model configuration
MODEL_CACHE_DIR=/tmp/fastvlm_models
MAX_MODEL_SIZE_MB=2048

# Security settings
ENABLE_SECURITY_LOGGING=true
SECURE_MODE=true

# Optional: External services
# SENTRY_DSN=https://your-sentry-dsn
# MONITORING_ENDPOINT=https://your-monitoring-endpoint
```

#### Secrets Validation

Create `src/fast_vlm_ondevice/config.py`:
```python
import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class SecurityConfig:
    """Security configuration management."""
    
    def __init__(self):
        self.secure_mode = self._get_bool_env('SECURE_MODE', True)
        self.max_model_size_mb = self._get_int_env('MAX_MODEL_SIZE_MB', 2048)
        self.enable_security_logging = self._get_bool_env('ENABLE_SECURITY_LOGGING', True)
        self.model_cache_dir = Path(os.getenv('MODEL_CACHE_DIR', '/tmp/fastvlm_models'))
    
    @staticmethod
    def _get_bool_env(key: str, default: bool) -> bool:
        """Get boolean environment variable."""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    @staticmethod
    def _get_int_env(key: str, default: int) -> int:
        """Get integer environment variable."""
        try:
            return int(os.getenv(key, str(default)))
        except ValueError:
            logger.warning(f"Invalid value for {key}, using default: {default}")
            return default
    
    def validate_config(self) -> bool:
        """Validate security configuration."""
        if self.secure_mode:
            # Validate secure settings
            if not self.model_cache_dir.parent.exists():
                logger.error(f"Model cache directory parent does not exist: {self.model_cache_dir.parent}")
                return False
            
            if self.max_model_size_mb < 100:
                logger.warning("Max model size is very small, this might cause issues")
        
        return True
```

### 5. Audit and Compliance

#### Security Audit Script

Create `scripts/security-audit.py`:
```python
#!/usr/bin/env python3
"""Security audit script for Fast VLM On-Device Kit."""

import subprocess
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

class SecurityAuditor:
    """Perform security audit of the codebase."""
    
    def __init__(self):
        self.results = {
            'vulnerabilities': [],
            'code_issues': [],
            'dependencies': [],
            'summary': {}
        }
    
    def run_bandit_scan(self) -> Dict[str, Any]:
        """Run Bandit security scanner."""
        try:
            result = subprocess.run(
                ['bandit', '-r', 'src/', '-f', 'json'],
                capture_output=True,
                text=True,
                check=False
            )
            return json.loads(result.stdout) if result.stdout else {}
        except Exception as e:
            return {'error': str(e)}
    
    def run_safety_check(self) -> Dict[str, Any]:
        """Run Safety dependency scanner."""
        try:
            result = subprocess.run(
                ['safety', 'check', '--json'],
                capture_output=True,
                text=True,
                check=False
            )
            return json.loads(result.stdout) if result.stdout else {}
        except Exception as e:
            return {'error': str(e)}
    
    def check_file_permissions(self) -> List[Dict[str, Any]]:
        """Check for overly permissive file permissions."""
        issues = []
        for file_path in Path('.').rglob('*'):
            if file_path.is_file():
                # Check for world-writable files
                if file_path.stat().st_mode & 0o002:
                    issues.append({
                        'file': str(file_path),
                        'issue': 'World-writable file',
                        'severity': 'HIGH'
                    })
        return issues
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        print("Running security audit...")
        
        # Run scans
        bandit_results = self.run_bandit_scan()
        safety_results = self.run_safety_check()
        permission_issues = self.check_file_permissions()
        
        # Compile results
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'code_security': bandit_results,
            'dependency_security': safety_results,
            'file_permissions': permission_issues,
            'summary': {
                'total_issues': len(bandit_results.get('results', [])) + 
                              len(safety_results.get('vulnerabilities', [])) +
                              len(permission_issues),
                'high_severity': 0,  # Calculate based on results
                'recommendations': [
                    "Enable automated security scanning in CI/CD",
                    "Regularly update dependencies",
                    "Implement security monitoring",
                    "Review and fix identified issues"
                ]
            }
        }
        
        return report

if __name__ == '__main__':
    auditor = SecurityAuditor()
    report = auditor.generate_report()
    
    # Save report
    with open('security-audit-report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Security audit completed. Found {report['summary']['total_issues']} issues.")
    print("Report saved to security-audit-report.json")
    
    # Exit with error code if issues found
    sys.exit(1 if report['summary']['total_issues'] > 0 else 0)
```

### 6. Incident Response

#### Security Incident Response Plan

Create `docs/security/incident-response.md`:
```markdown
# Security Incident Response Plan

## Incident Classification

### Severity Levels
- **Critical**: Data breach, code injection, privilege escalation
- **High**: Vulnerability with high impact potential
- **Medium**: Security misconfiguration, dependency vulnerability
- **Low**: Information disclosure, minor security issue

## Response Procedures

### Immediate Response (0-4 hours)
1. Assess severity and impact scope
2. Isolate affected systems if necessary  
3. Document incident details
4. Notify security team and stakeholders

### Investigation (4-24 hours)
1. Collect logs and evidence
2. Identify root cause
3. Assess data/system impact
4. Develop remediation plan

### Remediation (24-72 hours)
1. Apply security patches/fixes
2. Update security configurations
3. Test remediation effectiveness
4. Update monitoring/detection rules

### Recovery (72+ hours)
1. Restore normal operations
2. Conduct post-incident review
3. Update security procedures
4. Implement lessons learned

## Contact Information
- Security Team: security@fastvlm.com
- Emergency Contact: +1-XXX-XXX-XXXX
- External Security Consultant: security-consultant@company.com
```

## Implementation Checklist

### Phase 1: Foundation (Week 1)
- [ ] Update dependency scanning configuration
- [ ] Implement secure coding utilities
- [ ] Configure container security scanning
- [ ] Set up secrets management

### Phase 2: Monitoring (Week 2)
- [ ] Enable security logging
- [ ] Set up vulnerability monitoring
- [ ] Configure automated scanning in CI/CD
- [ ] Implement security metrics collection

### Phase 3: Response (Week 3)
- [ ] Create incident response procedures
- [ ] Set up security alerting
- [ ] Train team on security practices
- [ ] Document security runbooks

### Phase 4: Validation (Week 4)
- [ ] Run comprehensive security audit
- [ ] Perform penetration testing
- [ ] Review and fix identified issues
- [ ] Update security documentation

## Compliance Framework

### Security Standards Alignment
- **OWASP Top 10**: Address web application security risks
- **NIST Cybersecurity Framework**: Implement comprehensive security program
- **GDPR/Privacy**: Ensure data protection compliance (if applicable)
- **Supply Chain Security**: Secure development and deployment pipeline

This security implementation provides a comprehensive foundation for protecting the Fast VLM On-Device Kit while maintaining usability and performance.