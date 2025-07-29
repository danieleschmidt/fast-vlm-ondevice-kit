# Deployment and Operations Guide

This document covers deployment strategies, infrastructure requirements, and operational procedures for Fast VLM On-Device Kit.

## Deployment Overview

### Deployment Models

1. **Development Deployment**: Local development and testing
2. **CI/CD Pipeline**: Automated testing and model conversion
3. **Distribution Deployment**: Package distribution (PyPI, Swift Package Manager)
4. **Production Integration**: Integration with mobile applications

## Infrastructure Requirements

### Development Environment

#### Minimum Requirements
- **macOS**: 12.0+ (for Core ML conversion)
- **Python**: 3.10+ with pip
- **Xcode**: 15.0+ (for iOS development)
- **Memory**: 16GB RAM (32GB recommended for large models)
- **Storage**: 100GB available space
- **Network**: Stable internet for dependency downloads

#### Recommended Setup
```bash
# System setup script
#!/bin/bash
set -e

echo "Setting up Fast VLM development environment..."

# Install system dependencies
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS setup
    if ! command -v brew &> /dev/null; then
        echo "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    brew install python@3.10 git-lfs
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux setup (limited functionality)
    sudo apt-get update
    sudo apt-get install -y python3.10 python3.10-venv python3-pip git-lfs
fi

# Setup Python environment
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel

# Install project
pip install -e ".[dev]"
pre-commit install

echo "Development environment setup complete!"
```

### CI/CD Infrastructure

#### GitHub Actions Runners
- **Self-hosted macOS runners** for Core ML conversion
- **Ubuntu runners** for Python testing and linting
- **Matrix builds** across Python versions and platforms

#### Resource Allocation
```yaml
# GitHub Actions resource configuration
jobs:
  model-conversion:
    runs-on: [self-hosted, macOS, Apple-Silicon]
    resources:
      memory: "32GB"
      cpu: "8-cores"
      timeout: 120  # minutes
    
  python-tests:
    runs-on: ubuntu-latest
    resources:
      memory: "8GB" 
      cpu: "4-cores"
      timeout: 30
```

## Deployment Strategies

### 1. Python Package Deployment

#### PyPI Release Process
Create `scripts/release.py`:
```python
#!/usr/bin/env python3
"""Automated release script for Fast VLM On-Device Kit."""

import subprocess
import sys
import json
from pathlib import Path
from typing import Dict, Any

class ReleaseManager:
    """Manage package releases."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.version_file = self.project_root / "src/fast_vlm_ondevice/__init__.py"
    
    def get_current_version(self) -> str:
        """Extract current version from package."""
        init_content = self.version_file.read_text()
        for line in init_content.split('\n'):
            if line.startswith('__version__'):
                return line.split('"')[1]
        raise ValueError("Version not found in __init__.py")
    
    def update_version(self, new_version: str):
        """Update version in package files."""
        # Update __init__.py
        init_content = self.version_file.read_text()
        updated_content = init_content.replace(
            f'__version__ = "{self.get_current_version()}"',
            f'__version__ = "{new_version}"'
        )
        self.version_file.write_text(updated_content)
        
        print(f"Updated version to {new_version}")
    
    def run_tests(self) -> bool:
        """Run full test suite before release."""
        try:
            subprocess.run(["make", "check"], check=True, cwd=self.project_root)
            print("All tests passed!")
            return True
        except subprocess.CalledProcessError:
            print("Tests failed! Aborting release.")
            return False
    
    def build_package(self):
        """Build distribution packages."""
        subprocess.run(["python", "-m", "build"], check=True, cwd=self.project_root)
        print("Package built successfully!")
    
    def upload_to_pypi(self, test: bool = False):
        """Upload package to PyPI."""
        repository = "--repository testpypi" if test else ""
        cmd = f"python -m twine upload {repository} dist/*"
        subprocess.run(cmd.split(), check=True, cwd=self.project_root)
        
        target = "Test PyPI" if test else "PyPI"
        print(f"Package uploaded to {target}!")
    
    def create_git_tag(self, version: str):
        """Create and push git tag."""
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", f"Release v{version}"], check=True)
        subprocess.run(["git", "tag", f"v{version}"], check=True)
        subprocess.run(["git", "push", "origin", "main", "--tags"], check=True)
        print(f"Git tag v{version} created and pushed!")

def main():
    """Main release workflow."""
    if len(sys.argv) != 2:
        print("Usage: python release.py <version>")
        sys.exit(1)
    
    new_version = sys.argv[1]
    release_manager = ReleaseManager()
    
    print(f"Starting release process for version {new_version}")
    
    # Pre-release checks
    if not release_manager.run_tests():
        sys.exit(1)
    
    # Update version
    release_manager.update_version(new_version)
    
    # Build and upload
    release_manager.build_package()
    
    # Test upload first
    print("Uploading to Test PyPI...")
    release_manager.upload_to_pypi(test=True)
    
    # Confirm production upload
    confirm = input("Upload to production PyPI? (y/N): ")
    if confirm.lower() == 'y':
        release_manager.upload_to_pypi(test=False)
        release_manager.create_git_tag(new_version)
        print(f"Release {new_version} completed successfully!")
    else:
        print("Production upload cancelled.")

if __name__ == "__main__":
    main()
```

#### Package Distribution Configuration
Update `pyproject.toml`:
```toml
[project.scripts]
fastvlm-convert = "fast_vlm_ondevice.cli:main"
fastvlm-benchmark = "fast_vlm_ondevice.benchmark:main"

[project.entry-points."fastvlm.converters"]
pytorch = "fast_vlm_ondevice.converter:FastVLMConverter"

[tool.setuptools.package-data]
"fast_vlm_ondevice" = ["*.json", "*.yaml", "configs/*"]
```

### 2. Swift Package Deployment

#### Swift Package Manager Configuration
Update `ios/Package.swift`:
```swift
// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "FastVLMKit",
    platforms: [
        .iOS(.v15),
        .macOS(.v12),
        .tvOS(.v15),
        .watchOS(.v8)
    ],
    products: [
        .library(
            name: "FastVLMKit",
            targets: ["FastVLMKit"]
        ),
        .executable(
            name: "FastVLMCLI", 
            targets: ["FastVLMCLI"]
        )
    ],
    dependencies: [
        // Add external dependencies here if needed
    ],
    targets: [
        .target(
            name: "FastVLMKit",
            dependencies: [],
            resources: [
                .process("Resources")
            ]
        ),
        .executableTarget(
            name: "FastVLMCLI",
            dependencies: ["FastVLMKit"]
        ),
        .testTarget(
            name: "FastVLMKitTests",
            dependencies: ["FastVLMKit"],
            resources: [
                .process("TestResources")
            ]
        )
    ]
)
```

## Container Deployment

### Production Container Build

Create `docker/Dockerfile.production`:
```dockerfile
# Multi-stage production build
FROM python:3.10-slim as builder

WORKDIR /build
COPY requirements.txt pyproject.toml ./
COPY src/ src/

# Build wheel
RUN pip install build && python -m build

# Production image
FROM python:3.10-slim as production

# Create non-root user
RUN groupadd -r fastvlm && useradd -r -g fastvlm fastvlm

# Install runtime dependencies only
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Security hardening
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get purge -y --auto-remove

# Switch to non-root user
USER fastvlm
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import fast_vlm_ondevice; print('healthy')" || exit 1

ENTRYPOINT ["python", "-m", "fast_vlm_ondevice"]
```

### Kubernetes Deployment

Create `k8s/deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fastvlm-converter
  labels:
    app: fastvlm-converter
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fastvlm-converter
  template:
    metadata:
      labels:
        app: fastvlm-converter
    spec:
      containers:
      - name: converter
        image: fastvlm/converter:latest
        ports:
        - containerPort: 8000
        env:
        - name: MAX_MODEL_SIZE_MB
          value: "2048"
        - name: SECURE_MODE
          value: "true"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"  
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-cache
          mountPath: /app/models
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
```

## Monitoring and Observability

### Application Metrics

Create monitoring configuration in `monitoring/grafana-dashboard.json`:
```json
{
  "dashboard": {
    "title": "Fast VLM Operations Dashboard",
    "panels": [
      {
        "title": "Model Conversion Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(fastvlm_conversions_total[5m])",
            "legendFormat": "Conversions/sec"
          }
        ]
      },
      {
        "title": "Conversion Latency",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, fastvlm_conversion_duration_seconds_bucket)",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, fastvlm_conversion_duration_seconds_bucket)",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(fastvlm_errors_total[5m])",
            "legendFormat": "Errors/sec"
          }
        ]
      }
    ]
  }
}
```

### Log Aggregation

Create `monitoring/fluentd-config.yaml`:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/containers/fastvlm-*.log
      pos_file /var/log/fluentd-containers.log.pos
      tag kubernetes.*
      format json
      time_key timestamp
    </source>
    
    <filter kubernetes.**>
      @type kubernetes_metadata
    </filter>
    
    <match kubernetes.**>
      @type elasticsearch
      host elasticsearch.monitoring.svc.cluster.local
      port 9200
      index_name fastvlm-logs
      type_name _doc
    </match>
```

## Disaster Recovery

### Backup Strategy

Create `scripts/backup.py`:
```python
#!/usr/bin/env python3
"""Backup and recovery utilities."""

import shutil
import boto3
from pathlib import Path
from datetime import datetime
import logging

class BackupManager:
    """Manage backups for Fast VLM infrastructure."""
    
    def __init__(self, s3_bucket: str = None):
        self.s3_bucket = s3_bucket
        self.s3_client = boto3.client('s3') if s3_bucket else None
        self.logger = logging.getLogger(__name__)
    
    def backup_models(self, source_dir: Path, backup_dir: Path):
        """Backup converted models."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"models_backup_{timestamp}"
        
        shutil.copytree(source_dir, backup_path)
        self.logger.info(f"Models backed up to {backup_path}")
        
        if self.s3_client:
            self._upload_to_s3(backup_path, f"backups/models/{timestamp}")
    
    def backup_configurations(self, config_files: list):
        """Backup configuration files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path(f"config_backup_{timestamp}")
        backup_dir.mkdir()
        
        for config_file in config_files:
            if Path(config_file).exists():
                shutil.copy2(config_file, backup_dir)
        
        if self.s3_client:
            self._upload_to_s3(backup_dir, f"backups/configs/{timestamp}")
    
    def _upload_to_s3(self, local_path: Path, s3_key: str):
        """Upload backup to S3."""
        if local_path.is_file():
            self.s3_client.upload_file(str(local_path), self.s3_bucket, s3_key)
        else:
            for file_path in local_path.rglob('*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(local_path)
                    s3_file_key = f"{s3_key}/{relative_path}"
                    self.s3_client.upload_file(str(file_path), self.s3_bucket, s3_file_key)
```

### Recovery Procedures

Create `docs/operations/disaster-recovery.md`:
```markdown
# Disaster Recovery Procedures

## Recovery Time Objectives (RTO)
- **Development Environment**: 4 hours
- **CI/CD Pipeline**: 2 hours  
- **Package Distribution**: 1 hour
- **Documentation**: 1 hour

## Recovery Point Objectives (RPO)
- **Source Code**: 0 (Git versioning)
- **Model Artifacts**: 24 hours
- **Configuration**: 1 hour
- **Monitoring Data**: 4 hours

## Recovery Procedures

### 1. Source Code Recovery
```bash
# Clone repository
git clone https://github.com/yourusername/fast-vlm-ondevice-kit.git
cd fast-vlm-ondevice-kit

# Restore to specific commit if needed
git checkout <commit-hash>
```

### 2. Environment Recovery
```bash
# Automated environment setup
./scripts/setup-environment.sh

# Restore configurations
./scripts/restore-configs.sh
```

### 3. Model Recovery
```bash
# Download model backups from S3
aws s3 sync s3://fastvlm-backups/models/latest ./models/

# Verify model integrity
python scripts/verify-models.py
```

### 4. Service Recovery
```bash
# Redeploy services
kubectl apply -f k8s/
kubectl rollout status deployment/fastvlm-converter

# Verify services are healthy
kubectl get pods -l app=fastvlm-converter
```
```

## Operational Runbooks

### Deployment Checklist

Create `docs/operations/deployment-checklist.md`:
```markdown
# Deployment Checklist

## Pre-Deployment
- [ ] All tests passing in CI/CD
- [ ] Security scan completed
- [ ] Performance benchmarks meet requirements
- [ ] Documentation updated
- [ ] Backup of current environment created
- [ ] Rollback plan documented

## Deployment
- [ ] Deploy to staging environment
- [ ] Run smoke tests
- [ ] Validate monitoring and alerting
- [ ] Deploy to production
- [ ] Monitor deployment metrics
- [ ] Verify all services healthy

## Post-Deployment
- [ ] Run full integration tests
- [ ] Monitor error rates and performance
- [ ] Verify backup systems operational
- [ ] Update deployment documentation
- [ ] Notify stakeholders of completion
- [ ] Schedule post-deployment review
```

This comprehensive deployment and operations guide ensures reliable, scalable, and maintainable deployment of Fast VLM On-Device Kit across all environments.