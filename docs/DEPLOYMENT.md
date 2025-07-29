# Deployment Guide

This guide covers deploying Fast VLM On-Device Kit in production environments.

## Deployment Strategies

### 1. iOS App Distribution

#### App Store Distribution

**Requirements:**
- Apple Developer Account
- Code signing certificates
- App Store review compliance

**Build Process:**
```bash
# 1. Build Swift package
cd ios
swift build -c release

# 2. Archive iOS app
xcodebuild archive \
    -scheme FastVLMDemo \
    -archivePath FastVLMDemo.xcarchive

# 3. Export for App Store
xcodebuild -exportArchive \
    -archivePath FastVLMDemo.xcarchive \
    -exportPath . \
    -exportOptionsPlist ExportOptions.plist
```

#### Enterprise Distribution

**Requirements:**
- Apple Enterprise Developer Account
- Enterprise distribution certificate
- Mobile Device Management (MDM) system

**Configuration:**
```bash
# Configure enterprise signing
security unlock-keychain ~/Library/Keychains/login.keychain
codesign --force --sign "iPhone Distribution: Your Company" \
    --entitlements App.entitlements FastVLMDemo.app
```

### 2. Python Package Distribution

#### PyPI Distribution

**Preparation:**
```bash
# Build package
python -m build

# Check package
twine check dist/*

# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

#### Private Package Repository

**Setup with pip-tools:**
```bash
# Create private index
pip install devpi-server
devpi-server --start

# Upload package
devpi use http://localhost:3141
devpi upload dist/*
```

### 3. Container Deployment

#### Docker Production Build

```dockerfile
# Production Dockerfile
FROM python:3.11-slim as production

# Install only runtime dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY scripts/ ./scripts/

# Install package
RUN pip install -e .

# Configure runtime
USER 1000:1000
EXPOSE 8000
CMD ["python", "-m", "fast_vlm_ondevice", "--port", "8000"]
```

**Build and Deploy:**
```bash
# Build production image
docker build -t fast-vlm:latest -f Dockerfile .

# Tag for registry
docker tag fast-vlm:latest your-registry.com/fast-vlm:v1.0.0

# Push to registry
docker push your-registry.com/fast-vlm:v1.0.0
```

#### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fast-vlm-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fast-vlm
  template:
    metadata:
      labels:
        app: fast-vlm
    spec:
      containers:
      - name: fast-vlm
        image: your-registry.com/fast-vlm:v1.0.0
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        ports:
        - containerPort: 8000
        volumeMounts:
        - name: model-storage
          mountPath: /models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
```

## Environment Configuration

### Production Environment Variables

```bash
# Core configuration
export FAST_VLM_ENV=production
export FAST_VLM_LOG_LEVEL=INFO
export FAST_VLM_MODEL_PATH=/models/fast-vlm-base.mlpackage

# Performance tuning
export FAST_VLM_BATCH_SIZE=8
export FAST_VLM_MAX_WORKERS=4
export FAST_VLM_CACHE_SIZE=1000

# Security
export FAST_VLM_API_KEY_FILE=/secrets/api-key
export FAST_VLM_TLS_CERT=/certs/server.crt
export FAST_VLM_TLS_KEY=/certs/server.key

# Monitoring
export FAST_VLM_METRICS_ENABLED=true
export FAST_VLM_METRICS_PORT=9090
export FAST_VLM_TRACING_ENDPOINT=http://jaeger:14268/api/traces
```

### Configuration Files

**Production Config (`config/production.yaml`):**
```yaml
app:
  name: "Fast VLM On-Device"
  version: "1.0.0"
  environment: "production"

models:
  default: "fast-vlm-base"
  cache_size: 1000
  preload: true
  paths:
    fast-vlm-base: "/models/fast-vlm-base.mlpackage"
    fast-vlm-large: "/models/fast-vlm-large.mlpackage"

performance:
  batch_size: 8
  max_workers: 4
  timeout_seconds: 30
  memory_limit_mb: 4096

security:
  tls_enabled: true
  api_key_required: true
  rate_limit: 100  # requests per minute
  
logging:
  level: "INFO"
  format: "json"
  file: "/logs/fast-vlm.log"

monitoring:
  metrics_enabled: true
  health_check_endpoint: "/health"
  readiness_endpoint: "/ready"
```

## Performance Optimization

### iOS Performance Tuning

**Memory Management:**
```swift
// Configure model loading
let config = MLModelConfiguration()
config.computeUnits = .all  // Use CPU + GPU + ANE
config.allowLowPrecisionAccumulationOnGPU = true

// Optimize batch processing
let batchSize = 4  // Adjust based on available memory
let vlm = try FastVLM(modelPath: modelPath, configuration: config)
```

**Energy Efficiency:**
```swift
// Monitor energy usage
let energyProfiler = EnergyProfiler()
energyProfiler.startMeasuring()

// Your inference code here

let metrics = energyProfiler.stopMeasuring()
print("Energy consumed: \(metrics.milliwattHours) mWh")
```

### Python Performance Tuning

**Threading Configuration:**
```python
# Configure threading for optimal performance
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

# Use concurrent processing
from concurrent.futures import ThreadPoolExecutor

def process_batch(images):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_image, img) for img in images]
        return [f.result() for f in futures]
```

## Monitoring and Observability

### Health Checks

**Kubernetes Health Checks:**
```yaml
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
```

**Custom Health Check Implementation:**
```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    })

@app.get("/ready")
async def readiness_check():
    """Readiness check - verify models are loaded."""
    try:
        # Check if models are accessible
        model_ready = check_model_availability()
        return JSONResponse({
            "status": "ready" if model_ready else "not_ready",
            "models_loaded": model_ready
        })
    except Exception as e:
        return JSONResponse(
            {"status": "not_ready", "error": str(e)}, 
            status_code=503
        )
```

### Metrics Collection

**Prometheus Metrics:**
```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
inference_requests = Counter('inference_requests_total', 'Total inference requests')
inference_duration = Histogram('inference_duration_seconds', 'Inference duration')
model_memory_usage = Gauge('model_memory_usage_bytes', 'Model memory usage')

# Use in application
@inference_duration.time()
def run_inference(image, question):
    inference_requests.inc()
    # Your inference code
    return result
```

### Logging Configuration

**Structured Logging:**
```python
import structlog

logger = structlog.get_logger()

# Log inference events
logger.info(
    "inference_completed",
    model="fast-vlm-base",
    duration_ms=187,
    memory_usage_mb=892,
    device_type="iPhone 15 Pro"
)
```

## Security Considerations

### API Security

**Authentication:**
```python
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_token(token: str = Depends(security)):
    # Implement token verification
    if not verify_api_key(token.credentials):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return token
```

**Rate Limiting:**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/inference")
@limiter.limit("100/minute")
async def inference(request: Request, ...):
    # Your inference logic
    pass
```

### Data Privacy

**Input Sanitization:**
```python
def sanitize_input(image_data: bytes, question: str) -> tuple:
    # Validate image data
    if len(image_data) > MAX_IMAGE_SIZE:
        raise ValueError("Image too large")
    
    # Sanitize question text
    question = question.strip()[:MAX_QUESTION_LENGTH]
    
    return image_data, question
```

## Scaling Strategies

### Horizontal Scaling

**Load Balancing:**
```yaml
# nginx.conf
upstream fast_vlm_backend {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
}

server {
    listen 80;
    location / {
        proxy_pass http://fast_vlm_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Auto-scaling Configuration

**HPA Configuration:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fast-vlm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fast-vlm-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Disaster Recovery

### Backup Strategy

**Model Backup:**
```bash
#!/bin/bash
# backup-models.sh

BACKUP_DIR="/backups/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Backup model files
rsync -av /models/ "$BACKUP_DIR/models/"

# Backup configuration
rsync -av /config/ "$BACKUP_DIR/config/"

# Upload to cloud storage
aws s3 sync "$BACKUP_DIR" s3://your-backup-bucket/fast-vlm/
```

### Recovery Procedures

**Model Recovery:**
```bash
#!/bin/bash
# restore-models.sh

RESTORE_DATE="$1"
if [ -z "$RESTORE_DATE" ]; then
    echo "Usage: $0 YYYYMMDD"
    exit 1
fi

# Download from cloud storage
aws s3 sync s3://your-backup-bucket/fast-vlm/$RESTORE_DATE /tmp/restore/

# Restore models
rsync -av /tmp/restore/models/ /models/

# Restart services
kubectl rollout restart deployment/fast-vlm-deployment
```

## Troubleshooting

### Common Issues

**Memory Issues:**
- Reduce batch size
- Enable model quantization
- Use smaller model variants

**Performance Issues:**
- Check CPU/GPU utilization
- Monitor memory usage
- Verify network latency

**iOS Deployment Issues:**
- Verify code signing certificates
- Check device compatibility
- Validate model format

### Debug Tools

**iOS Debugging:**
```swift
// Enable Core ML debugging
let config = MLModelConfiguration()
config.allowLowPrecisionAccumulationOnGPU = true

// Add performance logging
let startTime = CFAbsoluteTimeGetCurrent()
let result = try await vlm.answer(image: image, question: question)
let duration = CFAbsoluteTimeGetCurrent() - startTime
print("Inference took \(duration * 1000)ms")
```

**Python Debugging:**
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Profile performance
import cProfile
cProfile.run('run_inference(image, question)', 'profile_stats')
```

This deployment guide provides comprehensive coverage for production deployment scenarios across iOS, Python, and containerized environments.