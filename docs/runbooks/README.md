# FastVLM Operational Runbooks

This directory contains operational procedures and runbooks for managing FastVLM On-Device Kit in production environments.

## Available Runbooks

- **[Incident Response](incident-response.md)** - Emergency response procedures
- **[Performance Issues](performance-troubleshooting.md)** - Diagnosing performance problems
- **[Model Deployment](model-deployment.md)** - Safe model deployment procedures
- **[Monitoring Setup](monitoring-setup.md)** - Observability configuration
- **[Backup & Recovery](backup-recovery.md)** - Data protection procedures

## Quick Reference

### Emergency Contacts
- **Primary Oncall**: See PagerDuty rotation
- **Security Issues**: security@yourdomain.com
- **Infrastructure**: infrastructure@yourdomain.com

### Key Metrics
- **Inference Latency**: <250ms (P95)
- **Memory Usage**: <1GB peak
- **Model Accuracy**: >70% VQAv2
- **Error Rate**: <1%

### Common Commands
```bash
# Check system health
make health-check

# Run diagnostics
python scripts/diagnostics.py

# View logs
kubectl logs -f deployment/fast-vlm-api

# Scale deployment
kubectl scale deployment/fast-vlm-api --replicas=5
```