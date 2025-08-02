# FastVLM On-Device Kit - Operational Runbooks

This directory contains operational runbooks for troubleshooting and managing FastVLM systems.

## Available Runbooks

### Performance Issues
- [Slow Model Conversion](slow-conversion.md) - Troubleshooting long conversion times
- [High Inference Latency](high-latency.md) - Dealing with slow inference performance
- [Memory Issues](high-memory.md) - Managing memory usage and leaks

### Error Management
- [High Error Rate](high-error-rate.md) - Investigating error spikes
- [Service Downtime](service-down.md) - Handling service outages

### Monitoring & Alerts
- [Alert Response Guide](alert-response.md) - How to respond to monitoring alerts
- [Metrics Collection](metrics-troubleshooting.md) - Fixing metrics collection issues

## General Troubleshooting Steps

### 1. Initial Assessment
```bash
# Check service health
make health-check

# Review recent logs
docker logs fastvm-app --tail=100

# Check system resources
htop
df -h
```

### 2. Metrics Review
- Access Grafana dashboard: http://localhost:3000
- Check Prometheus metrics: http://localhost:9090
- Review alert status in AlertManager

### 3. Common Commands
```bash
# Restart services
docker-compose restart

# Scale converter workers
docker-compose up --scale converter=3

# View detailed logs
docker-compose logs -f fastvm-app

# Run diagnostics
python scripts/diagnostics.py
```

## Escalation Procedures

### Severity Levels
- **Critical**: Service unavailable, data loss risk
- **Warning**: Performance degradation, recoverable errors
- **Info**: Normal operational events

### Contact Information
- **On-call**: Slack #fastvm-alerts
- **Engineering**: GitHub Issues
- **Documentation**: Update these runbooks after incidents

## Monitoring Dashboard Access

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **AlertManager**: http://localhost:9093

## Post-Incident Actions

1. Document the incident
2. Update runbooks with new learnings
3. Review and improve monitoring
4. Conduct blameless post-mortem if needed