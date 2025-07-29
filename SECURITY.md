# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. Please report security vulnerabilities via:

**Email**: fast-vlm@yourdomain.com

**Please include:**
- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Suggested fix (if known)

**Response Timeline:**
- Initial response: Within 48 hours
- Triage completion: Within 7 days
- Fix timeline: Depends on severity

## Security Considerations

### Model Security
- Models are executed in sandboxed Core ML runtime
- No network access during inference
- Input validation for images and text

### Data Privacy
- All processing happens on-device
- No data transmitted to external servers
- User images never leave the device

### Code Security
- Dependencies scanned for vulnerabilities
- Pre-commit hooks check for secrets
- Container images security-scanned

## Responsible Disclosure

We follow responsible disclosure practices:
1. Report received and acknowledged
2. Investigation and fix development
3. Coordinated public disclosure
4. Credit to reporter (if desired)

Thank you for helping keep Fast VLM On-Device Kit secure!