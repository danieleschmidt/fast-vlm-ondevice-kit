# GitHub Actions Workflows Implementation Manual

This manual provides step-by-step instructions for implementing the FastVLM On-Device Kit CI/CD workflows.

## Overview

Due to GitHub App permission limitations, the workflow files must be manually created by repository maintainers. This document provides complete implementation instructions.

## Required Setup

### 1. Repository Secrets

Configure the following secrets in GitHub Settings > Secrets and variables > Actions:

#### Production Deployment
```
PYPI_API_TOKEN          # PyPI publishing token
TEST_PYPI_API_TOKEN     # TestPyPI token for prereleases
HOMEBREW_TAP_TOKEN      # GitHub token for Homebrew formula updates
```

#### Security Scanning
```
SNYK_TOKEN              # Snyk security scanning token
CODECOV_TOKEN           # Codecov coverage reporting token
```

#### Notifications
```
SLACK_WEBHOOK_URL       # Slack webhook for CI/CD notifications
DISCORD_WEBHOOK_URL     # Optional Discord webhook
```

### 2. Repository Settings

#### Enable GitHub Pages
1. Go to Settings > Pages
2. Set Source to "GitHub Actions"
3. Enable GitHub Pages for documentation deployment

#### Branch Protection Rules
Create protection rules for `main` branch:
- Require pull request reviews
- Require status checks to pass
- Include administrators
- Restrict pushes to specific branches

#### Security Settings
1. Enable Dependabot alerts
2. Enable secret scanning
3. Enable code scanning (CodeQL)

## Workflow Implementation

### Step 1: Create Workflow Directory

```bash
mkdir -p .github/workflows
```

### Step 2: Copy Workflow Files

Copy the following files from `docs/workflows/examples/` to `.github/workflows/`:

1. **ci.yml** - Continuous Integration
   ```bash
   cp docs/workflows/examples/ci.yml .github/workflows/ci.yml
   ```

2. **cd.yml** - Continuous Deployment
   ```bash
   cp docs/workflows/examples/cd.yml .github/workflows/cd.yml
   ```

3. **security-scan.yml** - Security Scanning
   ```bash
   cp docs/workflows/examples/security-scan.yml .github/workflows/security-scan.yml
   ```

### Step 3: Environment-Specific Configuration

#### For Private Repositories
Add to workflow files:
```yaml
env:
  PRIVATE_REPO: true
```

#### For Public Repositories
Ensure artifact retention settings:
```yaml
- name: Upload artifacts
  uses: actions/upload-artifact@v3
  with:
    retention-days: 30  # Adjust as needed
```

### Step 4: Validate Workflow Syntax

Before committing, validate workflow syntax:

```bash
# Install act for local testing (optional)
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Validate workflow syntax
act --dryrun
```

## Workflow Customization

### CI Workflow Customization

#### Python Version Matrix
Modify the test matrix in `ci.yml`:
```yaml
strategy:
  matrix:
    python-version: ['3.10', '3.11', '3.12']  # Add/remove versions
```

#### Test Categories
Enable/disable test categories by modifying:
```yaml
- name: Run tests
  run: |
    pytest tests/ -v \
      -m "not slow" \              # Skip slow tests
      --cov=src/fast_vlm_ondevice \
      --cov-fail-under=85          # Adjust coverage threshold
```

#### iOS Testing
For repositories without iOS components:
```yaml
# Comment out or remove the test-ios job
# test-ios:
#   name: iOS/Swift Tests
#   runs-on: macos-latest
```

### CD Workflow Customization

#### Release Triggers
Modify release triggers in `cd.yml`:
```yaml
on:
  push:
    tags: [ 'v*', 'release-*' ]  # Customize tag patterns
```

#### Deployment Environments
Add staging deployment:
```yaml
deploy-staging:
  name: Deploy to Staging
  runs-on: ubuntu-latest
  environment: staging
  if: github.ref == 'refs/heads/develop'
```

### Security Scan Customization

#### Scan Frequency
Modify the schedule in `security-scan.yml`:
```yaml
schedule:
  - cron: '0 3 * * 1'  # Weekly on Monday at 3 AM
```

#### Security Tools
Enable/disable specific tools:
```yaml
# Disable Snyk if token not available
# - name: Run Snyk Container scan
#   if: ${{ env.SNYK_TOKEN != '' }}
```

## Advanced Configuration

### 1. Multi-Environment Deployment

Create environment-specific workflow files:

#### staging.yml
```yaml
name: Deploy to Staging
on:
  push:
    branches: [ develop ]
environment: staging
```

#### production.yml
```yaml
name: Deploy to Production
on:
  release:
    types: [ published ]
environment: production
```

### 2. Performance Testing

Add performance regression testing:
```yaml
performance-test:
  name: Performance Regression Test
  runs-on: ubuntu-latest
  steps:
  - name: Run benchmarks
    run: |
      pytest tests/performance/ \
        --benchmark-compare=baseline \
        --benchmark-compare-fail=mean:5%
```

### 3. Cross-Platform Testing

Extend the test matrix:
```yaml
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest, windows-latest]
    python-version: ['3.10', '3.11', '3.12']
    exclude:
      - os: windows-latest
        python-version: '3.12'  # Exclude if needed
```

## Monitoring and Maintenance

### 1. Workflow Health Monitoring

Create a workflow to monitor CI/CD health:
```yaml
name: Workflow Health Check
on:
  schedule:
    - cron: '0 6 * * 1'  # Weekly Monday 6 AM
jobs:
  health-check:
    runs-on: ubuntu-latest
    steps:
    - name: Check workflow success rates
      run: |
        # Script to analyze workflow success rates
        # Alert if success rate drops below threshold
```

### 2. Dependency Updates

Automate workflow dependency updates:
```yaml
name: Update Workflow Dependencies
on:
  schedule:
    - cron: '0 4 1 * *'  # Monthly on 1st at 4 AM
jobs:
  update-actions:
    runs-on: ubuntu-latest
    steps:
    - name: Update GitHub Actions
      uses: ActionsDesk/github-actions-updater@main
```

### 3. Cost Optimization

Monitor and optimize workflow costs:
- Use appropriate runner sizes
- Cache dependencies effectively
- Skip unnecessary jobs for draft PRs
- Implement job parallelization

## Troubleshooting

### Common Issues

#### 1. Permission Errors
```yaml
permissions:
  contents: read
  security-events: write
  pages: write
  id-token: write
```

#### 2. Secret Access
```yaml
# Check if secret is available
- name: Check secrets
  run: |
    if [ -z "${{ secrets.PYPI_API_TOKEN }}" ]; then
      echo "PYPI_API_TOKEN not set"
      exit 1
    fi
```

#### 3. iOS Build Issues
```yaml
# Use specific Xcode version
- name: Set up Xcode
  uses: maxim-lobanov/setup-xcode@v1
  with:
    xcode-version: '15.0'
```

### Debugging Workflows

#### Enable Debug Logging
```yaml
env:
  ACTIONS_STEP_DEBUG: true
  ACTIONS_RUNNER_DEBUG: true
```

#### Local Testing with Act
```bash
# Test specific job
act -j test-python

# Test with specific event
act push -e event.json
```

## Security Considerations

### 1. Workflow Security
- Use pinned action versions
- Avoid exposing secrets in logs
- Implement proper permission controls
- Use trusted actions only

### 2. Artifact Security
- Encrypt sensitive artifacts
- Set appropriate retention periods
- Limit artifact access

### 3. Environment Protection
- Require manual approval for production
- Implement environment-specific secrets
- Use deployment gates

## Support and Maintenance

### Documentation Updates
- Update this manual when workflows change
- Document custom modifications
- Maintain troubleshooting guide

### Version Management
- Tag workflow versions
- Maintain changelog for workflow updates
- Test workflow changes in forks

---

For additional support:
- Review GitHub Actions documentation
- Check workflow run logs
- Open issues in the repository
- Consult team documentation

**Last Updated**: January 2025  
**Version**: 1.0