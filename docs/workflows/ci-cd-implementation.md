# CI/CD Implementation Guide

This document provides the required GitHub Actions workflows for the Fast VLM On-Device Kit repository.

## Required Workflows

### 1. Continuous Integration (`ci.yml`)

```yaml
name: CI
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  python-tests:
    runs-on: macos-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      
      - name: Run tests
        run: make test
      
      - name: Run linting
        run: make lint
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  swift-tests:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - name: Test Swift package
        run: |
          cd ios
          swift test

  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Bandit security scan
        run: |
          pip install bandit
          bandit -r src/ -f json -o security-report.json
      
      - name: Upload security report
        uses: actions/upload-artifact@v3
        with:
          name: security-report
          path: security-report.json
```

### 2. Dependency Scanning (`dependency-scan.yml`)

```yaml
name: Dependency Scan
on:
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday
  push:
    paths: ['requirements*.txt', 'pyproject.toml', 'ios/Package.swift']

jobs:
  python-dependencies:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run safety check
        run: |
          pip install safety
          safety check -r requirements.txt
      
      - name: Run pip-audit
        run: |
          pip install pip-audit
          pip-audit
  
  swift-dependencies:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - name: Swift dependency audit
        run: |
          cd ios
          swift package show-dependencies
```

### 3. Performance Benchmarking (`benchmark.yml`)

```yaml
name: Performance Benchmark
on:
  push:
    branches: [main]
  schedule:
    - cron: '0 4 * * 1'  # Weekly benchmarks

jobs:
  benchmark:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup environment
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      
      - name: Run Python benchmarks
        run: |
          python -m pytest benchmarks/ --benchmark-json=benchmark-results.json
      
      - name: Upload benchmark results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmark-results.json
```

### 4. Release Automation (`release.yml`)

```yaml
name: Release
on:
  push:
    tags: ['v*']

jobs:
  release:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Build package
        run: |
          pip install build
          python -m build
      
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
      
      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          files: dist/*
          generate_release_notes: true
```

## Implementation Instructions

1. Create `.github/workflows/` directory in repository root
2. Add each workflow file with the specified content
3. Configure required secrets in repository settings:
   - `PYPI_API_TOKEN` for PyPI publishing
   - `CODECOV_TOKEN` for coverage reporting
4. Enable GitHub Actions in repository settings
5. Configure branch protection rules requiring CI to pass

## Additional Configurations

### Branch Protection
- Require PR reviews before merging
- Require status checks to pass
- Require branches to be up to date
- Dismiss stale reviews when new commits are pushed

### Secrets Management
- Store sensitive tokens in GitHub Secrets
- Use environment-specific secrets for staging/production
- Rotate secrets regularly
- Never commit secrets to code

### Monitoring
- Set up notification channels for failed builds
- Monitor benchmark performance over time
- Track dependency vulnerabilities
- Review security scan results regularly