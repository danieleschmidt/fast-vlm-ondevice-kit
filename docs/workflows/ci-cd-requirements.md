# CI/CD Workflow Requirements

This document outlines the required GitHub Actions workflows for the Fast VLM On-Device Kit project.

## Required Workflows

### 1. Python Testing Workflow

**File**: `.github/workflows/python-tests.yml`

**Triggers**:
- Push to main branch
- Pull requests to main
- Manual dispatch

**Jobs**:
- Test on Python 3.10, 3.11, 3.12
- Test on Ubuntu, macOS (for Core ML compatibility)
- Run pytest with coverage reporting
- Upload coverage to Codecov
- Cache pip dependencies

**Key Requirements**:
- Install Core ML tools on macOS runners
- Generate coverage reports
- Fail if coverage drops below 80%

### 2. Code Quality Workflow

**File**: `.github/workflows/code-quality.yml`

**Triggers**:
- Push to main branch
- Pull requests to main

**Jobs**:
- Black code formatting check
- isort import sorting check
- flake8 linting
- mypy type checking
- bandit security scanning
- pre-commit hook validation

### 3. iOS/Swift Testing Workflow  

**File**: `.github/workflows/ios-tests.yml`

**Triggers**:
- Push to main branch
- Pull requests affecting ios/ directory
- Manual dispatch

**Jobs**:
- Swift package build verification
- Unit test execution
- SwiftLint style checking
- Xcode project validation
- iOS simulator testing

**Requirements**:
- Use macOS runners (required for iOS builds)
- Test on multiple iOS versions (17.0+)
- Generate test reports

### 4. Documentation Workflow

**File**: `.github/workflows/docs.yml`

**Triggers**:
- Push to main branch
- Pull requests affecting docs/

**Jobs**:
- Build Sphinx documentation
- Check for broken links
- Deploy to GitHub Pages (on main branch)
- Generate API documentation

### 5. Security Scanning Workflow

**File**: `.github/workflows/security.yml`

**Triggers**:
- Push to main branch
- Schedule: weekly on Sundays
- Manual dispatch

**Jobs**:
- Dependency vulnerability scanning (Safety)
- Code security analysis (Bandit)
- Container security scanning (if Docker images)
- SBOM generation
- Secret detection

### 6. Release Workflow

**File**: `.github/workflows/release.yml`

**Triggers**:
- Tag creation (v*.*.*)
- Manual dispatch

**Jobs**:
- Build Python package
- Build iOS Swift package
- Run full test suite
- Generate release notes
- Upload to PyPI (when ready)
- Create GitHub release

## Workflow Integration Requirements

### Branch Protection Rules

**Main Branch Protection**:
- Require status checks to pass
- Require branches to be up to date
- Require review from code owners
- Dismiss stale reviews on new commits
- Restrict force pushes

**Required Status Checks**:
- `python-tests / test (3.10, ubuntu-latest)`
- `python-tests / test (3.11, macos-latest)`
- `code-quality / lint`
- `ios-tests / build-and-test`

### Secrets Configuration

**Required Repository Secrets**:
- `CODECOV_TOKEN`: For coverage reporting
- `PYPI_API_TOKEN`: For package publishing (when ready)

**Optional Secrets**:
- `SLACK_WEBHOOK`: For deployment notifications
- `APPLE_DEVELOPER_ID`: For iOS code signing (if needed)

### Environment Configuration

**Development Environment**:
- Automatic deployment for documentation
- PR preview deployments

**Production Environment**:
- Manual approval required
- Deploy only from main branch
- Comprehensive testing required

## Workflow File Templates

### Python Test Template Structure

```yaml
name: Python Tests
on: [push, pull_request]
jobs:
  test:
    strategy:
      matrix:
        python-version: [3.10, 3.11, 3.12]
        os: [ubuntu-latest, macos-latest]
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
      - name: Install dependencies
        run: make install
      - name: Run tests
        run: make test
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Code Quality Template Structure

```yaml
name: Code Quality
on: [push, pull_request]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
      - name: Install dependencies
        run: pip install pre-commit
      - name: Run pre-commit
        run: pre-commit run --all-files
```

### iOS Test Template Structure

```yaml
name: iOS Tests
on: [push, pull_request]
jobs:
  ios-test:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Swift Package
        run: cd ios && swift build
      - name: Run Swift Tests
        run: cd ios && swift test
```

## Performance and Optimization

### Caching Strategy

**Python Dependencies**:
- Cache pip packages by requirements hash
- Cache pre-commit environments
- Cache mypy cache directory

**iOS Dependencies**:
- Cache Swift Package Manager dependencies
- Cache Xcode derived data (if applicable)

### Parallel Execution

- Run Python and iOS tests in parallel
- Use matrix strategy for multiple Python versions
- Parallelize linting jobs when possible

### Resource Management

- Use appropriate runner sizes
- Optimize for build time vs. cost
- Clean up temporary files and caches

## Monitoring and Notifications

### Success Metrics

- Test coverage percentage
- Build success rate
- Average CI runtime
- Security vulnerability count

### Notification Strategy

- Slack notifications for failures on main
- Email notifications for security issues
- GitHub status updates for PR checks
- Release notifications to team channels

## Security Considerations

### Workflow Security

- Use pinned action versions
- Limit workflow permissions
- No secrets in PR workflows from forks
- Secure artifact handling

### Code Security

- Scan dependencies for vulnerabilities
- Check for hardcoded secrets
- Validate container images
- Monitor for malicious code changes

## Documentation Integration

- Auto-generate API documentation
- Update documentation on releases
- Validate documentation builds
- Check for broken links and references

This CI/CD setup ensures comprehensive testing, quality control, and secure deployment for the Fast VLM On-Device Kit project.