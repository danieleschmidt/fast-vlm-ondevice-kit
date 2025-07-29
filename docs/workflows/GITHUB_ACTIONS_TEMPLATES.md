# GitHub Actions Workflow Templates

This document provides complete GitHub Actions workflow templates for the Fast VLM On-Device Kit project. These templates should be saved in `.github/workflows/` directory.

**Note:** As per repository policy, workflow files cannot be automatically created. These templates should be manually implemented by the maintainer.

## 1. Python Testing Workflow

**File:** `.github/workflows/python-tests.yml`

```yaml
name: Python Tests

on:
  push:
    branches: [ main, develop ]
    paths:
      - 'src/**'
      - 'tests/**'
      - 'requirements*.txt'
      - 'pyproject.toml'
      - '.github/workflows/python-tests.yml'
  pull_request:
    branches: [ main ]
    paths:
      - 'src/**'
      - 'tests/**'
      - 'requirements*.txt'
      - 'pyproject.toml'
  workflow_dispatch:

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ['3.10', '3.11', '3.12']
        exclude:
          # Core ML tools only work on macOS, so limit some combinations
          - os: ubuntu-latest
            python-version: '3.12'

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        cache: 'pip'

    - name: Install system dependencies (macOS)
      if: runner.os == 'macOS'
      run: |
        # Install any macOS-specific dependencies for Core ML
        brew install --quiet --no-install libomp

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"

    - name: Create reports directory
      run: mkdir -p reports

    - name: Run tests with coverage
      run: |
        pytest tests/ -v \
          --cov=src/fast_vlm_ondevice \
          --cov-report=xml:reports/coverage.xml \
          --cov-report=html:reports/htmlcov \
          --cov-report=term-missing \
          --junitxml=reports/junit.xml \
          --cov-fail-under=75

    - name: Upload coverage to Codecov
      if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.11'
      uses: codecov/codecov-action@v3
      with:
        file: reports/coverage.xml
        flags: python
        name: codecov-umbrella
        fail_ci_if_error: false

    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results-${{ matrix.os }}-${{ matrix.python-version }}
        path: reports/

  benchmark:
    runs-on: macos-latest
    if: github.event_name == 'pull_request'
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        cache: 'pip'

    - name: Install dependencies
      run: |
        pip install -e ".[dev]"

    - name: Run performance benchmarks
      run: |
        pytest tests/performance/ -v \
          -m "performance" \
          --benchmark-save=pr-${{ github.event.number }}

    - name: Compare with baseline
      run: |
        # Compare with baseline benchmarks if they exist
        pytest tests/performance/ \
          --benchmark-compare=baseline \
          --benchmark-compare-fail=min:5% || true

    - name: Upload benchmark results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: .benchmarks/
```

## 2. Code Quality Workflow

**File:** `.github/workflows/code-quality.yml`

```yaml
name: Code Quality

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        cache: 'pip'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"

    - name: Run black
      run: black --check --diff src/ tests/

    - name: Run isort
      run: isort --check-only --diff src/ tests/

    - name: Run flake8
      run: flake8 src/ tests/

    - name: Run mypy
      run: mypy src/ --strict

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install bandit[toml] safety

    - name: Create reports directory
      run: mkdir -p reports

    - name: Run bandit security scan
      run: |
        bandit -r src/ -f json -o reports/bandit.json || true
        bandit -r src/ -f txt

    - name: Run safety check
      run: |
        safety check --json --output reports/safety.json || true
        safety check

    - name: Upload security reports
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: security-reports
        path: reports/

  pre-commit:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install pre-commit
      run: pip install pre-commit

    - name: Run pre-commit on all files
      run: pre-commit run --all-files --show-diff-on-failure
```

## 3. iOS/Swift Testing Workflow

**File:** `.github/workflows/ios-tests.yml`

```yaml
name: iOS Tests

on:
  push:
    branches: [ main, develop ]
    paths:
      - 'ios/**'
      - '.github/workflows/ios-tests.yml'
  pull_request:
    branches: [ main ]
    paths:
      - 'ios/**'
  workflow_dispatch:

jobs:
  swift-test:
    runs-on: macos-latest
    steps:
    - uses: actions/checkout@v4

    - name: Setup Xcode
      uses: maxim-lobanov/setup-xcode@v1
      with:
        xcode-version: latest-stable

    - name: Swift version
      run: swift --version

    - name: Resolve Swift package dependencies
      run: |
        cd ios
        swift package resolve

    - name: Build Swift package
      run: |
        cd ios
        swift build

    - name: Run Swift tests
      run: |
        cd ios
        swift test --enable-code-coverage

    - name: Generate test coverage
      run: |
        cd ios
        xcrun llvm-cov export -format="lcov" \
          .build/debug/FastVLMKitPackageTests.xctest/Contents/MacOS/FastVLMKitPackageTests \
          -instr-profile=.build/debug/codecov/default.profdata \
          > coverage.lcov

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ios/coverage.lcov
        flags: swift
        fail_ci_if_error: false

  swiftlint:
    runs-on: macos-latest
    steps:
    - uses: actions/checkout@v4

    - name: SwiftLint
      uses: norio-nomura/action-swiftlint@3.2.1
      with:
        args: --strict --path ios/
```

## 4. Documentation Workflow

**File:** `.github/workflows/docs.yml`

```yaml
name: Documentation

on:
  push:
    branches: [ main ]
    paths:
      - 'docs/**'
      - 'src/**'
      - '.github/workflows/docs.yml'
  pull_request:
    branches: [ main ]
    paths:
      - 'docs/**'
      - 'src/**'
  workflow_dispatch:

jobs:
  build-docs:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
        pip install sphinx sphinx-rtd-theme myst-parser

    - name: Build documentation
      run: |
        cd docs
        sphinx-build -W -b html . _build/html

    - name: Check for broken links
      run: |
        cd docs
        sphinx-build -W -b linkcheck . _build/linkcheck

    - name: Upload documentation artifacts
      uses: actions/upload-artifact@v3
      with:
        name: documentation
        path: docs/_build/html/

  deploy-docs:
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    needs: build-docs
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pages: write
      id-token: write
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
    - name: Download documentation
      uses: actions/download-artifact@v3
      with:
        name: documentation
        path: ./docs

    - name: Setup Pages
      uses: actions/configure-pages@v3

    - name: Upload to GitHub Pages
      uses: actions/upload-pages-artifact@v2
      with:
        path: './docs'

    - name: Deploy to GitHub Pages
      id: deployment
      uses: actions/deploy-pages@v2
```

## 5. Security Scanning Workflow

**File:** `.github/workflows/security.yml`

```yaml
name: Security Scanning

on:
  push:
    branches: [ main ]
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday at 6 AM UTC
  workflow_dispatch:

jobs:
  dependency-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install safety pip-audit

    - name: Run safety check
      run: |
        safety check --json --output safety-report.json || true
        safety check

    - name: Run pip-audit
      run: |
        pip-audit --format=json --output=pip-audit-report.json || true
        pip-audit

    - name: Upload security reports
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: security-dependency-reports
        path: |
          safety-report.json
          pip-audit-report.json

  codeql-analysis:
    name: CodeQL Analysis
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write

    strategy:
      fail-fast: false
      matrix:
        language: [ 'python', 'swift' ]

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Initialize CodeQL
      uses: github/codeql-action/init@v2
      with:
        languages: ${{ matrix.language }}

    - name: Setup Python (for Python analysis)
      if: matrix.language == 'python'
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install Python dependencies (for Python analysis)
      if: matrix.language == 'python'
      run: |
        pip install -e .

    - name: Autobuild
      uses: github/codeql-action/autobuild@v2

    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2

  secret-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - name: Run TruffleHog OSS
      uses: trufflesecurity/trufflehog@main
      with:
        path: ./
        base: main
        head: HEAD
        extra_args: --debug --only-verified
```

## 6. Release Workflow

**File:** `.github/workflows/release.yml`

```yaml
name: Release

on:
  push:
    tags:
      - 'v*.*.*'
  workflow_dispatch:
    inputs:
      version:
        description: 'Release version (e.g., v1.0.0)'
        required: true
        type: string

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -e ".[dev]"

    - name: Run full test suite
      run: |
        make check-ci

  build-python:
    needs: validate
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install build dependencies
      run: |
        pip install build twine

    - name: Build package
      run: |
        python -m build

    - name: Verify package
      run: |
        twine check dist/*

    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: python-package
        path: dist/

  build-ios:
    needs: validate
    runs-on: macos-latest
    steps:
    - uses: actions/checkout@v4

    - name: Setup Xcode
      uses: maxim-lobanov/setup-xcode@v1
      with:
        xcode-version: latest-stable

    - name: Build Swift package
      run: |
        cd ios
        swift build -c release

    - name: Run Swift tests
      run: |
        cd ios
        swift test

    - name: Create Swift package archive
      run: |
        cd ios
        swift package archive --output ../FastVLMKit.zip

    - name: Upload Swift package
      uses: actions/upload-artifact@v3
      with:
        name: swift-package
        path: FastVLMKit.zip

  create-release:
    needs: [build-python, build-ios]
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - name: Download Python package
      uses: actions/download-artifact@v3
      with:
        name: python-package
        path: dist/

    - name: Download Swift package
      uses: actions/download-artifact@v3
      with:
        name: swift-package

    - name: Generate release notes
      run: |
        # Extract version from tag or input
        VERSION=${GITHUB_REF#refs/tags/}
        if [ "${{ github.event_name }}" = "workflow_dispatch" ]; then
          VERSION="${{ github.event.inputs.version }}"
        fi
        
        # Generate changelog for this version
        echo "# Release $VERSION" > release_notes.md
        echo "" >> release_notes.md
        echo "## Changes" >> release_notes.md
        
        # Extract changelog section for this version
        sed -n "/## \[${VERSION#v}\]/,/## \[/p" CHANGELOG.md | head -n -1 >> release_notes.md || true

    - name: Create GitHub Release
      uses: softprops/action-gh-release@v1
      with:
        body_path: release_notes.md
        files: |
          dist/*
          FastVLMKit.zip
        draft: false
        prerelease: ${{ contains(github.ref, 'alpha') || contains(github.ref, 'beta') || contains(github.ref, 'rc') }}

  publish-pypi:
    needs: create-release
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v') && !contains(github.ref, 'alpha') && !contains(github.ref, 'beta')
    environment:
      name: pypi
      url: https://pypi.org/p/fast-vlm-ondevice
    permissions:
      id-token: write
    steps:
    - name: Download Python package
      uses: actions/download-artifact@v3
      with:
        name: python-package
        path: dist/

    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
```

## 7. Dependency Update Workflow

**File:** `.github/workflows/dependencies.yml`

```yaml
name: Dependency Updates

on:
  schedule:
    - cron: '0 4 * * 1'  # Weekly on Monday at 4 AM UTC
  workflow_dispatch:

jobs:
  update-python-deps:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install pip-tools
      run: pip install pip-tools

    - name: Update requirements
      run: |
        pip-compile --upgrade requirements.in
        pip-compile --upgrade requirements-dev.in

    - name: Create Pull Request
      uses: peter-evans/create-pull-request@v5
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
        commit-message: 'chore: update Python dependencies'
        title: 'Update Python Dependencies'
        body: |
          Automated dependency update
          
          This PR updates Python dependencies to their latest compatible versions.
          
          Please review the changes and ensure all tests pass before merging.
        branch: update-python-dependencies
        delete-branch: true

  update-swift-deps:
    runs-on: macos-latest
    steps:
    - uses: actions/checkout@v4

    - name: Update Swift package dependencies
      run: |
        cd ios
        swift package update

    - name: Check for changes
      id: verify-changed-files
      run: |
        if [ -n "$(git status ios/Package.resolved --porcelain)" ]; then
          echo "changed=true" >> $GITHUB_OUTPUT
        else
          echo "changed=false" >> $GITHUB_OUTPUT
        fi

    - name: Create Pull Request
      if: steps.verify-changed-files.outputs.changed == 'true'
      uses: peter-evans/create-pull-request@v5
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
        commit-message: 'chore: update Swift dependencies'
        title: 'Update Swift Dependencies'
        body: |
          Automated Swift dependency update
          
          This PR updates Swift package dependencies to their latest compatible versions.
          
          Please review the changes and ensure all tests pass before merging.
        branch: update-swift-dependencies
        delete-branch: true
```

## Repository Secrets Required

Add these secrets to your GitHub repository:

- `CODECOV_TOKEN`: For coverage reporting
- `PYPI_API_TOKEN`: For PyPI publishing (when ready)

## Branch Protection Rules

Configure these rules for the `main` branch:

1. **Require status checks to pass before merging**
   - `Python Tests / test (3.11, ubuntu-latest)`
   - `Python Tests / test (3.11, macos-latest)`
   - `Code Quality / lint`
   - `iOS Tests / swift-test`

2. **Require branches to be up to date before merging**

3. **Require review from code owners**

4. **Dismiss stale reviews when new commits are pushed**

5. **Restrict pushes that create files**

## Usage Instructions

1. Create the `.github/workflows/` directory in your repository
2. Copy each workflow template into separate `.yml` files
3. Customize the workflows based on your specific needs
4. Add required repository secrets
5. Configure branch protection rules
6. Test workflows with a pull request

These workflows provide comprehensive CI/CD coverage including testing, security scanning, documentation building, and automated releases.