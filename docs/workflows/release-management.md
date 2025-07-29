# Release Management Automation

## Overview

Comprehensive release automation system for Fast VLM On-Device Kit, implementing automated versioning, changelog generation, and release workflows for MATURING SDLC environments.

## Required GitHub Actions Workflow

Create `.github/workflows/release.yml`:

```yaml
name: Release Automation
on:
  workflow_dispatch:
    inputs:
      version_bump:
        description: 'Version bump type'
        required: true
        default: 'patch'
        type: choice
        options:
          - patch
          - minor
          - major
      dry_run:
        description: 'Dry run (no changes)'
        required: false
        default: false
        type: boolean

jobs:
  release:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
      
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          token: ${{ secrets.GITHUB_TOKEN }}
          
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install build twine
          
      - name: Configure Git
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          
      - name: Run Release Automation
        run: |
          python scripts/release_automation.py ${{ inputs.version_bump }} \
            ${{ inputs.dry_run && '--dry-run' || '' }}
            
      - name: Push changes
        if: ${{ !inputs.dry_run }}
        run: |
          git push origin main
          git push origin --tags
          
      - name: Create GitHub Release
        if: ${{ !inputs.dry_run }}
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: v${{ env.NEW_VERSION }}
          release_name: Release v${{ env.NEW_VERSION }}
          body_path: release-notes.md
          draft: false
          prerelease: false
          
      - name: Publish to PyPI
        if: ${{ !inputs.dry_run }}
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: |
          python -m build
          twine upload dist/*
```

## Automated Release Workflow

### 1. Version Management

```python
# Semantic versioning automation
def determine_version_bump(commits: List[str]) -> str:
    breaking_changes = any('BREAKING CHANGE' in commit for commit in commits)
    features = any(commit.startswith('feat:') for commit in commits)
    
    if breaking_changes:
        return 'major'
    elif features:
        return 'minor'
    else:
        return 'patch'
```

### 2. Changelog Generation

```python
# Automatic changelog from commits
changelog_config = {
    'feat': '🚀 New Features',
    'fix': '🐛 Bug Fixes',
    'docs': '📚 Documentation', 
    'perf': '⚡ Performance',
    'refactor': '🔧 Maintenance',
    'test': '🧪 Testing',
    'ci': '🔄 CI/CD',
    'style': '💄 Styling',
    'chore': '🏠 Chores'
}
```

### 3. Release Validation

Pre-release checks:
- All tests pass
- Code quality metrics meet thresholds
- Security scans pass
- Documentation is up to date
- No uncommitted changes
- Branch is up to date with main

## Multi-Platform Release Coordination

### 1. Python Package Release

```yaml
# PyPI publishing workflow
- name: Build Python Package
  run: |
    python -m build
    twine check dist/*
    
- name: Publish to PyPI
  if: github.event_name == 'release'
  env:
    TWINE_USERNAME: __token__  
    TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
  run: twine upload dist/*
```

### 2. Swift Package Release

```yaml
# Swift package validation
- name: Validate Swift Package
  run: |
    cd ios
    swift package resolve
    swift build
    swift test
    
- name: Update Package Registry
  if: github.event_name == 'release'
  run: |
    # Swift package registry update
    swift package-registry publish
```

### 3. Container Image Release

```yaml
# Docker image publishing
- name: Build and Push Docker Images
  uses: docker/build-push-action@v4
  with:
    context: .
    push: true
    tags: |
      fastvlm/ondevice-kit:latest
      fastvlm/ondevice-kit:${{ env.VERSION }}
    platforms: linux/amd64,linux/arm64
```

## Release Notes Automation

### 1. Automated Release Notes Generation

```python
class ReleaseNotesGenerator:
    def generate_release_notes(self, version: str, commits: List[Dict]) -> str:
        template = """
## What's Changed in v{version}

{highlights}

### 📊 Performance Improvements
{performance_changes}

### 🔧 Technical Changes
{technical_changes}

### 🐛 Bug Fixes
{bug_fixes}

### 📚 Documentation Updates
{doc_updates}

**Full Changelog**: {changelog_url}
**Installation**: `pip install fast-vlm-ondevice=={version}`
        """.strip()
        
        return template.format(
            version=version,
            highlights=self.extract_highlights(commits),
            performance_changes=self.filter_commits(commits, 'perf'),
            technical_changes=self.filter_commits(commits, 'refactor'),
            bug_fixes=self.filter_commits(commits, 'fix'),
            doc_updates=self.filter_commits(commits, 'docs'),
            changelog_url=f"https://github.com/repo/compare/v{prev_version}...v{version}"
        )
```

### 2. Release Asset Generation

```python
# Generate release assets
assets_to_generate = [
    'fast-vlm-ondevice-{version}.tar.gz',  # Source distribution
    'fast-vlm-ondevice-{version}-py3-none-any.whl',  # Python wheel
    'FastVLMKit-{version}.zip',  # Swift package
    'models-{version}.zip',  # Pre-trained models
    'benchmarks-{version}.json',  # Performance benchmarks
    'SBOM-{version}.json'  # Software Bill of Materials
]
```

## Hotfix Release Process

### 1. Automated Hotfix Workflow

```yaml
name: Hotfix Release
on:
  workflow_dispatch:
    inputs:
      hotfix_branch:
        description: 'Hotfix branch name'
        required: true
      severity:
        description: 'Severity level'
        type: choice
        options:
          - critical
          - high
          - medium

jobs:
  hotfix:
    if: startsWith(github.ref, 'refs/heads/hotfix/')
    runs-on: ubuntu-latest
    steps:
      - name: Emergency Release Process
        run: |
          # Skip some checks for critical hotfixes
          if [ "${{ inputs.severity }}" = "critical" ]; then
            echo "SKIP_SLOW_TESTS=true" >> $GITHUB_ENV
          fi
```

### 2. Rollback Automation

```python
class ReleaseRollback:
    def rollback_release(self, version: str) -> bool:
        """Automated release rollback"""
        try:
            # Revert git tag
            subprocess.run(['git', 'tag', '-d', f'v{version}'], check=True)
            subprocess.run(['git', 'push', 'origin', f':refs/tags/v{version}'], check=True)
            
            # Remove from PyPI (manual intervention required)
            logger.warning(f"Manual PyPI removal required for version {version}")
            
            # Update documentation
            self.update_rollback_documentation(version)
            
            return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
```

## Release Metrics and Analytics

### 1. Release Health Monitoring

```python
release_metrics = {
    'deployment_frequency': 'weekly',
    'lead_time_for_changes': '2_days',
    'change_failure_rate': '5%',
    'time_to_restore_service': '4_hours'
}

def track_release_metrics(version: str, deployment_time: datetime):
    """Track DORA metrics for releases"""
    metrics = {
        'version': version,
        'deployment_time': deployment_time.isoformat(),
        'lead_time': calculate_lead_time(version),
        'commit_count': get_commit_count_since_last_release(),
        'test_coverage': get_test_coverage(),
        'security_scan_results': get_security_scan_results()
    }
    
    # Send to monitoring system
    send_to_datadog(metrics)
```

### 2. Release Success Criteria

```yaml
# Release gates configuration
release_gates:
  required_checks:
    - all_tests_pass
    - security_scan_clean
    - performance_regression_check
    - documentation_updated
    
  approval_requirements:
    - technical_lead_approval
    - security_team_approval  # for major releases
    - product_owner_approval  # for feature releases
    
  automatic_rollback_triggers:
    - error_rate_spike: "> 5%"
    - performance_degradation: "> 20%"
    - critical_security_alert: true
```

## Integration with External Systems

### 1. Slack Notifications

```python
def send_release_notification(version: str, changelog: str):
    """Send release notification to Slack"""
    webhook_url = os.getenv('SLACK_RELEASE_WEBHOOK')
    
    message = {
        "text": f"🚀 Fast VLM v{version} Released!",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Fast VLM v{version}* has been successfully released!"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn", 
                    "text": f"```{changelog[:500]}...```"
                }
            }
        ]
    }
    
    requests.post(webhook_url, json=message)
```

### 2. JIRA Integration

```python
def update_jira_issues(version: str, commits: List[str]):
    """Update JIRA issues mentioned in commits"""
    jira_pattern = r'[A-Z]+-\d+'
    
    for commit in commits:
        issues = re.findall(jira_pattern, commit)
        for issue_key in issues:
            update_jira_issue(issue_key, {
                'fixVersion': version,
                'status': 'Released'
            })
```

## Security Considerations

### 1. Signed Releases

```yaml
- name: Sign Release Assets
  uses: sigstore/gh-action-sigstore-python@v1.2.0
  with:
    inputs: dist/*
    
- name: Generate SLSA Provenance
  uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v1.9.0
  with:
    base64-subjects: ${{ env.HASHES }}
```

### 2. Supply Chain Security

```python
# Generate Software Bill of Materials
def generate_sbom(version: str) -> Dict:
    """Generate SPDX-compliant SBOM"""
    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": f"SPDXRef-FastVLM-{version}",
        "name": f"fast-vlm-ondevice-{version}",
        "packages": [
            # Include all dependencies with versions and licenses
        ],
        "relationships": [
            # Define dependency relationships
        ]
    }
```

## Best Practices

### 1. Release Branching Strategy
- Use `main` branch for releases
- Create `hotfix/` branches for emergency fixes
- Tag releases with `v{version}` format
- Maintain release branches for long-term support

### 2. Documentation Updates
- Automatically update API documentation
- Generate migration guides for breaking changes
- Update installation instructions
- Refresh example code and tutorials

### 3. Backward Compatibility
- Maintain API compatibility in minor versions
- Provide deprecation warnings before removal
- Document breaking changes clearly
- Offer migration paths for major version updates

### 4. Testing Strategy
- Run full test suite before release
- Execute integration tests on multiple platforms
- Validate example applications
- Perform manual testing for critical paths

## References

- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
- [GitHub Release Automation](https://docs.github.com/en/actions/publishing-packages)
- [SLSA Supply Chain Security](https://slsa.dev/)