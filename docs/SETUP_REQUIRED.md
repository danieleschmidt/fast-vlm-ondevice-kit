# Manual Setup Required for Full SDLC Implementation

Due to GitHub App permission limitations, some components of the SDLC implementation require manual setup by repository maintainers.

## Required GitHub Permissions

The GitHub App lacks the following permissions needed for complete automation:

### Repository Settings
- **Branch protection rules**: Must be configured manually
- **Repository settings**: Description, topics, and homepage updates
- **Secrets management**: CI/CD secrets and tokens

### Workflow Creation
- **GitHub Actions workflows**: Cannot create workflow files directly
- **Workflow permissions**: May need adjustment for security scanning

## Manual Setup Instructions

### 1. GitHub Actions Workflows

Copy workflow files from `docs/workflows/examples/` to `.github/workflows/`:

```bash
mkdir -p .github/workflows
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/cd.yml .github/workflows/
cp docs/workflows/examples/security-scan.yml .github/workflows/
```

Refer to `docs/workflows/IMPLEMENTATION_MANUAL.md` for detailed setup instructions.

### 2. Repository Secrets

Configure the following secrets in Repository Settings > Secrets and variables > Actions:

```
PYPI_API_TOKEN          # For PyPI package publishing
TEST_PYPI_API_TOKEN     # For TestPyPI prereleases
SNYK_TOKEN              # For security scanning
CODECOV_TOKEN           # For coverage reporting
SLACK_WEBHOOK_URL       # For CI/CD notifications
HOMEBREW_TAP_TOKEN      # For Homebrew formula updates
```

### 3. Branch Protection Rules

Set up branch protection for `main` branch:
- Require pull request reviews (minimum 1)
- Require status checks to pass before merging
- Require branches to be up to date before merging
- Include administrators in restrictions
- Restrict pushes to main branch

### 4. Security Settings

Enable the following security features:
- Dependabot alerts and security updates
- Secret scanning alerts
- Code scanning with CodeQL
- Private vulnerability reporting

### 5. GitHub Pages

For documentation deployment:
- Go to Repository Settings > Pages
- Set Source to "GitHub Actions"
- Configure custom domain if desired

### 6. Repository Settings

Update repository metadata:
- Description: "Production-ready Vision-Language Models for mobile devices"
- Homepage: Documentation URL
- Topics: vision-language-model, mobile-ai, core-ml, ios, pytorch, quantization
- Include in the table of contents: Enabled

### 7. Issue and PR Templates

Copy issue templates to `.github/ISSUE_TEMPLATE/`:
```bash
mkdir -p .github/ISSUE_TEMPLATE
# Templates are available in docs/templates/
```

### 8. Code Owners

Review and update `.github/CODEOWNERS` file as needed for your team structure.

## Verification Steps

After manual setup, verify the following:

1. **Workflows**: Check that GitHub Actions workflows run successfully
2. **Security**: Verify security scanning is enabled and working
3. **Documentation**: Confirm GitHub Pages deployment works
4. **Notifications**: Test Slack/Discord webhook notifications
5. **Releases**: Verify automated release process works with tags

## Automation Scripts

The following scripts are provided to help with ongoing maintenance:

- `scripts/metrics_collector.py`: Automated metrics collection
- `scripts/quality_metrics.py`: Code quality reporting
- `scripts/build.sh`: Comprehensive build automation
- `scripts/release.sh`: Release automation

## Support

For questions about the SDLC implementation:
- Check the documentation in `docs/`
- Review workflow examples in `docs/workflows/examples/`
- Open an issue for support requests

---

**Note**: This setup is required only once. After implementation, the SDLC will operate automatically through GitHub Actions and other configured automation.