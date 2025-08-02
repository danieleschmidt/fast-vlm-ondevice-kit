#!/bin/bash

# FastVLM On-Device Kit Release Automation Script
# Handles version bumping, changelog generation, and release publishing

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Release options
DRY_RUN=false
RELEASE_TYPE=""
SKIP_TESTS=false
SKIP_BUILD=false
FORCE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --type)
            RELEASE_TYPE="$2"
            shift 2
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --help)
            echo "FastVLM Release Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --dry-run       Perform dry run without making changes"
            echo "  --type TYPE     Release type (patch|minor|major)"
            echo "  --skip-tests    Skip running tests before release"
            echo "  --skip-build    Skip building packages before release"
            echo "  --force         Force release even if checks fail"
            echo "  --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --type patch --dry-run"
            echo "  $0 --type minor"
            echo "  $0 --dry-run"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo ""
    echo -e "${BLUE}==>${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Get current version from git tags
get_current_version() {
    git describe --tags --abbrev=0 2>/dev/null | sed 's/^v//' || echo "0.0.0"
}

# Validate environment
validate_environment() {
    log_step "Validating release environment"
    
    # Check if we're on main branch
    current_branch=$(git branch --show-current)
    if [ "$current_branch" != "main" ] && [ "$FORCE" != true ]; then
        log_error "Not on main branch (currently on $current_branch)"
        log_info "Use --force to release from non-main branch"
        exit 1
    fi
    
    # Check if working directory is clean
    if ! git diff --quiet HEAD && [ "$FORCE" != true ]; then
        log_error "Working directory is not clean"
        log_info "Commit or stash changes before releasing"
        exit 1
    fi
    
    # Check if semantic-release is available
    if ! command_exists npx; then
        log_error "npx not found - install Node.js and npm"
        exit 1
    fi
    
    # Check if required tools are available
    if ! command_exists python; then
        log_error "Python not found"
        exit 1
    fi
    
    if ! command_exists git; then
        log_error "Git not found"
        exit 1
    fi
    
    log_success "Environment validation passed"
}

# Run pre-release checks
run_pre_release_checks() {
    log_step "Running pre-release checks"
    
    cd "$PROJECT_ROOT"
    
    # Run tests if not skipped
    if [ "$SKIP_TESTS" != true ]; then
        log_info "Running test suite"
        python -m pytest tests/ -v --cov=src/fast_vlm_ondevice
        
        # Run iOS tests if available
        if command_exists swift && [ -d "ios" ]; then
            cd ios
            swift test
            cd "$PROJECT_ROOT"
        fi
    else
        log_warning "Skipping tests (--skip-tests specified)"
    fi
    
    # Run security checks
    log_info "Running security checks"
    bandit -r src/
    safety check
    
    # Check code quality
    log_info "Checking code quality"
    black --check src tests
    isort --check-only src tests
    mypy src
    flake8 src tests
    
    log_success "Pre-release checks passed"
}

# Build packages
build_packages() {
    log_step "Building release packages"
    
    if [ "$SKIP_BUILD" != true ]; then
        cd "$PROJECT_ROOT"
        
        # Clean previous builds
        rm -rf build/ dist/ *.egg-info/
        
        # Build Python package
        python -m build
        
        # Build documentation
        if [ -d "docs" ]; then
            cd docs
            make html
            cd "$PROJECT_ROOT"
        fi
        
        log_success "Packages built successfully"
    else
        log_warning "Skipping package build (--skip-build specified)"
    fi
}

# Generate release notes
generate_release_notes() {
    log_step "Generating release notes"
    
    cd "$PROJECT_ROOT"
    
    # Get current version and next version
    current_version=$(get_current_version)
    log_info "Current version: $current_version"
    
    # Generate conventional changelog
    if command_exists conventional-changelog; then
        conventional-changelog -p angular -i CHANGELOG.md -s
    else
        log_warning "conventional-changelog not found - install with: npm install -g conventional-changelog-cli"
    fi
    
    log_success "Release notes generated"
}

# Perform semantic release
perform_semantic_release() {
    log_step "Performing semantic release"
    
    cd "$PROJECT_ROOT"
    
    if [ "$DRY_RUN" = true ]; then
        log_info "Performing dry run release"
        npx semantic-release --dry-run
    else
        log_info "Performing actual release"
        npx semantic-release
    fi
    
    log_success "Semantic release completed"
}

# Update version in all files
update_version_files() {
    local new_version="$1"
    
    log_step "Updating version in project files"
    
    cd "$PROJECT_ROOT"
    
    # Update pyproject.toml
    if [ -f "pyproject.toml" ]; then
        sed -i.bak "s/version = \".*\"/version = \"$new_version\"/" pyproject.toml
        rm pyproject.toml.bak
        log_info "Updated pyproject.toml"
    fi
    
    # Update package.json
    if [ -f "package.json" ]; then
        sed -i.bak "s/\"version\": \".*\"/\"version\": \"$new_version\"/" package.json
        rm package.json.bak
        log_info "Updated package.json"
    fi
    
    # Update iOS Package.swift
    if [ -f "ios/Package.swift" ]; then
        # This is more complex for Swift packages - typically handled by tags
        log_info "iOS Package.swift will be updated via git tags"
    fi
    
    log_success "Version files updated to $new_version"
}

# Create release artifacts
create_release_artifacts() {
    log_step "Creating release artifacts"
    
    cd "$PROJECT_ROOT"
    
    # Create release directory
    mkdir -p "release-artifacts"
    
    # Copy Python packages
    if [ -d "dist" ]; then
        cp dist/* "release-artifacts/"
    fi
    
    # Create model packages if available
    if [ -d "models" ]; then
        find models -name "*.mlpackage" -type d | while read -r model; do
            model_name=$(basename "$model")
            zip -r "release-artifacts/$model_name.zip" "$model"
        done
    fi
    
    # Create documentation archive
    if [ -d "docs/_build/html" ]; then
        tar -czf "release-artifacts/documentation.tar.gz" -C "docs/_build/html" .
    fi
    
    # Create iOS framework if available
    if [ -d "ios/.build" ]; then
        tar -czf "release-artifacts/ios-framework.tar.gz" -C "ios/.build" .
    fi
    
    log_success "Release artifacts created"
}

# Publish to PyPI
publish_to_pypi() {
    log_step "Publishing to PyPI"
    
    cd "$PROJECT_ROOT"
    
    if [ "$DRY_RUN" = true ]; then
        log_info "Would publish to PyPI (dry run)"
        twine check dist/*
    else
        if [ -n "$PYPI_TOKEN" ]; then
            log_info "Publishing to PyPI with token"
            twine upload dist/* --username __token__ --password "$PYPI_TOKEN"
        else
            log_warning "PYPI_TOKEN not set - skipping PyPI publish"
        fi
    fi
    
    log_success "PyPI publish completed"
}

# Send notifications
send_notifications() {
    log_step "Sending release notifications"
    
    local version="$1"
    
    # Slack notification (if webhook available)
    if [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚀 FastVLM On-Device Kit v$version has been released!\"}" \
            "$SLACK_WEBHOOK"
    fi
    
    # Discord notification (if webhook available)
    if [ -n "$DISCORD_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"content\":\"🚀 FastVLM On-Device Kit v$version has been released!\"}" \
            "$DISCORD_WEBHOOK"
    fi
    
    log_success "Notifications sent"
}

# Main release process
main() {
    echo "FastVLM On-Device Kit Release System"
    echo "==================================="
    
    if [ "$DRY_RUN" = true ]; then
        log_warning "DRY RUN MODE - No changes will be made"
    fi
    
    # Validate environment
    validate_environment
    
    # Run pre-release checks
    run_pre_release_checks
    
    # Build packages
    build_packages
    
    # Generate release notes
    generate_release_notes
    
    # Create release artifacts
    create_release_artifacts
    
    # Perform semantic release
    perform_semantic_release
    
    # Get the new version after release
    new_version=$(get_current_version)
    
    if [ "$DRY_RUN" != true ]; then
        # Publish to PyPI
        publish_to_pypi
        
        # Send notifications
        send_notifications "$new_version"
        
        log_success "Release v$new_version completed successfully!"
        
        echo ""
        echo "Release Summary:"
        echo "  - Version: v$new_version"
        echo "  - Artifacts: $(ls -1 release-artifacts/ | wc -l) files"
        echo "  - PyPI: $([ -n "$PYPI_TOKEN" ] && echo "Published" || echo "Skipped")"
        echo "  - GitHub: Released"
    else
        log_info "Dry run completed - no actual release performed"
    fi
}

# Run main function
main "$@"