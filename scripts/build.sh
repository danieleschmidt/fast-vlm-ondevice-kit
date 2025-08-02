#!/bin/bash

# FastVLM On-Device Kit Build Script
# Comprehensive build automation for all components

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
BUILD_DIR="$PROJECT_ROOT/build"
DIST_DIR="$PROJECT_ROOT/dist"

# Build options
CLEAN=false
PYTHON_BUILD=true
IOS_BUILD=true
DOCS_BUILD=true
DOCKER_BUILD=false
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=true
            shift
            ;;
        --no-python)
            PYTHON_BUILD=false
            shift
            ;;
        --no-ios)
            IOS_BUILD=false
            shift
            ;;
        --no-docs)
            DOCS_BUILD=false
            shift
            ;;
        --docker)
            DOCKER_BUILD=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "FastVLM Build Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --clean       Clean build directories before building"
            echo "  --no-python   Skip Python package build"
            echo "  --no-ios      Skip iOS Swift package build"
            echo "  --no-docs     Skip documentation build"
            echo "  --docker      Build Docker images"
            echo "  --verbose     Enable verbose output"
            echo "  --help        Show this help message"
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

# Clean build directories
clean_build() {
    log_step "Cleaning build directories"
    
    if [ -d "$BUILD_DIR" ]; then
        rm -rf "$BUILD_DIR"
        log_info "Removed $BUILD_DIR"
    fi
    
    if [ -d "$DIST_DIR" ]; then
        rm -rf "$DIST_DIR"
        log_info "Removed $DIST_DIR"
    fi
    
    # Clean Python cache
    find "$PROJECT_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "$PROJECT_ROOT" -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    find "$PROJECT_ROOT" -type f -name "*.pyc" -delete 2>/dev/null || true
    
    # Clean iOS build artifacts
    if [ -d "$PROJECT_ROOT/ios/.build" ]; then
        rm -rf "$PROJECT_ROOT/ios/.build"
        log_info "Cleaned iOS build artifacts"
    fi
    
    # Clean documentation build
    if [ -d "$PROJECT_ROOT/docs/_build" ]; then
        rm -rf "$PROJECT_ROOT/docs/_build"
        log_info "Cleaned documentation build"
    fi
    
    log_success "Build directories cleaned"
}

# Verify environment
verify_environment() {
    log_step "Verifying build environment"
    
    # Check Python
    if ! command_exists python; then
        log_error "Python not found"
        exit 1
    fi
    
    python_version=$(python --version 2>&1 | cut -d' ' -f2)
    log_info "Python version: $python_version"
    
    # Check pip
    if ! command_exists pip; then
        log_error "pip not found"
        exit 1
    fi
    
    # Check Swift (if iOS build enabled)
    if [ "$IOS_BUILD" = true ]; then
        if ! command_exists swift; then
            log_warning "Swift not found - skipping iOS build"
            IOS_BUILD=false
        else
            swift_version=$(swift --version | head -n1)
            log_info "Swift version: $swift_version"
        fi
    fi
    
    # Check Docker (if Docker build enabled)
    if [ "$DOCKER_BUILD" = true ]; then
        if ! command_exists docker; then
            log_warning "Docker not found - skipping Docker build"
            DOCKER_BUILD=false
        else
            docker_version=$(docker --version)
            log_info "Docker version: $docker_version"
        fi
    fi
    
    log_success "Environment verification complete"
}

# Install dependencies
install_dependencies() {
    log_step "Installing dependencies"
    
    # Install Python dependencies
    cd "$PROJECT_ROOT"
    pip install --upgrade pip
    pip install -e ".[dev]"
    
    # Install iOS dependencies
    if [ "$IOS_BUILD" = true ]; then
        cd "$PROJECT_ROOT/ios"
        swift package resolve
        cd "$PROJECT_ROOT"
    fi
    
    log_success "Dependencies installed"
}

# Run quality checks
run_quality_checks() {
    log_step "Running quality checks"
    
    cd "$PROJECT_ROOT"
    
    # Format check
    if ! black --check src tests; then
        log_warning "Code formatting issues found - running formatter"
        black src tests
    fi
    
    # Import sorting
    if ! isort --check-only src tests; then
        log_warning "Import order issues found - fixing imports"
        isort src tests
    fi
    
    # Type checking
    log_info "Running type checking"
    mypy src
    
    # Linting
    log_info "Running linting"
    flake8 src tests
    
    # Security scanning
    log_info "Running security scan"
    bandit -r src
    safety check
    
    log_success "Quality checks completed"
}

# Run tests
run_tests() {
    log_step "Running test suite"
    
    cd "$PROJECT_ROOT"
    
    # Python tests
    pytest tests/ --cov=src/fast_vlm_ondevice --cov-report=html --cov-report=term-missing
    
    # iOS tests
    if [ "$IOS_BUILD" = true ]; then
        cd "$PROJECT_ROOT/ios"
        swift test
        cd "$PROJECT_ROOT"
    fi
    
    log_success "All tests passed"
}

# Build Python package
build_python() {
    log_step "Building Python package"
    
    cd "$PROJECT_ROOT"
    
    # Clean previous builds
    rm -rf build/ dist/ *.egg-info/
    
    # Build package
    python -m build
    
    # Verify build
    if [ -d "dist" ] && [ "$(ls -A dist)" ]; then
        log_success "Python package built successfully"
        ls -la dist/
    else
        log_error "Python package build failed"
        exit 1
    fi
}

# Build iOS package
build_ios() {
    log_step "Building iOS Swift package"
    
    cd "$PROJECT_ROOT/ios"
    
    # Build Swift package
    swift build -c release
    
    # Run Swift tests
    swift test
    
    # Build iOS demo app (if Xcode available)
    if command_exists xcodebuild; then
        log_info "Building iOS demo app"
        if [ -f "FastVLMDemo.xcodeproj/project.pbxproj" ]; then
            xcodebuild -project FastVLMDemo.xcodeproj -scheme FastVLMDemo -destination 'platform=iOS Simulator,name=iPhone 15 Pro' build
        fi
    fi
    
    cd "$PROJECT_ROOT"
    log_success "iOS package built successfully"
}

# Build documentation
build_docs() {
    log_step "Building documentation"
    
    cd "$PROJECT_ROOT/docs"
    
    # Install documentation dependencies
    pip install sphinx sphinx-rtd-theme myst-parser
    
    # Build HTML documentation
    make html
    
    cd "$PROJECT_ROOT"
    log_success "Documentation built successfully"
}

# Build Docker images
build_docker() {
    log_step "Building Docker images"
    
    cd "$PROJECT_ROOT"
    
    # Build development image
    docker build -t fast-vlm-ondevice:dev --target development .
    
    # Build production image
    docker build -t fast-vlm-ondevice:latest --target production .
    
    # Build converter image
    docker build -f docker/Dockerfile.converter -t fast-vlm-converter:latest .
    
    log_success "Docker images built successfully"
}

# Generate build report
generate_report() {
    log_step "Generating build report"
    
    mkdir -p "$BUILD_DIR/reports"
    
    cat > "$BUILD_DIR/reports/build-report.md" << EOF
# FastVLM On-Device Kit Build Report

**Build Date:** $(date)
**Build Version:** $(git describe --tags --always --dirty)
**Git Commit:** $(git rev-parse HEAD)
**Git Branch:** $(git branch --show-current)

## Build Configuration

- Python Build: $PYTHON_BUILD
- iOS Build: $IOS_BUILD
- Documentation Build: $DOCS_BUILD
- Docker Build: $DOCKER_BUILD
- Clean Build: $CLEAN

## Environment

- Python Version: $(python --version 2>&1)
- pip Version: $(pip --version)
$(if [ "$IOS_BUILD" = true ]; then echo "- Swift Version: $(swift --version | head -n1)"; fi)
$(if [ "$DOCKER_BUILD" = true ]; then echo "- Docker Version: $(docker --version)"; fi)

## Build Artifacts

$(if [ -d "$DIST_DIR" ]; then
    echo "### Python Package"
    ls -la "$DIST_DIR/"
fi)

$(if [ -d "$PROJECT_ROOT/ios/.build" ]; then
    echo "### iOS Package"
    echo "iOS Swift package built successfully"
fi)

$(if [ -d "$PROJECT_ROOT/docs/_build" ]; then
    echo "### Documentation"
    echo "HTML documentation generated"
fi)

## Test Results

- Python Tests: $(if pytest --collect-only -q tests/ 2>/dev/null | tail -n1; then echo "✅ Passed"; else echo "❌ Failed"; fi)
$(if [ "$IOS_BUILD" = true ]; then echo "- iOS Tests: ✅ Passed"; fi)

## Quality Metrics

- Code Coverage: $(coverage report --show-missing | tail -n1 | awk '{print $4}' || echo "N/A")
- Security Scan: ✅ Passed
- Type Checking: ✅ Passed

---
Generated by FastVLM Build System
EOF

    log_success "Build report generated: $BUILD_DIR/reports/build-report.md"
}

# Main build process
main() {
    echo "FastVLM On-Device Kit Build System"
    echo "=================================="
    
    # Clean if requested
    if [ "$CLEAN" = true ]; then
        clean_build
    fi
    
    # Verify environment
    verify_environment
    
    # Create build directory
    mkdir -p "$BUILD_DIR" "$DIST_DIR"
    
    # Install dependencies
    install_dependencies
    
    # Run quality checks
    run_quality_checks
    
    # Run tests
    run_tests
    
    # Build components
    if [ "$PYTHON_BUILD" = true ]; then
        build_python
    fi
    
    if [ "$IOS_BUILD" = true ]; then
        build_ios
    fi
    
    if [ "$DOCS_BUILD" = true ]; then
        build_docs
    fi
    
    if [ "$DOCKER_BUILD" = true ]; then
        build_docker
    fi
    
    # Generate build report
    generate_report
    
    echo ""
    log_success "Build completed successfully!"
    echo ""
    echo "Build artifacts:"
    if [ "$PYTHON_BUILD" = true ] && [ -d "$DIST_DIR" ]; then
        echo "  - Python package: $DIST_DIR/"
    fi
    if [ "$IOS_BUILD" = true ]; then
        echo "  - iOS package: $PROJECT_ROOT/ios/.build/"
    fi
    if [ "$DOCS_BUILD" = true ]; then
        echo "  - Documentation: $PROJECT_ROOT/docs/_build/html/"
    fi
    echo "  - Build report: $BUILD_DIR/reports/build-report.md"
}

# Run main function
main "$@"