# Claude AI Assistant Context

This document provides context for AI assistants working with the FastVLM On-Device Kit repository.

## Project Overview

FastVLM On-Device Kit is a production-ready implementation of Apple's CVPR-25 FastVLM encoder for mobile applications. It provides PyTorch to Core ML conversion with INT4 quantization and Swift integration for iOS/macOS.

## Repository Structure

- `src/fast_vlm_ondevice/` - Python package for model conversion and quantization
- `ios/` - Swift package for iOS/macOS integration  
- `tests/` - Comprehensive test suite with pytest
- `docs/` - Extensive documentation including API, architecture, security
- `benchmarks/` - Performance measurement and automation tools
- `scripts/` - Utility scripts for checkpoints and releases

## Key Technologies

- **Python**: PyTorch, Core ML Tools, Transformers
- **Swift**: Core ML, Vision framework integration
- **Mobile**: iOS 17+, Apple Neural Engine optimization
- **CI/CD**: Advanced workflow templates and automation

## Development Commands

```bash
# Setup development environment
pip install -e ".[dev]"
pre-commit install

# Run tests with coverage
pytest --cov=src/fast_vlm_ondevice

# Format code
black src tests
isort src tests

# Type checking
mypy src

# Security scanning
bandit -r src

# Build documentation
cd docs && make html
```

## SDLC Maturity Level

This repository is at **ADVANCED** maturity (85%+) with:
- ✅ Comprehensive testing framework
- ✅ Code quality automation (black, isort, mypy, flake8)
- ✅ Security scanning (bandit)
- ✅ Pre-commit hooks
- ✅ Extensive documentation
- ✅ Performance benchmarking
- ✅ Docker containerization
- ✅ Multi-language support (Python + Swift)

## AI Assistant Guidelines

When working with this repository:

1. **Respect existing patterns** - Follow established code style and architecture
2. **Security first** - This handles ML models and mobile deployment
3. **Cross-platform considerations** - Changes may affect both Python and Swift components
4. **Performance awareness** - Mobile optimization is critical
5. **Documentation updates** - Keep docs synchronized with code changes

## Testing Strategy

- Unit tests for core functionality
- Integration tests for model conversion
- Performance benchmarks for mobile devices
- Security tests for model handling

## Common Tasks

- Model conversion optimization
- Mobile performance profiling  
- iOS integration enhancements
- Documentation improvements
- Security scanning updates