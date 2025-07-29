# Contributing to Fast VLM On-Device Kit

Thank you for your interest in contributing! This project aims to make Vision-Language Models accessible on mobile devices.

## Quick Start

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests: `python -m pytest` (once implemented)
5. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/fast-vlm-ondevice-kit.git
cd fast-vlm-ondevice-kit

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

## Contribution Areas

We welcome contributions in:

- **Model Optimizations**: INT4/INT8 quantization improvements
- **Mobile Integration**: iOS/Android deployment enhancements  
- **Performance**: Latency and memory optimizations
- **Documentation**: Examples, tutorials, API docs
- **Testing**: Unit tests, integration tests, benchmarks

## Code Standards

- Follow PEP 8 for Python code
- Use Swift style guide for iOS code
- Include docstrings for public APIs
- Add tests for new functionality
- Update documentation for user-facing changes

## Pull Request Process

1. Ensure your PR has a clear description
2. Reference related issues with `Fixes #123`
3. Include tests for new features
4. Update documentation as needed
5. Ensure CI checks pass

## Reporting Issues

Use GitHub Issues for:
- Bug reports with reproduction steps
- Feature requests with use cases
- Performance issues with profiling data
- Documentation improvements

## Security

Report security vulnerabilities to fast-vlm@yourdomain.com rather than public issues.

## License

By contributing, you agree your contributions will be licensed under the MIT License.