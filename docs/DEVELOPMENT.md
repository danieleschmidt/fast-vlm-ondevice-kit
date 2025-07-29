# Development Guide

This guide covers the development setup and workflows for Fast VLM On-Device Kit.

## Prerequisites

- Python 3.10+ with pip
- Xcode 15.0+ (for iOS development)
- Apple Silicon Mac (recommended for Core ML conversion)
- Git with LFS support

## Local Development Setup

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/fast-vlm-ondevice-kit.git
cd fast-vlm-ondevice-kit
```

### 2. Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install package in development mode
make install
# or: pip install -e ".[dev]"
```

### 3. Pre-commit Hooks

```bash
pre-commit install
```

### 4. iOS Development

```bash
cd ios
swift package resolve
```

## Development Workflow

### Code Changes

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes with tests
3. Run quality checks: `make check`
4. Commit with pre-commit hooks
5. Push and create pull request

### Testing

```bash
# Run Python tests
make test

# Run iOS tests
cd ios && swift test

# Run specific test file
pytest tests/test_converter.py -v
```

### Code Quality

```bash
# Format code
make format

# Run linters
make lint

# Run all checks
make check
```

## Project Structure

```
├── src/fast_vlm_ondevice/     # Python package
│   ├── converter.py           # PyTorch to Core ML conversion
│   └── quantization.py        # Quantization utilities
├── ios/                       # Swift package
│   ├── Sources/FastVLMKit/    # Swift library code
│   └── Tests/                 # Swift tests
├── tests/                     # Python tests
├── docs/                      # Documentation
├── examples/                  # Usage examples
└── scripts/                   # Utility scripts
```

## Key Components

### Model Conversion Pipeline

1. **PyTorch Model Loading** (`converter.py`)
   - Load pre-trained FastVLM checkpoints
   - Handle model architecture variations

2. **Quantization** (`quantization.py`)
   - INT4/INT8 weight quantization
   - Per-layer quantization strategies
   - Calibration dataset support

3. **Core ML Export** (`converter.py`)
   - Apple Neural Engine optimization
   - Flexible input shapes
   - Performance profiling

### iOS Integration

1. **Swift Package** (`ios/Sources/FastVLMKit/`)
   - Core ML model loading
   - Image preprocessing
   - Text tokenization
   - Inference orchestration

2. **Performance Utilities**
   - Latency measurement
   - Memory profiling
   - Energy impact tracking

## Adding New Features

### Python Components

1. Add implementation in `src/fast_vlm_ondevice/`
2. Write tests in `tests/`
3. Update `__init__.py` exports
4. Add documentation

### iOS Components

1. Add Swift code in `ios/Sources/FastVLMKit/`
2. Write tests in `ios/Tests/FastVLMKitTests/`
3. Update Package.swift if needed
4. Add usage examples

## Performance Optimization

### Model Optimization
- Use INT4 quantization for vision encoder
- Optimize fusion layer precision
- Profile memory usage patterns

### iOS Optimization
- Leverage Apple Neural Engine
- Minimize Core ML model loading time
- Optimize image preprocessing pipeline

## Troubleshooting

### Common Issues

**Import errors**: Ensure package installed with `pip install -e .`

**Core ML conversion fails**: Check PyTorch and coremltools versions

**iOS build errors**: Verify Xcode version and deployment targets

**Test failures**: Run `make clean` and reinstall dependencies

### Debug Tips

- Use `MLCOMPUTE_AVAILABLE_DEVICES=gpu` for GPU-only testing
- Enable Core ML debugging with Instruments
- Profile with `python -m cProfile` for Python bottlenecks
- Use Xcode Instruments for iOS performance analysis

## Contributing Guidelines

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed contribution guidelines.

## Resources

- [Core ML Documentation](https://developer.apple.com/documentation/coreml)
- [PyTorch to Core ML Guide](https://coremltools.readme.io/docs)
- [Apple Neural Engine Optimization](https://developer.apple.com/documentation/coreml/optimizing_neural_networks_for_apple_neural_engine)