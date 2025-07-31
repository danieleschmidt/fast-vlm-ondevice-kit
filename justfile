# Justfile for FastVLM On-Device Kit
# Modern task runner - install with: cargo install just
# Run `just` to see all available commands

# Default recipe displays help
default:
    @just --list

# Development setup
setup:
    #!/usr/bin/env bash
    echo "🚀 Setting up FastVLM development environment..."
    python -m pip install --upgrade pip
    pip install -e ".[dev]"
    pre-commit install
    echo "✅ Setup complete!"

# Code quality and formatting
format:
    #!/usr/bin/env bash
    echo "🎨 Formatting code..."
    black src tests
    isort src tests
    ruff format src tests

lint:
    #!/usr/bin/env bash
    echo "🔍 Linting code..."
    ruff check src tests
    black --check src tests
    isort --check-only src tests

typecheck:
    #!/usr/bin/env bash
    echo "🔎 Type checking..."
    mypy src

security:
    #!/usr/bin/env bash
    echo "🔒 Security scanning..."
    bandit -r src
    safety check --json

# Testing
test:
    #!/usr/bin/env bash
    echo "🧪 Running tests..."
    pytest --cov=src/fast_vlm_ondevice --cov-report=html --cov-report=term-missing

test-fast:
    #!/usr/bin/env bash
    echo "⚡ Running fast tests..."
    pytest -x --ff tests/

test-performance:
    #!/usr/bin/env bash
    echo "📊 Running performance tests..."
    python benchmarks/performance_automation.py --quick

# Quality gate - run all checks
quality: lint typecheck security test
    echo "✅ All quality checks passed!"

# Model operations
download-checkpoints model="fast-vlm-base":
    #!/usr/bin/env bash
    echo "📥 Downloading {{ model }} checkpoints..."
    python scripts/download_checkpoints.py --model {{ model }}

convert-model model="fast-vlm-base" output="FastVLM.mlpackage":
    #!/usr/bin/env bash
    echo "🔄 Converting {{ model }} to Core ML..."
    python -c "
    from src.fast_vlm_ondevice.converter import FastVLMConverter
    converter = FastVLMConverter()
    model = converter.load_pytorch_model('checkpoints/{{ model }}.pth')
    coreml_model = converter.convert_to_coreml(model, quantization='int4')
    coreml_model.save('{{ output }}')
    print('✅ Model converted to {{ output }}')
    "

# Development server and tools
serve port="8000":
    #!/usr/bin/env bash
    echo "🌐 Starting development server on port {{ port }}..."
    python -m http.server {{ port }}

jupyter:
    #!/usr/bin/env bash
    echo "📓 Starting Jupyter Lab..."
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# Documentation
docs:
    #!/usr/bin/env bash
    echo "📚 Building documentation..."
    cd docs && make html
    echo "✅ Documentation built in docs/_build/html/"

docs-serve:
    #!/usr/bin/env bash
    echo "📖 Serving documentation..."
    cd docs/_build/html && python -m http.server 8080

# Docker operations
docker-build:
    #!/usr/bin/env bash
    echo "🐳 Building Docker image..."
    docker build -t fast-vlm-ondevice .

docker-run:
    #!/usr/bin/env bash
    echo "🚀 Running Docker container..."
    docker run -it --rm -v $(pwd):/workspace fast-vlm-ondevice

docker-compose-up:
    #!/usr/bin/env bash
    echo "🐳 Starting Docker Compose services..."
    docker-compose up -d

docker-compose-down:
    #!/usr/bin/env bash
    echo "🛑 Stopping Docker Compose services..."
    docker-compose down

# Benchmarking and profiling
benchmark models="fast-vlm-base" iterations="100":
    #!/usr/bin/env bash
    echo "🏃 Running benchmarks for {{ models }}..."
    python benchmarks/performance_automation.py \
        --models {{ models }} \
        --iterations {{ iterations }} \
        --output benchmark-results

profile script="src/fast_vlm_ondevice/converter.py":
    #!/usr/bin/env bash
    echo "🔬 Profiling {{ script }}..."
    python -m cProfile -o profile.stats {{ script }}
    python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

# iOS development (requires macOS)
ios-build:
    #!/usr/bin/env bash
    echo "📱 Building iOS package..."
    cd ios && swift build

ios-test:
    #!/usr/bin/env bash
    echo "🧪 Running iOS tests..."
    cd ios && swift test

# Maintenance and cleanup
clean:
    #!/usr/bin/env bash
    echo "🧹 Cleaning up..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name ".coverage*" -delete 2>/dev/null || true
    echo "✅ Cleanup complete!"

update-deps:
    #!/usr/bin/env bash
    echo "📦 Updating dependencies..."
    pip-compile requirements.in
    pip-compile requirements-dev.in
    pip install -r requirements-dev.txt
    echo "✅ Dependencies updated!"

# Release management
version:
    #!/usr/bin/env bash
    echo "📌 Current version:"
    python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"

tag version:
    #!/usr/bin/env bash
    echo "🏷️ Creating tag v{{ version }}..."
    git tag -a v{{ version }} -m "Release v{{ version }}"
    echo "✅ Tag created. Push with: git push origin v{{ version }}"

# Utility commands
size:
    #!/usr/bin/env bash
    echo "📊 Project size analysis:"
    echo "Total lines of code:"
    find . -name "*.py" -not -path "./.venv/*" -not -path "./build/*" | xargs wc -l | tail -1
    echo "Directory sizes:"
    du -sh src/ tests/ docs/ ios/ benchmarks/ scripts/ 2>/dev/null || true

health:
    #!/usr/bin/env bash
    echo "🏥 Project health check:"
    echo "Python version: $(python --version)"
    echo "Pip version: $(pip --version)"
    echo "Pre-commit: $(pre-commit --version 2>/dev/null || echo 'Not installed')"
    echo "Docker: $(docker --version 2>/dev/null || echo 'Not available')"
    echo "Git status:"
    git status --porcelain || echo "Not a git repository"