#!/bin/bash
# DevContainer setup script for FastVLM On-Device Kit

set -e

echo "🚀 Setting up FastVLM On-Device Kit development environment..."

# Upgrade pip and install build tools
python -m pip install --upgrade pip setuptools wheel

# Install the project in development mode
echo "📦 Installing FastVLM dependencies..."
pip install -e ".[dev]"

# Install additional development tools
echo "🔧 Installing additional development tools..."
pip install \
    jupyterlab \
    notebook \
    ipywidgets \
    matplotlib \
    seaborn \
    plotly \
    pre-commit \
    pip-tools \
    safety \
    bandit[toml] \
    ruff

# Install pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
pre-commit install

# Create useful directories
echo "📁 Creating development directories..."
mkdir -p {checkpoints,models,reports,logs,experiments}

# Download sample checkpoints (if available)
echo "📥 Attempting to download sample checkpoints..."
if [ -f "scripts/download_checkpoints.py" ]; then
    python scripts/download_checkpoints.py --sample || echo "⚠️  Sample checkpoints not available"
fi

# Set up Git configuration helpers
echo "⚙️  Configuring Git helpers..."
git config --global --add safe.directory /workspaces/fast-vlm-ondevice-kit
git config pull.rebase false

# Create useful aliases
echo "📝 Setting up helpful aliases..."
cat >> ~/.bashrc << 'EOF'

# FastVLM development aliases
alias test='pytest --cov=src/fast_vlm_ondevice'
alias lint='ruff check src tests'
alias format='black src tests && isort src tests'
alias typecheck='mypy src'
alias security='bandit -r src'
alias quality='ruff check src tests && black --check src tests && isort --check-only src tests && mypy src && bandit -q -r src'
alias serve='python -m http.server 8000'
alias jupyter='jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'

# Quick project navigation
alias cdsrc='cd /workspaces/fast-vlm-ondevice-kit/src'
alias cdtests='cd /workspaces/fast-vlm-ondevice-kit/tests'
alias cddocs='cd /workspaces/fast-vlm-ondevice-kit/docs'
alias cdscripts='cd /workspaces/fast-vlm-ondevice-kit/scripts'
EOF

# Make sure the new aliases are available
source ~/.bashrc || true

echo "✅ FastVLM development environment setup complete!"
echo ""
echo "🎯 Quick Start Commands:"
echo "  test          - Run tests with coverage"
echo "  lint          - Check code quality with ruff"
echo "  format        - Format code with black and isort"
echo "  typecheck     - Run mypy type checking"
echo "  security      - Run security analysis with bandit"
echo "  quality       - Run all quality checks"
echo "  jupyter       - Start Jupyter Lab server"
echo ""
echo "📁 Project Structure:"
echo "  /workspaces/fast-vlm-ondevice-kit/  - Project root"
echo "  src/fast_vlm_ondevice/              - Python package"
echo "  ios/                                - Swift package"
echo "  tests/                              - Test suite"
echo "  docs/                               - Documentation"
echo "  benchmarks/                         - Performance tests"
echo ""
echo "🔗 Next Steps:"
echo "1. Review the README.md for project overview"
echo "2. Check docs/DEVELOPMENT.md for detailed setup"
echo "3. Run 'test' to verify installation"
echo "4. Start coding! 🎉"