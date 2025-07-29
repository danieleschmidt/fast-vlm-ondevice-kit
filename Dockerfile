# Fast VLM On-Device Kit - Development Environment
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy dependency files
COPY requirements.txt requirements-dev.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/
COPY scripts/ ./scripts/

# Install package in development mode
RUN pip install -e .

# Development stage
FROM base as development
RUN pip install pre-commit tox
COPY .pre-commit-config.yaml tox.ini ./
RUN pre-commit install-hooks
CMD ["bash"]

# Testing stage  
FROM base as testing
COPY Makefile ./
RUN make test
CMD ["python", "-m", "pytest", "tests/", "-v"]

# Production stage
FROM base as production
# Remove development dependencies
RUN pip uninstall -y pytest pytest-cov black isort mypy flake8 bandit pre-commit tox
# Keep only runtime dependencies
COPY scripts/download_checkpoints.py ./scripts/
EXPOSE 8000
CMD ["python", "-m", "fast_vlm_ondevice"]