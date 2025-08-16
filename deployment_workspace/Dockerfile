# FastVLM On-Device Kit - Production Container
FROM python:3.12-slim

# Set build arguments
ARG VERSION=latest
ARG BUILD_DATE

# Set metadata
LABEL version=$VERSION
LABEL build_date=$BUILD_DATE
LABEL project="fast-vlm-ondevice"
LABEL autonomous="true"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Create app user
RUN groupadd -r fastvlm && useradd -r -g fastvlm fastvlm

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY pyproject.toml ./
COPY README.md ./
COPY LICENSE ./

# Copy autonomous components
COPY autonomous_*.py ./
COPY hyper_scale_engine.py ./
COPY security_remediation.py ./

# Install package
RUN pip install -e .

# Copy scripts
COPY scripts/ ./scripts/
RUN chmod +x scripts/*.py

# Switch to non-root user
USER fastvlm

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import fast_vlm_ondevice; print('OK')" || exit 1

# Expose port
EXPOSE 8080

# Default command
CMD ["python3", "-m", "fast_vlm_ondevice.cli", "--help"]
