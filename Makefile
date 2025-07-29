.PHONY: install install-dev test test-unit test-integration test-performance lint format clean build check security benchmark docker-build docker-test help

help:		## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation targets
install:		## Install package dependencies only
	pip install -e .

install-dev:		## Install development dependencies
	pip install -e ".[dev]"
	pre-commit install

# Testing targets
test:			## Run all tests with coverage
	pytest tests/ -v --cov=src/fast_vlm_ondevice --cov-report=term-missing --cov-report=html

test-unit:		## Run unit tests only
	pytest tests/ -v -m "unit" --cov=src/fast_vlm_ondevice

test-integration:	## Run integration tests only
	pytest tests/ -v -m "integration"

test-performance:	## Run performance benchmarks
	pytest tests/performance/ -v -m "performance" --tb=short

test-fast:		## Run tests quickly (no coverage)
	pytest tests/ -v -x --tb=short

# Code quality targets
lint:			## Run all linters
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/
	mypy src/

format:			## Format code with black and isort
	black src/ tests/
	isort src/ tests/

security:		## Run security checks
	bandit -r src/ -f json -o reports/bandit.json || true
	safety check --json --output reports/safety.json || true
	@echo "Security reports generated in reports/"

# Build and packaging
build:			## Build Python package
	python -m build

clean:			## Clean build artifacts and caches
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/ reports/
	rm -rf .mypy_cache/ .ruff_cache/ .tox/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

# Comprehensive checks
check:			## Run all quality checks
	make lint
	make security  
	make test

check-ci:		## Run CI-style checks (strict)
	black --check --diff src/ tests/
	isort --check-only --diff src/ tests/
	flake8 src/ tests/
	mypy src/ --strict
	pytest tests/ --cov=src/fast_vlm_ondevice --cov-fail-under=80

# Multi-environment testing
tox:			## Run tests in multiple Python environments
	tox

tox-parallel:		## Run tox environments in parallel
	tox -p auto

# iOS/Swift targets
ios-setup:		## Setup iOS development environment
	cd ios && swift package resolve

ios-test:		## Run iOS/Swift tests
	cd ios && swift test

ios-build:		## Build iOS Swift package
	cd ios && swift build

# Docker targets
docker-build:		## Build Docker development image
	docker-compose build dev

docker-test:		## Run tests in Docker
	docker-compose run --rm test

docker-lint:		## Run linting in Docker
	docker-compose run --rm lint

docker-security:	## Run security scans in Docker
	docker-compose run --rm security

# Benchmarking and performance
benchmark:		## Run performance benchmarks
	pytest tests/performance/ -v --benchmark-only --benchmark-sort=mean

benchmark-save:		## Save benchmark results
	pytest tests/performance/ --benchmark-save=baseline

benchmark-compare:	## Compare with saved benchmarks
	pytest tests/performance/ --benchmark-compare=baseline

# Documentation
docs-build:		## Build documentation
	cd docs && sphinx-build -b html . _build/html

docs-serve:		## Serve documentation locally
	cd docs/_build/html && python -m http.server 8080

# Release and deployment
release-check:		## Check if ready for release
	@echo "Checking release readiness..."
	make check-ci
	@echo "✓ All checks passed"
	@echo "✓ Ready for release"

version-bump-patch:	## Bump patch version
	bump2version patch

version-bump-minor:	## Bump minor version  
	bump2version minor

version-bump-major:	## Bump major version
	bump2version major

# Utility targets
setup-dirs:		## Create necessary directories
	mkdir -p reports/ logs/ models/ checkpoints/

deps-update:		## Update dependency versions
	pip-compile requirements.in
	pip-compile requirements-dev.in

deps-sync:		## Sync installed packages with requirements
	pip-sync requirements.txt requirements-dev.txt