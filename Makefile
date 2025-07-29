.PHONY: install test lint format clean help

help:		## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:	## Install dependencies
	pip install -e ".[dev]"
	pre-commit install

test:		## Run tests
	pytest tests/ -v --cov=src/fast_vlm_ondevice

lint:		## Run linters
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/
	mypy src/

format:		## Format code
	black src/ tests/
	isort src/ tests/

clean:		## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -name "*.pyc" -delete

build:		## Build package
	python -m build

check:		## Run all checks
	make lint
	make test

ios-setup:	## Setup iOS development
	cd ios && swift package resolve