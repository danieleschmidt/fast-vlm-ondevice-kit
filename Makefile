.PHONY: install test demo lint clean

PYTHON := $(HOME)/anaconda3/bin/python3

install:
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest tests/ -v

demo:
	$(PYTHON) demo.py

lint:
	$(PYTHON) -m ruff check src/ tests/ || true

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ dist/ build/ *.egg-info src/*.egg-info
