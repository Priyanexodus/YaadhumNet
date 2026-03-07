# Makefile

.PHONY: test install clean

test:
	PYTHONPATH=. uv run pytest tests/

install:
	uv sync

clean:
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +