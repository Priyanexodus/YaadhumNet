# Makefile

.PHONY: test install-cpu install-gpu install-server clean

test:
	PYTHONPATH=. uv run pytest tests/

install-cpu:
	uv sync --extra client --extra cpu

install-gpu:
	uv sync --extra client

install-server:
	uv sync --extra server

clean:
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +