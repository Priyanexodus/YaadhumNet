.PHONY: install-cpu install-gpu install-server \
        stack-up stack-down stack-logs \
        wait-all run-federation \
        test test-unit test-wire test-infra test-health test-mlflow \
        integration-test test-all \
        clean

COMPOSE_FILE := docker-compose-test.yml
PYTEST       := PYTHONPATH=. uv run python -m pytest -v --tb=short
WAIT_TIMEOUT := 60

SUPERLINK_EXEC_PORT  := 9091
SUPERLINK_FLEET_PORT := 9092
MLFLOW_PORT          := 5000
POSTGRES_PORT        := 5432

define wait-for-port
	@elapsed=0; \
	echo "[wait] $(2) on port $(1)..."; \
	while ! nc -z localhost $(1) 2>/dev/null; do \
		sleep 2; elapsed=$$((elapsed + 2)); \
		if [ $$elapsed -ge $(WAIT_TIMEOUT) ]; then \
			echo "[error] Timeout waiting for $(2)"; exit 1; \
		fi; \
	done; \
	echo "[ok] $(2) ready"
endef

install-cpu:
	uv sync --extra client --extra cpu

install-gpu:
	uv sync --extra client

install-server:
	uv sync --extra server

stack-up:
	docker compose -f $(COMPOSE_FILE) up -d --build

stack-down:
	docker compose -f $(COMPOSE_FILE) down

stack-logs:
	docker compose -f $(COMPOSE_FILE) logs --tail=100 -f

wait-all:
	$(call wait-for-port,$(POSTGRES_PORT),PostgreSQL)
	$(call wait-for-port,$(MLFLOW_PORT),MLflow)
	$(call wait-for-port,$(SUPERLINK_EXEC_PORT),SuperLink Exec)
	$(call wait-for-port,$(SUPERLINK_FLEET_PORT),SuperLink Fleet)

run-federation:
	uv run flwr run . local-deployment --stream

test:
	PYTHONPATH=. uv run pytest tests/

test-unit:
	$(PYTEST) tests/unit/

test-wire:
	$(PYTEST) tests/integration/test_integration_wire.py

test-infra:
	$(PYTEST) tests/integration/test_integration_flower.py

test-health:
	$(PYTEST) tests/integration/test_integration_health.py

test-mlflow:
	$(PYTEST) tests/integration/test_integration_mlflow.py

integration-test: test-wire stack-up wait-all test-health test-mlflow test-infra
	@echo "[done] Integration tests passed"
	$(MAKE) stack-down

test-e2e:
	$(PYTEST) tests/e2e/test_full_federation.py

test-all: test-unit integration-test run-federation test-e2e
	@echo "[done] Full test suite passed"

clean:
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +