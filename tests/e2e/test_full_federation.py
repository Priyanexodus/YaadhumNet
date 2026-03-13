"""
tests/e2e/test_full_federation.py

E2E Test — Full Federation Run
Requires docker stack running and SuperNodes connected.

Runs one full flwr federation and verifies:
  1. flwr run submits successfully
  2. MLflow experiment exists after the run
  3. At least one FINISHED run was logged
  4. Expected metrics are present and valid

Run:
    make stack-up
    uv run pytest tests/e2e/test_full_federation.py -v
    make stack-down
"""

import time
import subprocess
import pytest
import mlflow

from moon_fl.server_app import NUM_ROUNDS

MLFLOW_URI      = "http://localhost:5000"
EXPERIMENT_NAME = "MOON-FL-Production"
POLL_INTERVAL   = 15    # seconds between MLflow polls
POLL_TIMEOUT    = 900   # 15 min max — adjust to your NUM_ROUNDS


@pytest.fixture(scope="module")
def federation_run():
    """Submit flwr run, poll MLflow until FINISHED, return the run."""
    result = subprocess.run(
        ["uv", "run", "flwr", "run", ".", "local-deployment"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f"flwr run failed to submit:\n{result.stderr}"
    )

    client  = mlflow.tracking.MlflowClient(MLFLOW_URI)
    elapsed = 0
    while elapsed < POLL_TIMEOUT:
        exp = client.get_experiment_by_name(EXPERIMENT_NAME)
        if exp:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string="attributes.status = 'FINISHED'",
                order_by=["start_time DESC"],
                max_results=1,
            )
            if runs:
                return runs[0]
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

    pytest.fail(f"No FINISHED run in MLflow after {POLL_TIMEOUT}s")


def test_federation_run_completes(federation_run):
    """Federation must produce a FINISHED MLflow run."""
    assert federation_run.info.status == "FINISHED"


def test_train_accuracy_logged(federation_run):
    """train_accuracy must be logged — missing means client metrics never reached server."""
    assert "train_accuracy" in federation_run.data.metrics, (
        "train_accuracy not found in MLflow metrics"
    )


def test_train_loss_logged(federation_run):
    """train_loss must be logged and finite."""
    assert "train_loss" in federation_run.data.metrics, (
        "train_loss not found in MLflow metrics"
    )
    loss = federation_run.data.metrics["train_loss"]
    assert loss >= 0,              f"train_loss is negative: {loss}"
    assert loss < float("inf"),    f"train_loss is infinite: {loss}"


def test_accuracy_is_valid_probability(federation_run):
    """train_accuracy must be in [0, 1]."""
    acc = federation_run.data.metrics["train_accuracy"]
    assert 0.0 <= acc <= 1.0, f"train_accuracy out of range: {acc}"


def test_num_rounds_param_logged(federation_run):
    """num_rounds param must be logged and match NUM_ROUNDS constant."""
    assert "num_rounds" in federation_run.data.params, (
        "num_rounds param not logged — server_fn mlflow.log_params may be broken"
    )
    assert int(federation_run.data.params["num_rounds"]) == NUM_ROUNDS, (
        f"Logged num_rounds != NUM_ROUNDS constant ({NUM_ROUNDS})"
    )