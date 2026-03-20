"""
tests/test_integration_mlflow.py

Layer 1 — MLflow Service Tests
Uses a dummy run — does not depend on a prior federation run.
Probe experiment and run are deleted after all tests complete.

Run:
    uv run pytest tests/test_integration_mlflow.py -v
"""

import pytest
import mlflow

pytestmark = pytest.mark.integration

MLFLOW_URI      = "http://localhost:5000"
EXPERIMENT_NAME = "test-integration-probe"


@pytest.fixture(scope="module", autouse=True)
def probe_run():
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient(MLFLOW_URI)

    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp and exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)
    exp_id = exp.experiment_id if exp else mlflow.create_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(experiment_id=exp_id) as run:
        mlflow.log_param("probe", "integration-test")
        mlflow.log_metrics({"accuracy": 0.75, "loss": 0.42})

    yield exp_id, run.info.run_id

    client.delete_run(run.info.run_id)
    client.delete_experiment(exp_id)


def test_experiment_exists(probe_run):
    """Probe experiment must be retrievable."""
    client = mlflow.tracking.MlflowClient(MLFLOW_URI)
    exp_id, _ = probe_run
    assert client.get_experiment(exp_id) is not None

def test_run_is_logged(probe_run):
    """Completed run must be retrievable from the experiment."""
    client = mlflow.tracking.MlflowClient(MLFLOW_URI)
    exp_id, _ = probe_run
    runs = client.search_runs(experiment_ids=[exp_id])
    assert len(runs) > 0

def test_metrics_are_stored(probe_run):
    """Logged metrics must survive the write→read cycle."""
    client = mlflow.tracking.MlflowClient(MLFLOW_URI)
    _, run_id = probe_run
    metrics = client.get_run(run_id).data.metrics
    assert abs(metrics["accuracy"] - 0.75) < 1e-6
    assert abs(metrics["loss"]     - 0.42) < 1e-6

def test_params_are_stored(probe_run):
    """Logged params must be retrievable alongside metrics."""
    client = mlflow.tracking.MlflowClient(MLFLOW_URI)
    _, run_id = probe_run
    assert "probe" in client.get_run(run_id).data.params