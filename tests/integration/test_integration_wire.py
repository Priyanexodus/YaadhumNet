"""
tests/integration/test_component_wiring.py

Layer 2 — Narrow Integration Tests (Component Wiring)
No Docker, no SuperLink, no federation run required.

Group 1 — MOONClient ↔ MOONModel        (parameter roundtrip)
Group 2 — MOONClient.fit() ↔ MOONModel  (training loop)
Group 3 — AutoScaleMOONStrategy ↔ FedAvg ↔ server_fn (server wiring)
"""

import copy

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

from moon_fl.model import MOONModel, ViTEncoder
from moon_fl.client_app import MOONClient
from moon_fl.server_app import (
    AutoScaleMOONStrategy,
    server_fn,
    NUM_ROUNDS,
)

from flwr.common import Parameters, FitRes, ndarrays_to_parameters
from flwr.server import ServerAppComponents
from flwr.server.client_proxy import ClientProxy


# ── Constants ─────────────────────────────────────────────────────────────────

BATCH_SIZE = 2
IMAGE_SIZE = 224
ENC_DIM    = 192   # deit_tiny_patch16_224 output dim
PROJ_DIM   = 256   # MOONModel proj_head output dim


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_data_loaders():
    """Mocked loaders returning synthetic tensors. num_classes is local — dynamic per node at runtime."""
    num_classes = 3

    def make_loader():
        batch = (
            torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE),
            torch.randint(0, num_classes, (BATCH_SIZE,)),
        )
        loader                         = MagicMock()
        loader.__iter__                = MagicMock(side_effect=lambda: iter([batch]))
        loader.dataset                 = MagicMock()
        loader.dataset.__len__         = MagicMock(return_value=BATCH_SIZE)
        loader.dataset.dataset.classes = [f"class_{i}" for i in range(num_classes)]
        return loader

    return make_loader(), make_loader(), num_classes


@pytest.fixture(scope="module")
def moon_model(mock_data_loaders):
    """MOONModel with mocked ViTEncoder — avoids pretrained weight download."""
    _, _, num_classes = mock_data_loaders

    with patch("timm.create_model") as mock_create:
        mock_backbone              = MagicMock()
        mock_backbone.num_features = ENC_DIM
        mock_backbone.side_effect  = lambda x: torch.randn(x.size(0), ENC_DIM)
        mock_create.return_value   = mock_backbone
        encoder                    = ViTEncoder(pretrained=False)
        encoder.backbone           = mock_backbone

    model = MOONModel(encoder, proj_dim=PROJ_DIM, num_classes=num_classes)
    model.eval()
    return model


@pytest.fixture(scope="module")
def moon_client(moon_model, mock_data_loaders):
    train_loader, val_loader, _ = mock_data_loaders
    return MOONClient(
        model=moon_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cpu",
        mu=5,
        temperature=0.5,
    )


# ── Group 1 — MOONClient ↔ MOONModel (Parameter Roundtrip) ───────────────────

class TestParameterRoundtrip:

    def test_get_parameters_returns_list_of_numpy(self, moon_client):
        """get_parameters must return list[np.ndarray] for Flower gRPC serialisation."""
        params = moon_client.get_parameters(config={})
        assert isinstance(params, list)
        assert all(isinstance(p, np.ndarray) for p in params)

    def test_get_parameters_count_matches_state_dict(self, moon_client):
        """Parameter count must match state_dict keys — mismatch silently misaligns layers."""
        params     = moon_client.get_parameters(config={})
        state_keys = list(moon_client.model.state_dict().keys())
        assert len(params) == len(state_keys), (
            f"get_parameters returned {len(params)} arrays, "
            f"state_dict has {len(state_keys)} keys"
        )

    def test_set_parameters_roundtrip_preserves_weights(self, moon_client):
        """set_parameters(get_parameters()) must reproduce identical weights."""
        original = moon_client.get_parameters(config={})
        moon_client.set_parameters(original)
        restored = moon_client.get_parameters(config={})

        for i, (orig, rest) in enumerate(zip(original, restored)):
            assert np.allclose(orig, rest, atol=1e-6), (
                f"Weight mismatch at parameter index {i} after set→get roundtrip"
            )

    def test_set_parameters_actually_modifies_model(self, moon_client):
        """set_parameters must not be a no-op — zeroed weights must reflect in the model."""
        zeroed = [np.zeros_like(p) for p in moon_client.get_parameters(config={})]
        moon_client.set_parameters(zeroed)
        restored = moon_client.get_parameters(config={})
        assert all(np.allclose(p, 0.0) for p in restored), \
            "Model weights unchanged after set_parameters"


# ── Group 2 — MOONClient.fit() ↔ MOONModel (Training Loop Boundary) ──────────

class TestFitWiring:

    @pytest.fixture(scope="class")
    def fit_result(self, moon_client):
        """Single fit() call with local_epochs=1, reused across all tests in this class."""
        params = moon_client.get_parameters(config={})
        config = {"local_epochs": 1, "mu": 5, "temperature": 0.5}
        return moon_client.fit(params, config)

    def test_fit_returns_three_tuple(self, fit_result):
        """fit() must return (parameters, num_samples, metrics) per Flower NumPyClient contract."""
        assert isinstance(fit_result, tuple) and len(fit_result) == 3, (
            f"fit() must return a 3-tuple, got {type(fit_result).__name__}"
        )

    def test_fit_num_samples_is_positive_int(self, fit_result):
        """num_samples must be a positive int — zero or float corrupts FedAvg weighted aggregation."""
        _, num_samples, _ = fit_result
        assert isinstance(num_samples, int) and num_samples > 0, (
            f"Expected positive int, got {type(num_samples).__name__}: {num_samples}"
        )

    def test_fit_metrics_contain_required_keys(self, fit_result):
        """fit() metrics must include train_accuracy and train_loss for server-side aggregation."""
        _, _, metrics = fit_result
        assert "train_accuracy" in metrics, "fit() metrics missing 'train_accuracy'"
        assert "train_loss"     in metrics, "fit() metrics missing 'train_loss'"

    def test_fit_loss_is_finite_and_non_negative(self, fit_result):
        """train_loss must be finite and non-negative — NaN propagates silently into MLflow."""
        _, _, metrics = fit_result
        loss = metrics["train_loss"]
        assert loss >= 0,         f"train_loss is negative: {loss}"
        assert np.isfinite(loss), f"train_loss is not finite: {loss}"

    def test_fit_initializes_previous_model_on_first_call(self, moon_model,
                                                           mock_data_loaders):
        """previous_model must be set after round 1 — None in round 2 crashes contrastive loss."""
        train_loader, val_loader, _ = mock_data_loaders
        fresh_client = MOONClient(
            model=copy.deepcopy(moon_model),
            train_loader=train_loader,
            val_loader=val_loader,
            device="cpu", mu=5, temperature=0.5,
        )
        assert fresh_client.previous_model is None

        params = fresh_client.get_parameters(config={})
        fresh_client.fit(params, {"local_epochs": 1, "mu": 5, "temperature": 0.5})

        assert fresh_client.previous_model is not None, \
            "previous_model still None after fit() — round 2 will crash"

    def test_fit_updates_weights(self, mock_data_loaders):
        """At least one weight must change after fit() — frozen weights silently break training."""
        train_loader, val_loader, num_classes = mock_data_loaders

        # Build a fresh model with random init weights — not deepcopy of the
        # shared moon_model fixture which may be zeroed by an earlier test.
        with patch("timm.create_model") as mock_create:
            mock_backbone              = MagicMock()
            mock_backbone.num_features = ENC_DIM
            mock_backbone.side_effect  = lambda x: torch.randn(x.size(0), ENC_DIM)
            mock_create.return_value   = mock_backbone
            encoder                    = ViTEncoder(pretrained=False)
            encoder.backbone           = mock_backbone
        fresh_model = MOONModel(encoder, proj_dim=PROJ_DIM, num_classes=num_classes)

        client = MOONClient(
            model=fresh_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device="cpu", mu=5, temperature=0.5,
        )
        before = copy.deepcopy(client.get_parameters(config={}))
        client.fit(before, {"local_epochs": 1, "mu": 5, "temperature": 0.5})
        after = client.get_parameters(config={})

        changed = any(
            not np.allclose(b, a, atol=1e-8) for b, a in zip(before, after)
        )
        assert changed, "No weights changed after fit() — model may be frozen"


# ── Group 3 — AutoScaleMOONStrategy ↔ FedAvg ↔ server_fn (Server Wiring) ─────

class TestStrategyWiring:

    @pytest.fixture(scope="class")
    def strategy(self):
        return AutoScaleMOONStrategy(
            num_rounds=NUM_ROUNDS,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=1,
            min_evaluate_clients=1,
            min_available_clients=1,
        )

    def _make_parameters(self, moon_model) -> Parameters:
        """Build a real Flower Parameters object using Flower's own serialisation."""
        arrays = [val.cpu().numpy() for val in moon_model.state_dict().values()]
        return ndarrays_to_parameters(arrays)

    def test_configure_fit_sets_min_clients_to_available(self, strategy, moon_model):
        """min_fit_clients must equal num_available() — stale value silently runs rounds with 1 client."""
        client_manager = MagicMock()
        client_manager.num_available.return_value = 2

        with patch.object(type(strategy).__bases__[0], "configure_fit",
                          return_value=[]):
            strategy.configure_fit(
                1, self._make_parameters(moon_model), client_manager
            )

        assert strategy.min_fit_clients == 2, (
            f"Expected min_fit_clients=2, got {strategy.min_fit_clients}"
        )

    def test_configure_fit_config_contains_required_keys(self, strategy, moon_model):
        """fit_config must include local_epochs, round, mu, temperature — missing keys fall back silently."""
        client_manager = MagicMock()
        client_manager.num_available.return_value = 1

        with patch.object(type(strategy).__bases__[0], "configure_fit",
                          return_value=[]):
            strategy.configure_fit(
                1, self._make_parameters(moon_model), client_manager
            )

        config = strategy.on_fit_config_fn(server_round=1)
        for key in ("local_epochs", "round", "mu", "temperature"):
            assert key in config, f"fit_config missing key: '{key}'"

    def test_aggregate_fit_returns_non_none(self, strategy, moon_model):
        """aggregate_fit must not return None — Flower silently skips the weight update if it does."""
        proxy   = MagicMock(spec=ClientProxy)
        fit_res = FitRes(
            status=MagicMock(),
            parameters=self._make_parameters(moon_model),
            num_examples=10,
            metrics={"train_accuracy": 0.8, "train_loss": 0.3},
        )

        with patch("mlflow.log_metrics"):
            result = strategy.aggregate_fit(1, [(proxy, fit_res)], [])

        assert result is not None, \
            "aggregate_fit returned None — Flower will skip the weight update"

    def test_server_fn_returns_serverapp_components(self):
        """server_fn must return ServerAppComponents — any crash here surfaces at deploy time."""
        with patch("mlflow.set_experiment"), \
             patch("mlflow.start_run"),       \
             patch("mlflow.log_params"):
            components = server_fn(MagicMock())

        assert isinstance(components, ServerAppComponents), (
            f"Expected ServerAppComponents, got {type(components).__name__}"
        )

    def test_strategy_and_config_num_rounds_consistent(self):
        """strategy.num_rounds and ServerConfig.num_rounds must match — divergence misfires mlflow.end_run()."""
        with patch("mlflow.set_experiment"), \
             patch("mlflow.start_run"),       \
             patch("mlflow.log_params"):
            components = server_fn(MagicMock())

        assert components.strategy.num_rounds == components.config.num_rounds, (
            f"strategy.num_rounds ({components.strategy.num_rounds}) != "
            f"config.num_rounds ({components.config.num_rounds})"
        )