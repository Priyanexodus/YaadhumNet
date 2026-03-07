from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.common import Context
import mlflow


# ── Metric aggregation ───────────────────────────────────────────
def weighted_average(metrics):
    total = sum(n for n, _ in metrics)
    acc = sum(n * m["accuracy"] for n, m in metrics) / total
    return {"accuracy": acc}

def weighted_train_average(metrics):
    total = sum(n for n, _ in metrics)
    acc      = sum(n * m.get("train_accuracy", 0) for n, m in metrics) / total
    loss_avg = sum(n * m.get("train_loss", 0)     for n, m in metrics) / total
    return {"train_accuracy": acc, "train_loss": loss_avg}


# ── Auto-scaling strategy with MLflow logging ────────────────────
class AutoScaleMOONStrategy(FedAvg):

    def configure_fit(self, server_round, parameters, client_manager):
        num_clients = client_manager.num_available()

        # Update thresholds dynamically
        self.min_fit_clients      = num_clients
        self.min_evaluate_clients = num_clients
        self.min_available_clients = num_clients

        # Closure — injects client_manager into fit_config
        self.on_fit_config_fn = self._make_fit_config(client_manager)

        print(f"Round {server_round}: {num_clients} clients detected")
        return super().configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        num_clients = client_manager.num_available()
        self.min_evaluate_clients = num_clients
        return super().configure_evaluate(server_round, parameters, client_manager)

    def _make_fit_config(self, client_manager):
        """Returns a fit_config function with client_manager baked in."""
        def fit_config(server_round):
            return {
                "local_epochs": 5,
                "round":        server_round,
                # num_partitions removed — no Dirichlet partitioning ✅
                "mu":           5,
                "temperature":  0.5,
            }
        return fit_config

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate fit results and log train metrics to MLflow."""
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            _, metrics = aggregated
            mlflow.log_metrics({
                "train_accuracy": metrics.get("train_accuracy", 0),
                "train_loss":     metrics.get("train_loss", 0),
            }, step=server_round)
            print(f"Round {server_round} | train_accuracy: {metrics.get('train_accuracy', 0):.4f} | train_loss: {metrics.get('train_loss', 0):.4f}")
        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluate results and log eval metrics to MLflow."""
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        if metrics:
            mlflow.log_metrics({
                "eval_accuracy": metrics.get("accuracy", 0),
                "eval_loss":     loss,
            }, step=server_round)
            print(f"Round {server_round} | eval_accuracy: {metrics.get('accuracy', 0):.4f} | eval_loss: {loss:.4f}")
        return loss, metrics


# ── Server entry point ───────────────────────────────────────────
def server_fn(context: Context):
    mlflow.set_tracking_uri("http://<EC2-PUBLIC-IP>:5000")   # ← replace before deploying
    mlflow.set_experiment("MOON-FL-Production")
    mlflow.start_run()

    strategy = AutoScaleMOONStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=2,        # wait for at least 2 clients
        fit_metrics_aggregation_fn=weighted_train_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    return ServerAppComponents(         # fix: tuple → ServerAppComponents
        strategy=strategy,
        config=ServerConfig(num_rounds=20),
    )

app = ServerApp(server_fn=server_fn)