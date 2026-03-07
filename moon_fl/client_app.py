from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from moon_fl.model import MOONModel, ViTEncoder, MOON_contrastive_loss  # fix: src.model → moon_fl.model
from moon_fl.task import get_dataloader
import torch, copy
import torch.nn.functional as F


class MOONClient(NumPyClient):
    def __init__(self, model, train_loader, val_loader, device, mu, temperature):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.mu = mu
        self.temperature = temperature
        self.previous_model = None
        
    def get_parameters(self, config):
        """Return the current local model parameters."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        """Set the local model parameters from a list of numpy arrays."""
        keys = list(self.model.state_dict().keys())
        state = dict(zip(keys, [torch.tensor(val) for val in parameters]))
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)

    def fit(self, parameters, config):
        local_epochs = config.get("local_epochs", 5)
        mu           = config.get("mu", self.mu)
        temperature  = config.get("temperature", self.temperature)

        self.set_parameters(parameters)

        global_model = copy.deepcopy(self.model)
        global_model.eval()

        if self.previous_model is None:
            self.previous_model = copy.deepcopy(self.model)
        self.previous_model.eval()

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0.01, momentum=0.9, weight_decay=1e-5
        )
        self.model.train()

        total = correct = 0
        running_loss = 0.0

        for _ in range(local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                logits, z = self.model(x)
                loss_sup = F.cross_entropy(logits, y)

                with torch.no_grad():
                    _, z_glob = global_model(x)
                    _, z_prev = self.previous_model(x)

                loss_con = MOON_contrastive_loss(z, z_glob, z_prev, temperature)
                loss = loss_sup + mu * loss_con

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    correct += (logits.argmax(1) == y).sum().item()
                    total += y.size(0)
                    running_loss += loss.item() * y.size(0)

        self.previous_model = copy.deepcopy(self.model)

        return self.get_parameters(config={}), len(self.train_loader.dataset), {
            "train_accuracy": correct / total if total > 0 else 0.0,
            "train_loss":     running_loss / total if total > 0 else 0.0,
        }

    def evaluate(self, parameters, config):      # fix: indented inside class
        """Evaluate the model on the local validation set."""

        self.set_parameters(parameters)
        self.model.eval()

        loss = total = correct = 0

        with torch.no_grad():
            for x, y in self.val_loader:         # fix: use val_loader not train_loader
                x, y = x.to(self.device), y.to(self.device)
                logits, _ = self.model(x)
                loss += F.cross_entropy(logits, y, reduction="sum").item()
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)

        return float(loss / total), total, {"accuracy": correct / total}

def client_fn(context: Context):
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = context.node_config["data-path"]

    # Infer num_classes from the user's dataset folder structure
    train_loader = get_dataloader(data_path, split="train")
    val_loader   = get_dataloader(data_path, split="val")
    
    num_classes = len(train_loader.dataset.dataset.classes)  # ImageFolder.classes

    model = MOONModel(ViTEncoder(), proj_dim=256, num_classes=num_classes).to(device)  # ← dynamic

    return MOONClient(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        mu=5,
        temperature=0.5,
    ).to_client()

app = ClientApp(client_fn=client_fn)