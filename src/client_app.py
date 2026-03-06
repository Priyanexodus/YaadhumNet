from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from src.model import MOONModel, ViTEncoder, MOON_contrastive_loss
from moon_fl.task import get_dataloader
import mlflow, torch, copy
import torch.nn.functional as F

class MOONClient(NumPyClient):
    def __init__(self, model, train_loader, device, mu, temperature):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.mu = mu
        self.temperature = temperature
        self.previous_model = None

    def get_parameters(self, config):
        """Return the current local model parameters"""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self,parameters):
        """Set the current local model parameters"""

        keys = list(self.model.state_dict().keys())
        state = dict(zip(keys, [torch.tensor(val) for val in parameters]))
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)

    def fit(self, parameters, config):
        """The function is used to train the model."""

        # 1. Setup global and previous models
        global_model = copy.deepcopy(self.model)
        self.set_parameters(parameters)
        global_model.load_state_dict(self.model.state_dict())
        global_model.eval() # Good practice to freeze the global model

        if self.previous_model == None:
            self.previous_model = copy.deepcopy(self.model)
        self.previous_model.eval() # Good practice to freeze the previous model

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
        self.model.train()

        total = correct = 0
        running_loss = 0.0

        # 2. Single training loop
        for _ in range(config.get("local_epochs",10)):
            for x,y in self.train_loader:
                x,y = x.to(self.device), y.to(self.device)

                # Forward pass
                logits,z = self.model(x)
                loss_sup = F.cross_entropy(logits, y)

                with torch.no_grad():
                    _, z_glob = global_model(x)
                    _, Z_prev = self.previous_model(x)

                # Calculate MOON loss
                loss_con = MOON_contrastive_loss(z, z_glob, Z_prev, self.temperature)
                loss = loss_sup + self.mu * loss_con

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track metrics
                with torch.no_grad():
                    correct += (logits.argmax(1) == y).sum().item()
                    total += y.size(0)
                    running_loss += loss.item()

        # 3. Update previous model for the next round
        self.previous_model = copy.deepcopy(self.model)

        train_acc = correct / total if total > 0 else 0.0

        # Return results
        return self.get_parameters(config={}), len(self.train_loader.dataset), {
            "train_accuracy": train_acc,
            "train_loss": running_loss / total,
        }


def evaluate(self, parameters, config):

    self.set_parameters(parameters)
    self.model.eval()
    loss = total = correct = 0

    with torch.no_grad(): # Added parentheses to torch.no_grad
        for x,y in self.train_loader:
            x,y = x.to(self.device), y.to(self.device)
            logits, _ = self.model(x)
            loss += F.cross_entropy(logits, y, reduction="sum").item() # Added reduction="sum" and .item() to loss
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

    return float(loss), total, {"accuracy": correct/total}


def client_fn(context: Context):    
    partition_id = context.node_config['partition-id']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MOONModel(ViTEncoder(), proj_dim=256, num_classes=10).to(device)
    loader = get_dataloader(partition_id) # loads local data  
    return MOONClient(model, loader, device, mu=5, temperature=0.5).to_client()

app = ClientApp(client_fn=client_fn)
