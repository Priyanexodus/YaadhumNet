import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split


# ── Transforms ──────────────────────────────────────────────────
def get_transforms():
    return T.Compose([
        T.Resize((224, 224)),                 # fix: force exact square for ViT
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225))
    ])

# ── Main DataLoader Function ─────────────────────────────────────
def get_dataloader(
    data_path: str,                     # path to this client's local dataset
    split: str = "train",               # "train" or "val"
    batch_size: int = 4,
    val_ratio: float = 0.2,             # 80/20 train-val split
) -> DataLoader:
    """
    Returns a DataLoader for the client's local dataset.

    Expected data_path structure (standard ImageFolder format):
        data/
        ├── class_1/
        │   ├── img1.jpg
        │   └── img2.jpg
        ├── class_2/
        │   └── img3.jpg
        └── ...

    Args:
        data_path:   Root directory of this client's local dataset
        split:       "train" or "val"
        batch_size:  Batch size
        val_ratio:   Fraction of data reserved for validation (default 20%)

    Returns:
        DataLoader for this client's local train or val split
    """
    dataset = ImageFolder(root=data_path, transform=get_transforms())

    # Split into train / val
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size

    train_subset, val_subset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # reproducible split
    )

    is_train = split == "train"
    chosen = train_subset if is_train else val_subset

    return DataLoader(
        chosen,
        batch_size=batch_size,
        shuffle=is_train,       # shuffle train, not val
        drop_last=is_train,     # drop last batch only for train
        num_workers=2,
        pin_memory=True,
    )