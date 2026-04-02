"""
Training script for U-Net segmentation.

Implements training with a combined Dice + BCE loss function, which is
standard for binary medical image segmentation. Uses synthetic data by
default for demonstration.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

import yaml

from model import UNet


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation.

    Directly optimizes the Dice coefficient (2 * |X intersect Y| / (|X| + |Y|)),
    which measures the overlap between predicted and ground truth masks.

    This is more robust to class imbalance than cross-entropy alone. In medical
    imaging, the foreground (e.g., tumor) often occupies a tiny fraction of the
    image, so a model could achieve high accuracy by predicting all background.
    Dice loss prevents this by focusing on overlap quality.

    Args:
        smooth: Smoothing factor to avoid division by zero and stabilize gradients.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (batch, 1, H, W) sigmoid probabilities
            targets:     (batch, 1, H, W) binary ground truth

        Returns:
            Scalar Dice loss (1 - Dice coefficient)
        """
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )

        return 1.0 - dice


class CombinedLoss(nn.Module):
    """Combined BCE + Dice loss.

    BCE provides stable gradients throughout training, while Dice loss drives
    the model toward better overlap metrics. Using both together typically
    outperforms either loss alone.
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (batch, 1, H, W) raw model output (before sigmoid)
            targets: (batch, 1, H, W) binary ground truth
        """
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(torch.sigmoid(logits), targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class SyntheticSegmentationDataset(Dataset):
    """Synthetic dataset for demonstrating U-Net training.

    Generates random images with circular "objects" as segmentation targets.
    Replace with a real medical imaging dataset for meaningful results.
    """

    def __init__(self, num_samples: int, image_size: int, in_channels: int = 1) -> None:
        self.images = []
        self.masks = []

        for _ in range(num_samples):
            image = torch.randn(in_channels, image_size, image_size) * 0.3

            # Create a random circular mask as the segmentation target
            mask = torch.zeros(1, image_size, image_size)
            cx = torch.randint(image_size // 4, 3 * image_size // 4, (1,)).item()
            cy = torch.randint(image_size // 4, 3 * image_size // 4, (1,)).item()
            radius = torch.randint(image_size // 8, image_size // 4, (1,)).item()

            y_grid, x_grid = torch.meshgrid(
                torch.arange(image_size), torch.arange(image_size), indexing="ij"
            )
            circle = ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) < radius ** 2
            mask[0] = circle.float()

            # Make the object visible in the image
            image[:, circle] += 1.0

            self.images.append(image)
            self.masks.append(mask)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.masks[idx]


def dice_coefficient(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute the Dice coefficient for evaluation.

    Args:
        predictions: (batch, 1, H, W) sigmoid probabilities
        targets:     (batch, 1, H, W) binary ground truth
        threshold:   Binarization threshold for predictions

    Returns:
        Dice coefficient (0 to 1, higher is better)
    """
    preds = (predictions > threshold).float().view(-1)
    targs = targets.view(-1)
    intersection = (preds * targs).sum()
    return (2.0 * intersection / (preds.sum() + targs.sum() + 1e-8)).item()


def train(config_path: str = "config.yaml") -> None:
    """Main training loop."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    dataset = SyntheticSegmentationDataset(
        num_samples=config["data"]["num_samples"],
        image_size=config["training"]["image_size"],
        in_channels=config["model"]["in_channels"],
    )

    train_size = int(config["data"]["train_split"] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"])

    # Model
    model = UNet(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        features=config["model"]["features"],
        bilinear=config["model"]["bilinear"],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    # Optimizer and loss
    optimizer = Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["optimizer"]["weight_decay"],
    )

    criterion = CombinedLoss(
        bce_weight=config["training"]["bce_weight"],
        dice_weight=config["training"]["dice_weight"],
    )

    # Training loop
    best_dice = 0.0

    for epoch in range(1, config["training"]["max_epochs"] + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            images, masks = images.to(device), masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Validation
        model.eval()
        val_dice = 0.0
        val_batches = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                predictions = torch.sigmoid(logits)
                val_dice += dice_coefficient(predictions, masks)
                val_batches += 1

        avg_dice = val_dice / val_batches
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, val_dice={avg_dice:.4f}")

        if avg_dice > best_dice:
            best_dice = avg_dice
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best_unet.pt")
            print(f"  Saved best model (dice={best_dice:.4f})")


if __name__ == "__main__":
    train()
