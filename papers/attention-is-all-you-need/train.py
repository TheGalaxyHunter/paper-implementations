"""
Training script for the Transformer model.

Uses a synthetic copy task by default: the model learns to copy input sequences.
This is a standard sanity check for sequence-to-sequence models.
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

import yaml

from model import Transformer


class CopyDataset(Dataset):
    """Synthetic copy task: model must learn to reproduce the input sequence.

    This is a standard debugging/validation task for seq2seq models.
    """

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int) -> None:
        self.data = torch.randint(2, vocab_size, (num_samples, seq_len))
        # Token 0 = PAD, Token 1 = BOS/EOS

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.data[idx]


class TransformerLRScheduler:
    """Learning rate schedule from Section 5.3 of the paper.

    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    This increases linearly for the first warmup_steps, then decreases
    proportionally to the inverse square root of the step number.
    """

    def __init__(self, optimizer: Adam, d_model: int, warmup_steps: int = 4000) -> None:
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self) -> None:
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5),
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


class LabelSmoothingLoss(nn.Module):
    """Label smoothing cross-entropy loss (Section 5.4).

    Instead of one-hot targets, uses a smoothed distribution that assigns
    (1 - smoothing) to the correct class and distributes the remaining
    probability mass uniformly.
    """

    def __init__(self, vocab_size: int, padding_idx: int = 0, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch * seq_len, vocab_size)
            target: (batch * seq_len,)
        """
        logits = logits.view(-1, self.vocab_size)
        target = target.view(-1)

        log_probs = torch.log_softmax(logits, dim=-1)

        # Smooth distribution: uniform over all classes
        smooth_loss = -log_probs.sum(dim=-1) / self.vocab_size
        # NLL for the correct class
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)

        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        # Ignore padding positions
        mask = target != self.padding_idx
        loss = (loss * mask).sum() / mask.sum()

        return loss


def train(config_path: str = "config.yaml") -> None:
    """Main training loop."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    dataset = CopyDataset(
        vocab_size=config["data"]["vocab_size"],
        seq_len=config["data"]["seq_len"],
        num_samples=config["data"]["num_samples"],
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"])

    # Model
    model = Transformer(
        src_vocab_size=config["data"]["vocab_size"],
        tgt_vocab_size=config["data"]["vocab_size"],
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        n_encoder_layers=config["model"]["n_encoder_layers"],
        n_decoder_layers=config["model"]["n_decoder_layers"],
        d_ff=config["model"]["d_ff"],
        dropout=config["model"]["dropout"],
        max_seq_len=config["model"]["max_seq_len"],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    # Optimizer with the paper's learning rate schedule
    optimizer = Adam(
        model.parameters(),
        lr=0.0,
        betas=tuple(config["optimizer"]["betas"]),
        eps=config["optimizer"]["eps"],
    )
    scheduler = TransformerLRScheduler(
        optimizer,
        d_model=config["model"]["d_model"],
        warmup_steps=config["training"]["warmup_steps"],
    )

    criterion = LabelSmoothingLoss(
        vocab_size=config["data"]["vocab_size"],
        smoothing=config["training"]["label_smoothing"],
    )

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(1, config["training"]["max_epochs"] + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            src, tgt = src.to(device), tgt.to(device)

            # Decoder input: shift right (teacher forcing)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # Create causal mask for decoder
            tgt_mask = Transformer.generate_causal_mask(tgt_input.size(1), device=device)

            # Forward pass
            logits = model(src, tgt_input, tgt_mask=tgt_mask)
            loss = criterion(logits, tgt_output)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                tgt_mask = Transformer.generate_causal_mask(tgt_input.size(1), device=device)

                logits = model(src, tgt_input, tgt_mask=tgt_mask)
                loss = criterion(logits, tgt_output)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches

        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            print(f"  Saved best model (val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    train()
