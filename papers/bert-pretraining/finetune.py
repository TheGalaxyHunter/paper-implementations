"""
BERT Fine-tuning Script for Sequence Classification.

Demonstrates the fine-tuning paradigm: take a pre-trained BERT model,
add a classification head, and train end-to-end on a downstream task.

Uses synthetic data for demonstration. Replace with a real dataset
(e.g., SST-2, MNLI) for meaningful results.
"""

import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

import yaml

from model import BertForSequenceClassification


class SyntheticClassificationDataset(Dataset):
    """Synthetic binary classification dataset.

    Generates random sequences with labels. In practice, replace with
    a real dataset and proper tokenization.
    """

    CLS_TOKEN = 2
    SEP_TOKEN = 3

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int) -> None:
        self.samples = []
        for _ in range(num_samples):
            length = random.randint(5, seq_len - 2)
            tokens = torch.randint(5, vocab_size, (length,))
            # Construct input: [CLS] tokens [SEP]
            input_ids = torch.cat([
                torch.tensor([self.CLS_TOKEN]),
                tokens,
                torch.tensor([self.SEP_TOKEN]),
            ])
            # Pad to fixed length
            padding = torch.zeros(seq_len - len(input_ids), dtype=torch.long)
            attention_mask = torch.cat([
                torch.ones(len(input_ids), dtype=torch.long),
                torch.zeros(len(padding), dtype=torch.long),
            ])
            input_ids = torch.cat([input_ids, padding])

            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "label": torch.tensor(random.randint(0, 1), dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def finetune(config_path: str = "config.yaml") -> None:
    """Fine-tuning loop for sequence classification."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    dataset = SyntheticClassificationDataset(
        vocab_size=config["model"]["vocab_size"],
        seq_len=config["finetuning"]["max_seq_len"],
        num_samples=2000,
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config["finetuning"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["finetuning"]["batch_size"])

    # Model
    model = BertForSequenceClassification(
        config["model"],
        num_classes=config["finetuning"]["num_classes"],
    ).to(device)

    # Load pre-trained weights if available
    pretrained_path = Path("checkpoints/bert_pretrained.pt")
    if pretrained_path.exists():
        print("Loading pre-trained BERT weights...")
        model.bert.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True))
    else:
        print("No pre-trained weights found. Training from scratch.")

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    # Optimizer with lower learning rate for fine-tuning
    optimizer = AdamW(
        model.parameters(),
        lr=config["finetuning"]["learning_rate"],
        weight_decay=0.01,
    )
    criterion = nn.CrossEntropyLoss()

    # Training
    best_val_acc = 0.0

    for epoch in range(1, config["finetuning"]["max_epochs"] + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                logits = model(input_ids, attention_mask=attention_mask)
                predictions = logits.argmax(dim=-1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/bert_finetuned.pt")
            print(f"  Saved best model (val_acc={best_val_acc:.4f})")


if __name__ == "__main__":
    finetune()
