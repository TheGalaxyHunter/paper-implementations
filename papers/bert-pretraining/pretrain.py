"""
BERT Pre-training Script.

Implements the two pre-training objectives:
1. Masked Language Model (MLM): Predict randomly masked tokens
2. Next Sentence Prediction (NSP): Predict if sentence B follows A

Uses synthetic data for demonstration. Replace with real corpus for
meaningful pre-training.
"""

import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import yaml

from model import BertForPreTraining


class SyntheticPreTrainingDataset(Dataset):
    """Synthetic dataset for demonstrating BERT pre-training.

    Generates random token sequences with:
    - MLM: 15% of tokens masked (80% [MASK], 10% random, 10% unchanged)
    - NSP: 50% real pairs, 50% random pairs

    In practice, you would replace this with a real corpus (e.g., Wikipedia +
    BooksCorpus) with proper tokenization.
    """

    MASK_TOKEN = 4  # [MASK] token id
    CLS_TOKEN = 2   # [CLS] token id
    SEP_TOKEN = 3   # [SEP] token id

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        num_samples: int,
        mlm_probability: float = 0.15,
    ) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.mlm_probability = mlm_probability

        # Generate "sentences" as random token sequences
        half_len = (seq_len - 3) // 2  # Account for [CLS], [SEP], [SEP]
        self.sentences = [
            torch.randint(5, vocab_size, (half_len,)) for _ in range(num_samples * 2)
        ]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        # Build sentence pair
        sent_a = self.sentences[idx * 2]

        # 50% chance: real next sentence, 50% random
        is_next = random.random() < 0.5
        if is_next:
            sent_b = self.sentences[idx * 2 + 1]
        else:
            random_idx = random.randint(0, len(self.sentences) - 1)
            sent_b = self.sentences[random_idx]

        # Construct input: [CLS] sent_a [SEP] sent_b [SEP]
        tokens = torch.cat([
            torch.tensor([self.CLS_TOKEN]),
            sent_a,
            torch.tensor([self.SEP_TOKEN]),
            sent_b,
            torch.tensor([self.SEP_TOKEN]),
        ])

        # Segment IDs: 0 for sentence A (including [CLS] and first [SEP]), 1 for sentence B
        segment_ids = torch.cat([
            torch.zeros(len(sent_a) + 2, dtype=torch.long),
            torch.ones(len(sent_b) + 1, dtype=torch.long),
        ])

        # Apply MLM masking
        input_ids = tokens.clone()
        mlm_labels = torch.full_like(tokens, -100)  # -100 = ignore in loss

        for i in range(len(tokens)):
            if tokens[i] in (self.CLS_TOKEN, self.SEP_TOKEN):
                continue
            if random.random() < self.mlm_probability:
                mlm_labels[i] = tokens[i]
                rand = random.random()
                if rand < 0.8:
                    input_ids[i] = self.MASK_TOKEN
                elif rand < 0.9:
                    input_ids[i] = random.randint(5, self.vocab_size - 1)
                # else: keep original (10% of the time)

        return {
            "input_ids": input_ids,
            "segment_ids": segment_ids,
            "attention_mask": torch.ones_like(input_ids),
            "mlm_labels": mlm_labels,
            "nsp_label": torch.tensor(1 if is_next else 0, dtype=torch.long),
        }


def train(config_path: str = "config.yaml") -> None:
    """Main pre-training loop."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    dataset = SyntheticPreTrainingDataset(
        vocab_size=config["model"]["vocab_size"],
        seq_len=config["pretraining"]["max_seq_len"],
        num_samples=5000,
        mlm_probability=config["pretraining"]["mlm_probability"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["pretraining"]["batch_size"],
        shuffle=True,
    )

    # Model
    model = BertForPreTraining(config["model"]).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config["pretraining"]["learning_rate"],
        weight_decay=0.01,
    )

    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    nsp_criterion = nn.CrossEntropyLoss()

    # Training
    for epoch in range(1, config["pretraining"]["max_epochs"] + 1):
        model.train()
        total_mlm_loss = 0.0
        total_nsp_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}", leave=False):
            input_ids = batch["input_ids"].to(device)
            segment_ids = batch["segment_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mlm_labels = batch["mlm_labels"].to(device)
            nsp_labels = batch["nsp_label"].to(device)

            mlm_logits, nsp_logits = model(input_ids, segment_ids, attention_mask)

            mlm_loss = mlm_criterion(mlm_logits.view(-1, config["model"]["vocab_size"]), mlm_labels.view(-1))
            nsp_loss = nsp_criterion(nsp_logits, nsp_labels)
            loss = mlm_loss + nsp_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_mlm_loss += mlm_loss.item()
            total_nsp_loss += nsp_loss.item()
            num_batches += 1

        avg_mlm = total_mlm_loss / num_batches
        avg_nsp = total_nsp_loss / num_batches
        print(f"Epoch {epoch}: mlm_loss={avg_mlm:.4f}, nsp_loss={avg_nsp:.4f}")

    # Save pre-trained model
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.bert.state_dict(), "checkpoints/bert_pretrained.pt")
    print("Saved pre-trained BERT model.")


if __name__ == "__main__":
    train()
