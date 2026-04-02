[![CI](https://github.com/TheGalaxyHunter/paper-implementations/actions/workflows/ci.yml/badge.svg)](https://github.com/TheGalaxyHunter/paper-implementations/actions/workflows/ci.yml)

# paper-implementations

**Clean PyTorch implementations of influential research papers.**

Focused implementations of papers I find foundational to modern deep learning.
Each folder is self-contained with the model, training script, notes on the paper,
and (where possible) reproduced results.

The goal is clarity over cleverness: every implementation is written from scratch,
well-commented, and designed to be read alongside the original paper.

## Implemented Papers

| Paper | Area | Year | Key Contribution |
|-------|------|------|------------------|
| [Attention Is All You Need](papers/attention-is-all-you-need/) | NLP / Sequence Modeling | 2017 | Replaced recurrence with self-attention; introduced the Transformer architecture |
| [BERT: Pre-training of Deep Bidirectional Transformers](papers/bert-pretraining/) | NLU / Pre-training | 2019 | Bidirectional pre-training with masked language modeling; set new SOTA across NLU benchmarks |
| [U-Net: Convolutional Networks for Biomedical Image Segmentation](papers/unet-segmentation/) | Medical Imaging | 2015 | Encoder-decoder with skip connections for precise segmentation with limited training data |

## Repository Structure

```
papers/
├── attention-is-all-you-need/
│   ├── model.py        # Full Transformer (encoder + decoder)
│   ├── train.py        # Training script
│   ├── config.yaml     # Hyperparameters
│   ├── notes.md        # Key insights and implementation notes
│   └── README.md       # Paper summary
├── bert-pretraining/
│   ├── model.py        # BERT architecture
│   ├── pretrain.py     # MLM + NSP pre-training
│   ├── finetune.py     # Fine-tuning for classification
│   ├── config.yaml
│   ├── notes.md
│   └── README.md
└── unet-segmentation/
    ├── model.py        # U-Net architecture
    ├── train.py        # Training with Dice loss
    ├── config.yaml
    ├── notes.md
    └── README.md
```

## Getting Started

### Prerequisites

- Python >= 3.10
- PyTorch >= 2.0

### Installation

```bash
git clone https://github.com/TheGalaxyHunter/paper-implementations.git
cd paper-implementations
pip install -e .
```

### Running an Implementation

Each paper folder is self-contained. For example:

```bash
cd papers/attention-is-all-you-need
python train.py
```

Check each paper's `README.md` for specific instructions and expected results.

## Design Principles

1. **From scratch**: No wrapping of library calls. Every layer is implemented explicitly.
2. **Readable**: Code is written to be read alongside the paper. Comments reference specific sections and equations.
3. **Self-contained**: Each folder runs independently. No shared utility modules that obscure logic.
4. **Typed**: Full type hints throughout for clarity and IDE support.

## Contributing

Contributions are welcome. If you'd like to add a paper implementation:

1. Fork the repository
2. Create a new folder under `papers/` with the paper name (kebab-case)
3. Include at minimum: `model.py`, `train.py`, `config.yaml`, `notes.md`, and `README.md`
4. Follow the existing code style: type hints, docstrings, paper references in comments
5. Open a pull request with a brief summary of the paper and your implementation approach

## Planned Papers

- ResNet (He et al., 2015)
- GPT-2 (Radford et al., 2019)
- Vision Transformer / ViT (Dosovitskiy et al., 2020)
- Diffusion Models (Ho et al., 2020)

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

**Ankit Das** - [@TheGalaxyHunter](https://github.com/TheGalaxyHunter)
