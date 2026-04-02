# U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)

**Paper**: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

**Authors**: Ronneberger, Fischer, Brox (University of Freiburg)

## Summary

U-Net introduced an encoder-decoder architecture with skip connections specifically designed
for biomedical image segmentation. The key insight is that combining high-resolution features
from the encoder (contracting path) with upsampled features from the decoder (expanding path)
allows precise localization while retaining semantic context. The architecture works remarkably
well with very few training images, which is critical in medical imaging where annotated data
is scarce.

## Key Contributions

- **Symmetric encoder-decoder architecture**: The "U" shape with contracting and expanding paths
  that mirror each other
- **Skip connections**: Concatenate feature maps from the encoder to the decoder at corresponding
  resolution levels, preserving fine-grained spatial information
- **Works with limited data**: Achieves strong results with as few as 30 training images, using
  heavy data augmentation (elastic deformations)
- **Overlap-tile strategy**: Enables seamless segmentation of arbitrarily large images by
  predicting overlapping tiles

## Implementation

| File | Description |
|------|-------------|
| `model.py` | U-Net architecture with configurable depth and channels |
| `train.py` | Training loop with Dice loss + BCE |
| `config.yaml` | Hyperparameters and architecture settings |
| `notes.md` | Analysis of skip connections and design decisions |

## Running

```bash
python train.py
```

Uses randomly generated data by default. Replace the dataset with real medical
images for meaningful training.

## Architecture Overview

```
Input (1, 572, 572)
    |
[Encoder]
    Conv 3x3 -> Conv 3x3 -> 64 channels
    MaxPool 2x2
    Conv 3x3 -> Conv 3x3 -> 128 channels
    MaxPool 2x2
    Conv 3x3 -> Conv 3x3 -> 256 channels
    MaxPool 2x2
    Conv 3x3 -> Conv 3x3 -> 512 channels
    MaxPool 2x2
    |
[Bottleneck]
    Conv 3x3 -> Conv 3x3 -> 1024 channels
    |
[Decoder]
    UpConv 2x2 -> Concat skip -> Conv 3x3 -> Conv 3x3 -> 512 channels
    UpConv 2x2 -> Concat skip -> Conv 3x3 -> Conv 3x3 -> 256 channels
    UpConv 2x2 -> Concat skip -> Conv 3x3 -> Conv 3x3 -> 128 channels
    UpConv 2x2 -> Concat skip -> Conv 3x3 -> Conv 3x3 -> 64 channels
    |
Conv 1x1 -> num_classes
```

## References

- Original paper: https://arxiv.org/abs/1505.04597
- U-Net project page: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
