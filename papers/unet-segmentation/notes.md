# Notes: U-Net for Biomedical Image Segmentation

## Why Skip Connections Matter

The fundamental tension in segmentation is between "what" and "where":

- **Deep layers** have large receptive fields and understand what objects are present,
  but their spatial resolution is low. They know "this is a cell" but not exactly where
  the boundary is.
- **Shallow layers** have high spatial resolution and capture edges, textures, and fine
  details, but they lack semantic understanding. They see boundaries but don't know what
  they belong to.

Skip connections solve this by concatenating shallow (high-resolution) feature maps with
deep (high-semantic) feature maps at each decoder level. The decoder then has access to
both precise spatial information AND semantic context.

This is fundamentally different from just using a deeper encoder. Adding more layers
increases the receptive field but inevitably loses spatial precision through pooling.
Skip connections let you recover that precision.

## Encoder-Decoder Symmetry

U-Net's architecture is almost perfectly symmetric:

- **Encoder (contracting path)**: Each level doubles the channels and halves the spatial
  resolution. This follows the standard pattern of increasing abstraction.
- **Decoder (expanding path)**: Each level halves the channels and doubles the spatial
  resolution. Transposed convolutions (or bilinear upsampling) handle the upscaling.

At each decoder level, the feature maps from the corresponding encoder level are
concatenated (not added, concatenated). This gives the decoder a richer feature set:
the upsampled deep features provide context, and the skip-connected shallow features
provide spatial precision.

## Why Concatenation, Not Addition?

U-Net concatenates encoder features with decoder features (doubling the channel count),
rather than adding them (keeping channels the same). This is a deliberate choice:

- **Concatenation** preserves all information from both sources. The subsequent
  convolutions learn how to combine them.
- **Addition** forces the features to be in the same "space" and can lose information
  through destructive interference.

Later architectures (like ResNet-based U-Nets) sometimes use addition (residual
connections), but the original U-Net's concatenation approach remains common in
medical imaging.

## Working with Limited Training Data

U-Net was designed for a specific constraint in medical imaging: very few annotated
training images (sometimes fewer than 30). Two strategies make this work:

1. **Heavy data augmentation**: The paper emphasizes elastic deformations, which simulate
   realistic tissue deformations. This is far more effective than simple rotations/flips
   for medical images, because tissue has inherent variability that elastic transforms
   capture well.

2. **Overlap-tile strategy**: For large images, U-Net predicts overlapping tiles. The
   context for border pixels comes from the surrounding area (with mirror padding at
   image boundaries). This means every pixel gets a prediction informed by full context.

## Loss Function: Dice vs. Cross-Entropy

The original paper uses pixel-wise cross-entropy with a weight map that emphasizes
boundaries between touching cells. In practice, many implementations combine:

- **Binary Cross-Entropy (BCE)**: Standard pixel-wise classification loss
- **Dice Loss**: Directly optimizes the Dice coefficient (overlap metric), which is
  more robust to class imbalance

Class imbalance is extremely common in medical segmentation (e.g., a tumor might occupy
2% of the image). BCE alone would let the model achieve 98% accuracy by predicting
"background" everywhere. Dice loss penalizes this because it considers the overlap
between predicted and ground truth masks.

The combination `BCE + Dice` tends to work best: BCE provides stable gradients early
in training, while Dice loss drives the model toward better overlap metrics.

## Batch Normalization

The original U-Net paper does not use batch normalization (it predates BN becoming
standard in segmentation). Modern implementations almost always add BN after each
convolution, which significantly improves training stability and convergence speed.

This implementation includes BN as it's considered essential in practice.

## Implementation Observations

- The original paper uses "valid" convolutions (no padding), which causes the output
  to be smaller than the input. Modern implementations typically use "same" padding
  to keep dimensions consistent, simplifying the skip connections.
- The number of channels (64, 128, 256, 512, 1024) follows a simple doubling pattern.
  This can be adjusted for computational constraints.
- For multi-class segmentation, change the final 1x1 convolution to output N channels
  (one per class) and use cross-entropy loss instead of BCE.
