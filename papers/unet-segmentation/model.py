"""
U-Net: Convolutional Networks for Biomedical Image Segmentation
(Ronneberger et al., 2015)
https://arxiv.org/abs/1505.04597

Full implementation of U-Net from scratch. The architecture follows the
original paper's encoder-decoder structure with skip connections, adapted
with modern practices (batch normalization, same-padding convolutions).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DoubleConv(nn.Module):
    """Two consecutive 3x3 convolutions, each followed by BatchNorm and ReLU.

    This is the basic building block of U-Net. Each level of the encoder and
    decoder applies this block. The original paper uses "valid" (unpadded)
    convolutions, but we use "same" padding for simplicity.

    Args:
        in_channels:  Number of input channels
        out_channels: Number of output channels
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class EncoderBlock(nn.Module):
    """Single encoder (contracting path) block.

    Applies max pooling to reduce spatial dimensions by 2x, then the double
    convolution block. The feature map before pooling is returned separately
    for the skip connection.

    Args:
        in_channels:  Number of input channels
        out_channels: Number of output channels
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, in_channels, H, W)
        Returns:
            (batch, out_channels, H/2, W/2)
        """
        return self.conv(self.pool(x))


class DecoderBlock(nn.Module):
    """Single decoder (expanding path) block.

    Upsamples the input by 2x (using transposed convolution or bilinear
    interpolation), concatenates the skip connection from the encoder,
    then applies the double convolution block.

    The concatenation of encoder features with upsampled decoder features
    is the key innovation of U-Net: it combines high-resolution spatial
    information (from the encoder) with deep semantic information (from
    the decoder).

    Args:
        in_channels:  Number of input channels (from deeper level)
        out_channels: Number of output channels
        bilinear:     If True, use bilinear upsampling. If False, use transposed conv.
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False) -> None:
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # After concatenation with skip, channels = in_channels + out_channels
            # But if bilinear, we reduce first
            self.conv = DoubleConv(in_channels + out_channels, out_channels)
        else:
            # Transposed convolution: learnable upsampling
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # After upsampling: in_channels//2; after concat with skip: in_channels//2 + out_channels
            self.conv = DoubleConv(in_channels // 2 + out_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        """
        Args:
            x:    (batch, in_channels, H, W) - from deeper decoder level
            skip: (batch, out_channels, 2H, 2W) - from corresponding encoder level

        Returns:
            (batch, out_channels, 2H, 2W)
        """
        x = self.up(x)

        # Handle potential size mismatch due to odd dimensions
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        # Concatenate along channel dimension (the skip connection)
        x = torch.cat([skip, x], dim=1)

        return self.conv(x)


class UNet(nn.Module):
    """U-Net: Convolutional Networks for Biomedical Image Segmentation.

    Encoder-decoder architecture with skip connections for precise segmentation.
    The encoder progressively downsamples while increasing channels, the decoder
    upsamples while decreasing channels, and skip connections carry fine-grained
    spatial information from encoder to decoder at each resolution level.

    Args:
        in_channels:  Number of input image channels (1 for grayscale, 3 for RGB)
        out_channels: Number of output channels (1 for binary, N for multi-class)
        features:     List of channel counts for each encoder level.
                      Default [64, 128, 256, 512] follows the original paper.
        bilinear:     Whether to use bilinear upsampling (True) or transposed
                      convolutions (False, default). The original paper uses
                      transposed convolutions.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: list[int] | None = None,
        bilinear: bool = False,
    ) -> None:
        super().__init__()

        if features is None:
            features = [64, 128, 256, 512]

        self.features = features
        self.bilinear = bilinear

        # Initial double convolution (no pooling)
        self.input_conv = DoubleConv(in_channels, features[0])

        # Encoder (contracting path)
        self.encoders = nn.ModuleList()
        for i in range(len(features) - 1):
            self.encoders.append(EncoderBlock(features[i], features[i + 1]))

        # Bottleneck: deepest level of the U
        bottleneck_channels = features[-1] * 2
        self.bottleneck = EncoderBlock(features[-1], bottleneck_channels)

        # Decoder (expanding path)
        self.decoders = nn.ModuleList()

        # First decoder takes bottleneck output and connects to last encoder level
        self.decoders.append(DecoderBlock(bottleneck_channels, features[-1], bilinear))

        # Remaining decoders
        for i in range(len(features) - 1, 0, -1):
            self.decoders.append(DecoderBlock(features[i], features[i - 1], bilinear))

        # Final 1x1 convolution to map features to output classes
        self.output_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, in_channels, H, W) input image

        Returns:
            (batch, out_channels, H, W) segmentation logits
        """
        # Store encoder outputs for skip connections
        skip_connections: list[Tensor] = []

        # Initial convolution
        x = self.input_conv(x)
        skip_connections.append(x)

        # Encoder path: progressively downsample
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path: progressively upsample with skip connections
        # Skip connections are used in reverse order (deepest first)
        skip_connections = skip_connections[::-1]

        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skip_connections[i])

        # Final classification layer
        return self.output_conv(x)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test: verify the architecture produces correct output shape
    model = UNet(in_channels=1, out_channels=1, features=[64, 128, 256, 512])
    x = torch.randn(1, 1, 256, 256)
    output = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters:   {count_parameters(model):,}")
