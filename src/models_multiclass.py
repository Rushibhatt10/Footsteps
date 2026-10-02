"""
Multi-Class Audio Event Classification Neural Networks.
Designed specifically to discriminate between Footsteps, Claps, Knocks, and Other audio.

Features:
- 3-channel input: Log-Mel (80 bands) + Delta (velocity) + Delta-Delta (acceleration)
- Depthwise-separable convolutions for efficient CPU real-time inference (< 5 ms)
- Squeeze-and-Excitation (SE) channel attention to adaptively focus on transient/resonant frequency bands
- Concatenated GAP (Global Average Pooling) + GMP (Global Max Pooling) to capture both sustained envelope and peak impulse
- 4 output classes: [0: OTHER, 1: FOOTSTEP, 2: CLAP, 3: KNOCK]
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable convolution block: DepthwiseConv -> BN -> GELU -> PointwiseConv -> BN."""
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, kernel_size=3, stride=stride, padding=1, groups=in_c, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.bn_dw(self.dw(x)))
        x = F.gelu(self.bn_pw(self.pw(x)))
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4), bias=False),
            nn.GELU(),
            nn.Linear(max(channels // reduction, 4), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.gap(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class MultiClassResidualBlock(nn.Module):
    """Residual block with Depthwise-Separable Convolutions and SE Attention."""
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.ds_conv = DepthwiseSeparableConv(in_c, out_c, stride=stride)
        self.se = SEBlock(out_c)
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.se(self.ds_conv(x))
        return F.gelu(out + self.shortcut(x))


class MultiClassAudioNet(nn.Module):
    """
    4-Class Audio Event Classifier:
    0: OTHER
    1: FOOTSTEP
    2: CLAP
    3: KNOCK
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 4, dropout: float = 0.35):
        super().__init__()
        self.num_classes = num_classes

        # Stem: 3x3 conv to learn initial multi-channel representation
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )

        # 3 Stages with progressive spatial downsampling (stride=2)
        self.stage1 = MultiClassResidualBlock(32, 48, stride=2)   # 80x47 -> 40x24
        self.stage2 = MultiClassResidualBlock(48, 64, stride=2)   # 40x24 -> 20x12
        self.stage3 = MultiClassResidualBlock(64, 96, stride=2)   # 20x12 -> 10x6

        # Multi-scale pooling: Global Average Pooling + Global Max Pooling concatenated
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))

        # Classification Head: 2 * 96 = 192 features -> 96 -> num_classes
        self.head = nn.Sequential(
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(96, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        avg_pool = self.gap(x).flatten(1)
        max_pool = self.gmp(x).flatten(1)
        feat = torch.cat([avg_pool, max_pool], dim=1)
        logits = self.head(feat)
        return logits

    def predict_proba(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Returns softmax probabilities for each class."""
        logits = self.forward(x)
        return F.softmax(logits / max(temperature, 1e-4), dim=-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = MultiClassAudioNet(in_channels=3, num_classes=4)
    dummy = torch.randn(4, 3, 80, 47)
    logits = net(dummy)
    probs = net.predict_proba(dummy)
    print("MultiClassAudioNet Parameter Count:", count_parameters(net))
    print("Input shape:", dummy.shape)
    print("Logits shape:", logits.shape)
    print("Probs shape:", probs.shape)
    print("Probs sum across classes:", probs.sum(dim=-1).detach().numpy())
