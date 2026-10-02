"""
Neural Network Architectures for Footstep vs Non-Footstep Detection.
Includes:
1. BaselineFootstepCNN - Canonical Conv2D + BatchNorm + MaxPool + GAP + Dense architecture.
2. ResAudioNet - Residual Audio ConvNet with Channel Attention (Squeeze-and-Excitation).
3. EfficientFootstepNet - Lightweight depthwise-separable CNN with SE attention and
   multi-scale pooling. Accepts 3-channel (mel + delta + delta²) input for V3 training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. BASELINE CNN (Section 13)
# =====================================================================
class BaselineFootstepCNN(nn.Module):
    """
    Standard CNN Baseline adhering to Section 13 specification:
    Input -> Conv2D -> BatchNorm -> ReLU -> MaxPool
          -> Conv2D -> BatchNorm -> ReLU -> MaxPool
          -> Conv2D -> BatchNorm -> ReLU
          -> GlobalAveragePooling -> Dense -> Dropout -> Output (1 logit)
    """
    def __init__(self, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        # Head
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits.squeeze(-1)


# =====================================================================
# 2. IMPROVED MODEL: ResAudioNet with Squeeze-and-Excitation (Section 14)
# =====================================================================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        weights = self.fc(x).view(b, c, 1, 1)
        return x * weights


class ResAudioBlock(nn.Module):
    """Residual block with 2 Conv2D layers and SE attention."""
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.se = SEBlock(out_c)

        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.relu(out + residual)
        return out


class ResAudioNet(nn.Module):
    """
    Improved Audio Classifier designed for low-latency CPU real-time inference
    while suppressing transient false positives via SE attention and residual feature maps.
    """
    def __init__(self, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.layer1 = ResAudioBlock(32, 48, stride=2)
        self.layer2 = ResAudioBlock(48, 64, stride=2)
        self.layer3 = ResAudioBlock(64, 128, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        logits = self.head(x)
        return logits.squeeze(-1)


# =====================================================================
# 3. EFFICIENT FOOTSTEP NET (V3) — Depthwise Separable + SE + Multi-Scale Pool
# =====================================================================
class DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable convolution block: DepthwiseConv → BN → GELU → PointwiseConv → BN."""
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


class SEBlockV3(nn.Module):
    """Squeeze-and-Excitation channel attention (V3 variant)."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4), bias=False),
            nn.GELU(),
            nn.Linear(max(channels // reduction, 4), channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.gap(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class EfficientFootstepBlock(nn.Module):
    """DS-Conv block with residual connection and SE attention."""
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.ds_conv = DepthwiseSeparableConv(in_c, out_c, stride=stride)
        self.se = SEBlockV3(out_c)
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.se(self.ds_conv(x))
        return F.gelu(out + self.shortcut(x))


class EfficientFootstepNet(nn.Module):
    """
    V3 Architecture: Efficient depthwise-separable CNN for 3-channel
    (mel + delta + delta²) 80-band spectrogram input.

    Key design choices:
    - 3-channel input captures both spectral AND temporal dynamics
    - Depthwise-separable convs: lightweight, real-time capable on CPU
    - SE attention: reweights channel importance per frame
    - Multi-scale pooling (GAP + GMP concatenated): richer summary statistics
      than GAP alone — preserves peak energy information for transients
    - GELU activations: smoother gradients vs ReLU, better for soft-threshold tasks

    Target: < 3ms CPU inference, > 82% recording-level F1.
    """
    def __init__(self, in_channels: int = 3, dropout: float = 0.35):
        super().__init__()
        # Stem: standard 3×3 conv to learn multi-channel fusion
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        # Three EfficientFootstepBlocks with progressive stride-2 downsampling
        self.block1 = EfficientFootstepBlock(32, 48, stride=2)
        self.block2 = EfficientFootstepBlock(48, 64, stride=2)
        self.block3 = EfficientFootstepBlock(64, 96, stride=2)

        # Multi-scale pooling: GAP + GMP concatenated → 2×96 = 192 features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(96, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # Multi-scale pool: average + max, then concatenate
        avg = self.gap(x).flatten(1)  # (B, 96)
        mx  = self.gmp(x).flatten(1)  # (B, 96)
        x = torch.cat([avg, mx], dim=1)  # (B, 192)
        return self.head(x).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    """Returns total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(model_name: str = "baseline", device: str = "cpu") -> nn.Module:
    """Factory function for instantiating models."""
    if model_name.lower() == "baseline":
        model = BaselineFootstepCNN()
    elif model_name.lower() in ("resaudionet", "improved"):
        model = ResAudioNet()
    elif model_name.lower() in ("efficientfootstepnet", "v3", "efficient"):
        model = EfficientFootstepNet()
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose from: baseline, resaudionet, efficientfootstepnet")
    return model.to(device)
