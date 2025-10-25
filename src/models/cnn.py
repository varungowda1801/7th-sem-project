from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        out = self.dropout(out)
        return out


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1, use_se: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )
        self.se = SEBlock(out_ch) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = self.se(x)
        return x


class ImprovedKannadaCNN(nn.Module):
    """Enhanced CNN architecture with residual connections, attention, and better regularization"""
    def __init__(self, in_channels: int = 1, embedding_dim: int = 256, num_classes: int = 49):
        super().__init__()
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1, dropout=0.1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, dropout=0.15)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, dropout=0.2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, dropout=0.25)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Enhanced classifier with multiple layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.BatchNorm1d(embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1, dropout: float = 0.1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, dropout))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, dropout))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.gap(x).flatten(1)
        logits = self.classifier(x)
        
        # For compatibility with existing code, return normalized features
        features = F.normalize(x, p=2, dim=1)
        return features, logits


class KannadaCNN(nn.Module):
    """Original CNN for backward compatibility"""
    def __init__(self, in_channels: int = 1, embedding_dim: int = 128, num_classes: int = 49):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, dropout=0.1),
            ConvBlock(32, 64, dropout=0.15),
            ConvBlock(64, 128, dropout=0.2),
            ConvBlock(128, 256, dropout=0.25),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Linear(256, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x)
        h = self.gap(h).flatten(1)
        z = F.normalize(self.embedding(h))
        logits = self.classifier(z)
        return z, logits


