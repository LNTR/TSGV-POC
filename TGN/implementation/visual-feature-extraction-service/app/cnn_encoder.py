import torch
import torch.nn as nn
from torchvision.models import resnet18, vgg16


class VGG16(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            from torchvision.models import VGG16_Weights

            weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = vgg16(weights=weights)
        except Exception:
            # Backward compatibility with older torchvision API
            self.model = vgg16(pretrained=pretrained, progress=True)

        features = list(self.model.classifier.children())[:-1]
        self.model.classifier = nn.Sequential(*features)

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model(x)


class ResNet18(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights

            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = resnet18(weights=weights)
        except Exception:
            # Backward compatibility with older torchvision API
            self.model = resnet18(pretrained=pretrained, progress=True)

        self.model.fc = nn.Identity()

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model(x)


def build_cnn_encoder(encoder: str, pretrained: bool = True) -> nn.Module:
    normalized = encoder.strip().lower()
    if normalized == "vgg16":
        return VGG16(pretrained=pretrained)
    # resnet18 is supported here as a replacement-effort probe for the thesis
    # work. The goal is to exercise backbone swapability behind the existing
    # API seam rather than to claim a better grounding model.
    if normalized == "resnet18":
        return ResNet18(pretrained=pretrained)
    raise ValueError(f"Unsupported encoder: {encoder}")
