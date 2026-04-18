import torch
import torch.nn as nn
from torchvision.models import vgg16


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
