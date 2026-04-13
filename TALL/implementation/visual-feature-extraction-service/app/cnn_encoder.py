from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn


class C3D(nn.Module):
    """C3D network with access to the fc6 embedding used by original TALL."""

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 487)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward_fc6(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.reshape(-1, 8192)
        return self.relu(self.fc6(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.forward_fc6(x)
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)
        return self.fc8(h)


def _extract_state_dict(payload: object) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        if "state_dict" in payload and isinstance(payload["state_dict"], Mapping):
            return payload["state_dict"]
        return payload  # type: ignore[return-value]
    raise ValueError("unsupported C3D checkpoint format")


def _normalize_state_dict_keys(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized[key.removeprefix("module.")] = value
    return normalized


class C3DFeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = C3D()

    def load_weights(self, weights_path: str, map_location: str | torch.device = "cpu") -> None:
        payload = torch.load(weights_path, map_location=map_location)
        state_dict = _normalize_state_dict_keys(_extract_state_dict(payload))
        self.model.load_state_dict(state_dict, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model.forward_fc6(x)
