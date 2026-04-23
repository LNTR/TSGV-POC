from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CTRLConfig:
    visual_feature_size: int
    sentence_embedding_size: int
    semantic_size: int = 1024
    mlp_hidden_size: int = 1000
    max_window_scales: int = 8
    context_size: int = 1
    sample_every_sec: float = 5.0
    nms_threshold: float = 0.45


class CTRLTemporalLocalizer(nn.Module):
    def __init__(self, config: CTRLConfig):
        super().__init__()
        self.config = config
        self.visual_projection = nn.Linear(config.visual_feature_size, config.semantic_size)
        self.text_projection = nn.Linear(config.sentence_embedding_size, config.semantic_size)
        self.fusion = nn.Sequential(
            nn.Linear(config.semantic_size * 4, config.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(config.mlp_hidden_size, 3),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        candidate_features: torch.Tensor,
        sentence_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if candidate_features.ndim != 2:
            raise ValueError("candidate_features must have shape [num_candidates, feature_dim]")
        if sentence_embedding.ndim != 1:
            raise ValueError("sentence_embedding must have shape [feature_dim]")

        visual = F.normalize(self.visual_projection(candidate_features), dim=-1)
        text = F.normalize(self.text_projection(sentence_embedding).unsqueeze(0), dim=-1)
        text = text.expand_as(visual)

        fused = torch.cat([visual * text, visual + text, visual, text], dim=-1)
        outputs = self.fusion(fused)

        scores = outputs[:, 0]
        start_offsets = outputs[:, 1]
        end_offsets = outputs[:, 2]
        return scores, start_offsets, end_offsets

    def save(self, path: str) -> None:
        payload = {
            "state_dict": self.state_dict(),
            "config": asdict(self.config),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, map_location: str | torch.device = "cpu") -> "CTRLTemporalLocalizer":
        payload = torch.load(Path(path), map_location=map_location)
        if not isinstance(payload, dict):
            raise ValueError("saved TALL model must be a dictionary payload")
        config = CTRLConfig(**payload["config"])
        model = cls(config)
        model.load_state_dict(payload["state_dict"])
        return model
