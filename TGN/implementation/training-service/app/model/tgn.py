from typing import List

import numpy as np
import torch

from .grounder import Grounder
from .interactor import Interactor
from .textual_lstm_encoder import TextualLSTMEncoder
from .visual_lstm_encoder import VisualLSTMEncoder


class TGN(torch.nn.Module):
    def __init__(
        self,
        word_embed_size: int,
        hidden_size_textual: int,
        hidden_size_visual: int,
        hidden_size_ilstm: int,
        k_scales: int,
        visual_feature_size: int,
    ):
        super().__init__()

        self.word_embed_size = word_embed_size
        self.hidden_size_textual = hidden_size_textual
        self.hidden_size_visual = hidden_size_visual
        self.hidden_size_ilstm = hidden_size_ilstm
        self.k_scales = k_scales
        self.visual_feature_size = visual_feature_size

        self.textual_lstm_encoder = TextualLSTMEncoder(
            embed_size=word_embed_size,
            hidden_size=hidden_size_textual,
        )
        self.visual_lstm_encoder = VisualLSTMEncoder(
            input_size=visual_feature_size,
            hidden_size=hidden_size_visual,
        )
        self.interactor = Interactor(
            hidden_size_ilstm=hidden_size_ilstm,
            hidden_size_visual=hidden_size_visual,
            hidden_size_textual=hidden_size_textual,
        )
        self.grounder = Grounder(input_size=hidden_size_ilstm, k_scales=k_scales)

    def forward(self, features_v: List[torch.Tensor], textual_input: torch.Tensor, lengths_t: List[int]):
        lengths_v = [v.shape[0] for v in features_v]
        mask = self._generate_visual_mask(lengths_v)
        features_v_padded = self._pad_visual_data(features_v)

        h_s = self.textual_lstm_encoder(textual_input, lengths_t)
        h_v = self.visual_lstm_encoder(features_v_padded, lengths_v)
        h_r = self.interactor(h_v, h_s)

        probs = self.grounder(h_r)
        return probs, mask

    def _generate_visual_mask(self, lengths: List[int]):
        n_batch = len(lengths)
        max_len = int(np.max(lengths))
        mask = torch.ones(n_batch, max_len, self.k_scales)

        for i, length in enumerate(lengths):
            mask[i, length:, :] = 0

        return mask.to(self.device)

    def _pad_visual_data(self, visual_data: List[torch.Tensor]):
        feature_dim = visual_data[0].shape[1]
        max_len = int(np.max([v.shape[0] for v in visual_data]))

        padded = []
        for v in visual_data:
            pad = torch.zeros([max_len - v.shape[0], feature_dim], device=self.device)
            padded.append(torch.cat([v.to(self.device), pad]).unsqueeze(dim=0))

        return torch.cat(padded, dim=0)

    @property
    def device(self) -> torch.device:
        return self.grounder.projection.weight.device

    def save(self, path: str):
        params = {
            "args": {
                "word_embed_size": self.word_embed_size,
                "hidden_size_textual": self.hidden_size_textual,
                "hidden_size_visual": self.hidden_size_visual,
                "hidden_size_ilstm": self.hidden_size_ilstm,
                "k_scales": self.k_scales,
                "visual_feature_size": self.visual_feature_size,
            },
            "state_dict": self.state_dict(),
        }
        torch.save(params, path)
