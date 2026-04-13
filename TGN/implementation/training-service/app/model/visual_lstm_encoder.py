from typing import List

import torch
from torch.nn import LSTM
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class VisualLSTMEncoder(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.encoder = LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=False)

    def forward(self, x: torch.Tensor, lengths: List[int]) -> torch.Tensor:
        x = x.permute(1, 0, 2)
        x = pack(x, lengths, enforce_sorted=False)
        enc_hiddens, _ = self.encoder(x)
        enc_hiddens, _ = unpack(enc_hiddens)
        return enc_hiddens.permute(1, 0, 2)
