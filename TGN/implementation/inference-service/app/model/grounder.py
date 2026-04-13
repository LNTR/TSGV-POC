import torch
from torch.nn import Linear


class Grounder(torch.nn.Module):
    def __init__(self, input_size: int, k_scales: int):
        super().__init__()
        self.projection = Linear(input_size, k_scales, bias=True)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.projection(input_tensor))
