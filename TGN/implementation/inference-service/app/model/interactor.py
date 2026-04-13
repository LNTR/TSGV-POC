import torch
from torch.nn import LSTMCell, Linear


class Interactor(torch.nn.Module):
    def __init__(self, hidden_size_textual: int, hidden_size_visual: int, hidden_size_ilstm: int):
        super().__init__()
        self.projection_S = Linear(hidden_size_textual, hidden_size_ilstm, bias=True)
        self.projection_V = Linear(hidden_size_visual, hidden_size_ilstm, bias=True)
        self.projection_R = Linear(hidden_size_ilstm, hidden_size_ilstm, bias=True)
        self.projection_w = Linear(hidden_size_ilstm, 1, bias=True)

        self.hidden_size_ilstm = hidden_size_ilstm
        self.iLSTM = LSTMCell(hidden_size_textual + hidden_size_visual, hidden_size_ilstm)

    def forward(self, h_v: torch.Tensor, h_s: torch.Tensor) -> torch.Tensor:
        n_batch, t_steps = h_v.shape[0], h_v.shape[1]

        h_r_prev = torch.zeros([n_batch, self.hidden_size_ilstm], device=self.device)
        c_r_prev = torch.zeros([n_batch, self.hidden_size_ilstm], device=self.device)

        outputs = []
        for t in range(t_steps):
            beta_t = self.projection_w(
                torch.tanh(
                    self.projection_R(h_r_prev).unsqueeze(dim=1)
                    + self.projection_S(h_s)
                    + self.projection_V(h_v[:, t, :]).unsqueeze(dim=1)
                )
            ).squeeze(dim=2)

            alpha_t = torch.softmax(beta_t, dim=1)
            h_t_s = torch.bmm(h_s.permute(0, 2, 1), alpha_t.unsqueeze(dim=2)).squeeze(dim=2)
            r_t = torch.cat([h_v[:, t, :], h_t_s], dim=1)

            h_r_prev, c_r_prev = self.iLSTM(r_t, (h_r_prev, c_r_prev))
            outputs.append(h_r_prev.unsqueeze(dim=1))

        return torch.cat(outputs, dim=1)

    @property
    def device(self) -> torch.device:
        return self.projection_S.weight.device
