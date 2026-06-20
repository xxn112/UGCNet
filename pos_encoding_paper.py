import math
import torch
import torch.nn as nn


class DepthPositionalEncoding(nn.Module):
    """TVDSS-based sinusoidal positional encoding (paper Section 3.3).

    Each depth point is encoded with its true TVDSS value so that the same
    physical depth produces a consistent positional representation across
    wells. depth_scale normalizes TVDSS to a stable range.
    """

    def __init__(self, d_model, dropout=0.1, depth_scale=1000.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.depth_scale = depth_scale
        self.d_model = d_model

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.register_buffer('div_term', div_term)

    def forward(self, x, depth):
        # x:     (B, L, d_model)
        # depth: (B, L)  TVDSS values
        depth_norm = depth.unsqueeze(-1) / self.depth_scale

        pe = torch.zeros_like(x)
        pe[..., 0::2] = torch.sin(depth_norm * self.div_term)
        if self.d_model % 2 == 1:
            pe[..., 1::2] = torch.cos(depth_norm * self.div_term[:pe[..., 1::2].shape[-1]])
        else:
            pe[..., 1::2] = torch.cos(depth_norm * self.div_term)

        return self.dropout(x + pe)
