"""Qudmi v0 model: a small transformer encoder over a window of sparse tracker signal,
predicting the full-body pose at the window's last (most recent) frame.

No causal masking is needed: every frame in the input window is already at-or-before the
frame we're predicting (the windowing in qudmi.data.amass always anchors on the last frame),
so full bidirectional attention across the window is safe and lets the model use the whole
window's context to disambiguate the current pose (e.g. distinguishing a fast arm swing from
a slow one needs a few frames of history either way).
"""

import torch
from torch import nn

from qudmi.data.amass import INPUT_DIM, OUTPUT_DIM


class QudmiTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        output_dim: int = OUTPUT_DIM,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 64,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, window, input_dim) -> (batch, output_dim), pose at the last frame."""
        T = x.shape[1]
        h = self.input_proj(x) + self.pos_embedding[:, :T]
        h = self.encoder(h)
        return self.output_head(h[:, -1])

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
