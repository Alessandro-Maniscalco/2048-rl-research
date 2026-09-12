"""Whole-board MLP with 51 return quantiles for each of the four actions.

The input is the 16 cell exponents, or learned per-cell tile embeddings. There
are no engineered merge/corner/snake features. QR-DQN's update is shared with
the Transformer comparison; only this network architecture changes.
"""
import torch
from torch import nn

from rl2048.agents.mlp_q import MLPQNetwork


class QuantileMLP(MLPQNetwork):
    def __init__(self, width=800, input_encoding='exponents', depth=2,
                 embedding_dim=16, residual=False, num_quantiles=51):
        if input_encoding not in ('exponents', 'embedding'):
            raise ValueError('QR MLP uses simple exponents or learned embeddings only')
        if not isinstance(num_quantiles, int) or num_quantiles < 2:
            raise ValueError('Use at least two quantiles')
        super().__init__(width, input_encoding, depth, embedding_dim, residual)
        self.architecture = 'mlp_qr'
        self.outputs = 4
        self.num_quantiles = num_quantiles
        scalar = self.layers[-1]
        head = nn.Linear(width, 4 * num_quantiles)
        # Like the QR Transformer, start with the paired scalar model's
        # mean predictions; quantile-specific gradients separate them.
        with torch.no_grad():
            head.weight.copy_(scalar.weight.repeat_interleave(num_quantiles, dim=0))
            head.bias.copy_(scalar.bias.repeat_interleave(num_quantiles))
        self.layers[-1] = head

    def quantile_values(self, boards):
        return super().forward(boards).reshape(-1, 4, self.num_quantiles)

    def forward(self, boards):
        return self.quantile_values(boards).mean(-1)
