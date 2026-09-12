"""Sixteen tile tokens -> Transformer -> one symmetric afterstate value.

The teacher is a source of scalar labels, never an input feature. A fixed
training-set mean/std converts the network's normalized output back to the
usual future-points/128 units. All eight board views share every learned weight.
"""
import numpy as np
import torch
from torch import nn

from rl2048.agents.transformer_q import TransformerQNetwork


class SymmetricAfterstateTransformer(TransformerQNetwork):
    def __init__(self, width=128, input_encoding='embedding', depth=4,
                 embedding_dim=None, heads=4, value_scale=1., value_offset=0.):
        if not np.isfinite(value_scale) or value_scale <= 0 or not np.isfinite(value_offset):
            raise ValueError('Value scale must be finite and positive; offset must be finite')
        if embedding_dim is not None and embedding_dim != width:
            raise ValueError('Transformer tile embedding dimension equals its token width')
        super().__init__(width,input_encoding,depth,heads)
        self.architecture='afterstate_sym_transformer'
        self.embedding_dim=width
        self.outputs=1
        self.readout=nn.Linear(16*width,1)
        nn.init.zeros_(self.readout.weight);nn.init.zeros_(self.readout.bias)
        cells=np.arange(16).reshape(4,4)
        self.register_buffer('symmetry_permutations',torch.tensor(np.stack(
            [np.rot90(cells,k).ravel() for k in range(4)]+
            [np.rot90(np.fliplr(cells),k).ravel() for k in range(4)])))
        self.register_buffer('value_scale',torch.tensor(float(value_scale)))
        self.register_buffer('value_offset',torch.tensor(float(value_offset)))

    def forward(self, boards):
        views=boards.reshape(-1,16)[:,self.symmetry_permutations].reshape(-1,16)
        normalized=super().forward(views).reshape(-1,8).mean(1)
        return self.value_offset+self.value_scale*normalized
