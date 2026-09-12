"""Learn local tile patterns before combining the complete 4x4 afterstate.

Three unpadded 2x2 convolutions reduce the board from 4x4 to 3x3 to 2x2
to 1x1. The final receptive field covers every tile. Filters are learned
and shared across locations; there are no supplied corner or snake features.
"""
import torch
from torch import nn
from torch.nn import functional as F


class AfterstateCNN(nn.Module):
    def __init__(self, width=224, input_encoding='embedding', depth=3, embedding_dim=16):
        super().__init__()
        if depth != 3 or width < 1 or embedding_dim < 1:
            raise ValueError('Three 2x2 layers cover a 4x4 board; dimensions must be positive')
        if input_encoding not in ('exponents', 'embedding', 'one_hot'):
            raise ValueError('Only tile inputs, no engineered relational features')
        self.architecture = 'afterstate_cnn'
        self.input_encoding = input_encoding
        self.depth, self.embedding_dim = depth, embedding_dim
        self.embedding = nn.Embedding(18, embedding_dim) if input_encoding == 'embedding' else None
        channels = {'exponents': 1, 'embedding': embedding_dim, 'one_hot': 18}[input_encoding]
        self.features = nn.Sequential(
            nn.Conv2d(channels, width, 2), nn.ReLU(),
            nn.Conv2d(width, width, 2), nn.ReLU(),
            nn.Conv2d(width, width, 2), nn.ReLU())
        self.readout = nn.Linear(width, 1)
        # Match the MLP: initially choose using immediate merge points alone.
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def encode_inputs(self, boards):
        b = boards.reshape(-1, 4, 4)
        if self.input_encoding == 'exponents':
            return b.float().unsqueeze(1) / 16
        ranks = b.long().clamp(0, 17)
        cells = self.embedding(ranks) if self.embedding is not None else F.one_hot(ranks, 18).float()
        return cells.permute(0, 3, 1, 2).contiguous()

    def forward(self, boards):
        whole_board = self.features(self.encode_inputs(boards)).flatten(1)
        return self.readout(whole_board).squeeze(-1)
