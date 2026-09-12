"""Whole-board MLPs: an explicit encoding, hidden layers, four action values.

Relational features retain absolute exponents: doubling tiles preserves local
differences, but is NOT a symmetry of rewards and the fixed 2/4 spawn process.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ResidualLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, x):
        return (x + F.relu(self.linear(x))) / (2 ** .5)


class MLPQNetwork(nn.Module):
    def __init__(self, width=512, input_encoding='exponents', depth=2,
                 embedding_dim=16, residual=False):
        super().__init__()
        if depth < 1 or width < 1 or embedding_dim < 1:
            raise ValueError('Network dimensions must be positive.')
        self.architecture = 'mlp_q'
        self.input_encoding = input_encoding
        self.depth, self.embedding_dim, self.residual = depth, embedding_dim, residual
        # Eighteen categories: empty through rank 17. Absolute features in the
        # relational encoding remain unbounded; categorical inputs cap at 17.
        sizes = {'exponents':16, 'one_hot':288, 'embedding':16*embedding_dim,
                 'relational':80}
        if input_encoding not in sizes:
            raise ValueError('Unknown MLP encoding.')
        self.embedding = nn.Embedding(18, embedding_dim) if input_encoding == 'embedding' else None
        pairs = [(r*4+c,r*4+c+1) for r in range(4) for c in range(3)]
        pairs += [(r*4+c,(r+1)*4+c) for r in range(3) for c in range(4)]
        self.register_buffer('edge_indices', torch.tensor(pairs).T)
        layers = [nn.Linear(sizes[input_encoding], width), nn.ReLU()]
        for _ in range(depth-1):
            layers += [ResidualLayer(width)] if residual else [nn.Linear(width,width),nn.ReLU()]
        layers += [nn.Linear(width,4)]
        self.layers = nn.Sequential(*layers)

    def encode_inputs(self, boards):
        b = boards.reshape(-1,16)
        if self.input_encoding == 'exponents':
            return b.float()/16
        if self.input_encoding == 'one_hot':
            return F.one_hot(b.long().clamp(0,17),18).float().flatten(1)
        if self.input_encoding == 'embedding':
            return self.embedding(b.long().clamp(0,17)).flatten(1)
        left, right = b[:,self.edge_indices[0]], b[:,self.edge_indices[1]]
        occupied = (left>0)&(right>0)
        differences = (left.float()-right.float())*occupied/16
        equal = ((left==right)&occupied).float()
        return torch.cat((b.float()/16, (b==0).float(), differences, equal),1)

    def forward(self, boards):
        return self.layers(self.encode_inputs(boards))
