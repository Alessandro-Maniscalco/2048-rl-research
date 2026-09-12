"""A four-logit CNN student of recorded search actions; no value/critic head.

Eighteen independent tile categories distinguish 32768, 65536 and 131072.
The original CNN's 16-category encoder can initialize this larger input without
changing its features on boards containing only ranks 0..15. Newly added rank
channels initially copy rank15, but their parameters subsequently learn apart.
"""
import torch
from torch import nn

from .ml2048_network import CNNEncoder, CNNActorNetwork


class NativePolicy(nn.Module):
    architecture = 'cnn_policy18'
    input_encoding = 'one_hot'
    outputs = 4
    width = 1024

    def __init__(self):
        super().__init__()
        self._encoder = CNNEncoder(1024, num_classes=18)
        self._actor = CNNActorNetwork(1024, 256, 64)

    def forward(self, boards):
        features = self._encoder(boards.long().reshape(-1, 16).clamp(0, 17))
        return self._actor(features, None)[:, [2, 1, 3, 0]]

    def transfer_encoder(self, source):
        """Copy only the encoder, leaving the identically seeded fresh actor alone."""
        old = {k.removeprefix('_encoder.'): v for k, v in source.items()
               if k.startswith('_encoder.')}
        new = self._encoder.state_dict()
        for name, value in old.items():
            if new[name].shape == value.shape:
                new[name].copy_(value)
            elif name.startswith('_depthwise_'):
                new[name][:256].copy_(value)
                for start in (256, 272):
                    new[name][start:start+16].copy_(value[240:256])
                # Old depthwise biases are zero in the source, but learned biases
                # could be nonzero: absent new channels must contribute zero.
                if name.endswith('.bias'):
                    new[name][256:].zero_()
            elif name.startswith('_pointwise_') and name.endswith('.weight'):
                new[name][:, :256].copy_(value)
                for start in (256, 272):
                    new[name][:, start:start+16].copy_(value[:, 240:256])
            else:
                raise ValueError(f'Unexpected encoder parameter: {name}')
        self._encoder.load_state_dict(new)
