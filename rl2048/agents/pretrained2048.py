"""Import an actual pretrained 2048 CNN; logits are not Q-values.

Upstream board ranks are row-major. Its actions are left/right/up/down;
ours are up/right/down/left. The adapter explicitly reorders the outputs.
"""
import torch
from torch import nn
from .ml2048_network import CNNEncoder,CNNActorNetwork,CNNCriticNetwork


class Pretrained2048(nn.Module):
    architecture='pretrained_ml2048'
    input_encoding='one_hot'
    outputs=5
    width=1024
    def __init__(self):
        super().__init__()
        self._encoder=CNNEncoder(1024)
        self._actor=CNNActorNetwork(1024,256,64)
        self._critic=CNNCriticNetwork(1024,256,64)

    def forward(self,boards):
        features=self._encoder(boards.long().reshape(-1,16).clamp(0,15))
        logits=self._actor(features,None)[:,[2,1,3,0]]
        value=self._critic(features,None)
        return torch.cat((logits,value[:,None]),1)

    def forward_policy(self, boards):
        """Actor-only inference/training; the old shaped-reward critic is unused."""
        features = self._encoder(boards.long().reshape(-1, 16).clamp(0, 15))
        return self._actor(features, None)[:, [2, 1, 3, 0]]

    @classmethod
    def from_external(cls,path,reset_critic=False):
        model=cls()
        saved=torch.load(path,map_location='cpu',weights_only=True)
        model.load_state_dict(saved['policy_state'],strict=True)
        if reset_critic:
            # Original critic predicts a different, strongly shaped reward.
            # Keep pretrained actor/encoder; recalibrate the value output for
            # our raw-score/128 objective instead of treating old values as Q.
            nn.init.zeros_(model._critic._out.weight)
            nn.init.zeros_(model._critic._out.bias)
        return model
