"""TQL-inspired layer-wise adaptive attention entropy control.

Source: https://arxiv.org/html/2602.01439v1, equations 2--4 and appendix C.
This is an online discrete Double-DQN adaptation, not a reproduction of TQL's
offline continuous-action algorithm, VALUE token or flow actor.
"""
import math
import torch
from torch import nn


class AttentionEntropyControl(nn.Module):
    def __init__(self,layers,initial_alpha=.01,target_fraction=.8):
        super().__init__()
        if not 0<initial_alpha or not 0<target_fraction<=1:
            raise ValueError('Positive alpha and target fraction in (0,1] required.')
        self.log_alpha=nn.Parameter(torch.full((layers,),math.log(initial_alpha)))
        target=torch.full((layers,),target_fraction*math.log(16))
        target[-1]=max(0.,float(target[-1])-.5)
        self.register_buffer('target',target)

    def losses(self,entropy):
        alpha=self.log_alpha.exp()
        # Separate gradients: Q weights maximize entropy, temperatures track
        # the target. Neither loss backpropagates into the other's parameters.
        attention_loss=-(alpha.detach()*entropy).mean()
        temperature_loss=(alpha*(entropy.detach()-self.target)).mean()
        return attention_loss,temperature_loss
