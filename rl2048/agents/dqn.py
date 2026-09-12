"""Exactly 16 numeric cell inputs -> 4 Q-values -> largest legal Q.

The network sees the current board only. It never tries a move before choosing.
DQN: Mnih et al., https://www.nature.com/articles/nature14236
Double DQN: van Hasselt et al., https://arxiv.org/abs/1509.06461
"""
from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F
from rl2048.agents.neural import chosen,optimize,soft_update


class QNetwork(nn.Module):
    def __init__(self,width=256,input_encoding='exponents',raw_divisor=65536.):
        super().__init__()
        if input_encoding not in ('exponents','raw','relative'):
            raise ValueError('Unknown scalar input encoding.')
        self.input_encoding=input_encoding
        if not math.isfinite(raw_divisor) or raw_divisor<=0:
            raise ValueError('Raw tile divisor must be finite and positive.')
        self.raw_divisor=float(raw_divisor)
        self.architecture='scalar'
        self.layers=nn.Sequential(nn.Linear(16,width),nn.ReLU(),
                                  nn.Linear(width,width),nn.ReLU(),nn.Linear(width,4))

    def encode_inputs(self,boards):
        # Storage uses exponents: 0=empty, 1=2, 2=4, ... .
        exponents=boards.float()
        if self.input_encoding=='exponents':
            return exponents/16 # all 16 inputs approximately in [0,1]
        tiles=torch.where(boards>0,torch.exp2(exponents),0.)
        if self.input_encoding=='raw':
            return tiles/self.raw_divisor # actual powers of two, with a fixed scale
        return tiles/tiles.amax(dim=1,keepdim=True).clamp_min(1) # largest tile -> 1

    def forward(self,boards):
        return self.layers(self.encode_inputs(boards))


def bellman_target(rewards,terminated,next_q_online,next_q_target,next_masks,gamma,double=True):
    if double:
        # Online network chooses; slower target network evaluates.
        actions=next_q_online.masked_fill(~next_masks,-1e9).argmax(1)
        continuation=chosen(next_q_target,actions)
    else:
        continuation=next_q_target.masked_fill(~next_masks,-1e9).max(1).values
    continuation=torch.where(terminated,0.,continuation)
    return rewards+gamma*continuation


class DQN:
    def __init__(self,device='cpu',width=256,input_encoding='exponents',lr=3e-4,gamma=.99,double=True,dueling=False,raw_divisor=65536.,architecture='scalar',depth=2,embedding_dim=16,residual=False,target_tau=.005,heads=4,attention_entropy=0.,entropy_target=.8,weight_decay=0.,**_):
        network=QNetwork
        if dueling:
            from rl2048.agents.dueling_dqn import DuelingQNetwork
            network=DuelingQNetwork
        if architecture == 'transformer_q':
            if dueling:
                raise ValueError('This Transformer has a plain four-Q readout.')
            from rl2048.agents.transformer_q import TransformerQNetwork
            self.policy=TransformerQNetwork(width,input_encoding,depth,heads).to(device)
        elif architecture == 'mlp_q':
            if dueling:
                raise ValueError('Use scalar architecture for the existing dueling variant.')
            from rl2048.agents.mlp_q import MLPQNetwork
            self.policy=MLPQNetwork(width,input_encoding,depth,embedding_dim,residual).to(device)
        else:
            self.policy=network(width,input_encoding,raw_divisor).to(device)
        self.target=deepcopy(self.policy).requires_grad_(False)
        self.optimizer=(torch.optim.AdamW(self.policy.parameters(),lr=lr,weight_decay=weight_decay)
            if weight_decay else torch.optim.Adam(self.policy.parameters(),lr=lr))
        self.gamma,self.double=gamma,double
        self.target_tau=target_tau
        self.entropy_control=None
        if attention_entropy:
            if architecture!='transformer_q':raise ValueError('Attention control requires a Transformer.')
            from rl2048.agents.attention_entropy import AttentionEntropyControl
            self.entropy_control=AttentionEntropyControl(depth,attention_entropy,entropy_target).to(device)
            self.entropy_optimizer=torch.optim.Adam(self.entropy_control.parameters(),lr=lr)

    def update(self,batch):
        with torch.no_grad():
            target=bellman_target(batch['rewards'],batch['terminated'],self.policy(batch['next_states']),
                                  self.target(batch['next_states']),batch['next_masks'],batch.get('discounts',self.gamma),self.double)
        entropy=None
        if self.entropy_control is not None:
            q,entropy=self.policy.forward_with_entropy(batch['states'])
        else:q=self.policy(batch['states'])
        prediction=chosen(q,batch['actions'])
        # Huber is less sensitive to rare large merges than squared error.
        loss=F.smooth_l1_loss(prediction,target)
        total=loss
        if entropy is not None:
            attention_loss,temperature_loss=self.entropy_control.losses(entropy)
            total=loss+attention_loss
            optimize(self.entropy_optimizer,temperature_loss,self.entropy_control.parameters())
        optimize(self.optimizer,total,self.policy.parameters())
        soft_update(self.target,self.policy,self.target_tau)
        metrics={'q_loss':loss.detach(),'mean_q':prediction.detach().mean(),
                'mean_abs_td_error':(target-prediction).detach().abs().mean()}
        if entropy is not None:
            metrics.update(attention_entropy=entropy.detach().mean(),attention_loss=attention_loss.detach(),
                attention_alpha=self.entropy_control.log_alpha.detach().exp().mean())
        return metrics
