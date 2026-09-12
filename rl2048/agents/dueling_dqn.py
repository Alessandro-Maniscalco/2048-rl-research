"""Same 16 inputs and four Q outputs; separate value and action advantage.

Wang et al. (2016), https://arxiv.org/abs/1511.06581.
Q(s,a) = V(s) + A(s,a) - mean_a A(s,a).
"""
from torch import nn
from rl2048.agents.dqn import QNetwork


class DuelingQNetwork(QNetwork):
    def __init__(self,width=256,input_encoding='exponents',raw_divisor=65536.):
        super().__init__(width,input_encoding,raw_divisor)
        self.architecture='scalar_dueling'
        self.value_head=nn.Linear(width,1)

    def forward(self,boards):
        hidden=self.encode_inputs(boards)
        for layer in self.layers[:-1]:
            hidden=layer(hidden)
        advantages=self.layers[-1](hidden)
        state_value=self.value_head(hidden)
        return state_value+advantages-advantages.mean(dim=1,keepdim=True)
