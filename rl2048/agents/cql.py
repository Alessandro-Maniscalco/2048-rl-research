"""Discrete CQL(H), with an exact log-sum-exp over legal actions.

Kumar et al. (2020), https://arxiv.org/abs/2006.04779, equation 4.
Uses a Double-DQN Bellman target plus the conservative penalty. The learned
Q-network itself is the greedy policy; there is no separate actor.
"""
from copy import deepcopy
import torch
from rl2048.agents.neural import BoardNet, chosen, optimize, soft_update


def conservative_penalty(q,actions,masks):
    return (q.masked_fill(~masks,-1e9).logsumexp(dim=1)-chosen(q,actions)).mean()


class CQL:
    def __init__(self,device='cpu',width=256,lr=3e-4,gamma=.99,cql_alpha=1.,architecture="mlp",**_):
        self.policy=BoardNet(4,width,architecture).to(device)
        self.target=deepcopy(self.policy).requires_grad_(False)
        self.optimizer=torch.optim.Adam(self.policy.parameters(),lr=lr)
        self.gamma,self.alpha=gamma,cql_alpha

    def update(self,batch):
        with torch.no_grad():
            next_action=self.policy(batch['next_states']).masked_fill(~batch['next_masks'],-1e9).argmax(1)
            next_q=chosen(self.target(batch['next_states']),next_action)
            target=batch['rewards']/128+self.gamma*(~batch['terminated'])*next_q
        q=self.policy(batch['states'])
        bellman_loss=(chosen(q,batch['actions'])-target).square().mean()
        penalty=conservative_penalty(q,batch['actions'],batch['masks'])
        loss=bellman_loss+self.alpha*penalty
        optimize(self.optimizer,loss,self.policy.parameters())
        soft_update(self.target,self.policy)
        return {'bellman_loss':bellman_loss.detach(),'conservative_penalty':penalty.detach()}
