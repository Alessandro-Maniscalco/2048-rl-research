"""Discrete SAC with explicit V, twin Q critics, target V, and actor.

Combines the explicit-V SAC formulation (Haarnoja et al., 2018,
https://arxiv.org/abs/1801.01290) with exact categorical expectations
(Christodoulou, 2019, https://arxiv.org/abs/1910.07207).
Modern discrete SAC can eliminate the separate V network; keeping it here
makes the user's requested value/critic distinction visible. Fixed temperature.
"""
from copy import deepcopy
import torch
from rl2048.agents.neural import BoardNet, chosen, masked_log_probs, optimize, soft_update


def soft_value(log_probs,q_values,temperature):
    return (log_probs.exp()*(q_values-temperature*log_probs)).sum(-1)


class SAC:
    def __init__(self,device='cpu',width=256,lr=3e-4,gamma=.99,temperature=.05,architecture="mlp",**_):
        self.policy=BoardNet(4,width,architecture).to(device)
        self.value=BoardNet(1,width,architecture).to(device)
        self.target_value=deepcopy(self.value).requires_grad_(False)
        self.q1=BoardNet(4,width,architecture).to(device); self.q2=BoardNet(4,width,architecture).to(device)
        self.policy_opt=torch.optim.Adam(self.policy.parameters(),lr=lr)
        self.value_opt=torch.optim.Adam(self.value.parameters(),lr=lr)
        self.q_parameters=list(self.q1.parameters())+list(self.q2.parameters())
        self.q_opt=torch.optim.Adam(self.q_parameters,lr=lr)
        self.gamma,self.temperature=gamma,temperature

    def update(self,batch):
        states,actions=batch['states'],batch['actions']
        with torch.no_grad():
            target=batch['rewards']/128+self.gamma*(~batch['terminated'])*self.target_value(batch['next_states']).squeeze(1)
        q_loss=(chosen(self.q1(states),actions)-target).square().mean()+(chosen(self.q2(states),actions)-target).square().mean()
        optimize(self.q_opt,q_loss,self.q_parameters)
        with torch.no_grad():
            min_q=torch.minimum(self.q1(states),self.q2(states))
            logs=masked_log_probs(self.policy(states),batch['masks'])
            value_target=soft_value(logs,min_q,self.temperature)
        value_loss=(self.value(states).squeeze(1)-value_target).square().mean()
        optimize(self.value_opt,value_loss,self.value.parameters())
        logs=masked_log_probs(self.policy(states),batch['masks'])
        policy_loss=-soft_value(logs,min_q,self.temperature).mean()
        optimize(self.policy_opt,policy_loss,self.policy.parameters())
        soft_update(self.target_value,self.value)
        return {'q_loss':q_loss.detach(),'value_loss':value_loss.detach(),'policy_loss':policy_loss.detach()}
