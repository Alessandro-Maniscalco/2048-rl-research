"""Offline AWR: regress returns, then imitate actions weighted by advantage.

Peng et al. (2019), https://arxiv.org/abs/1910.00177, equations 15-16.
This version uses complete Monte Carlo returns (a supported simpler variant),
not the paper's default TD(lambda) return estimator.
"""
import torch
from rl2048.agents.neural import BoardNet, chosen, masked_log_probs, optimize


class AWR:
    def __init__(self, device='cpu', width=256, lr=3e-4, beta=10., architecture="mlp",**_):
        self.policy=BoardNet(4,width,architecture).to(device)
        self.value=BoardNet(1,width,architecture).to(device)
        self.policy_opt=torch.optim.Adam(self.policy.parameters(),lr=lr)
        self.value_opt=torch.optim.Adam(self.value.parameters(),lr=lr)
        self.beta=beta

    def update(self,batch):
        states,actions,returns=batch['states'],batch['actions'],batch['returns']
        values=self.value(states).squeeze(1)
        value_loss=(values-returns).square().mean()
        optimize(self.value_opt,value_loss,self.value.parameters())
        with torch.no_grad():
            advantage=returns-self.value(states).squeeze(1)
            weights=(advantage/self.beta).clamp(max=3.).exp() # cap ~20
        logs=masked_log_probs(self.policy(states),batch['masks'])
        policy_loss=-(weights*chosen(logs,actions)).mean()
        optimize(self.policy_opt,policy_loss,self.policy.parameters())
        return {'value_loss':value_loss.detach(),'policy_loss':policy_loss.detach(),'weight_mean':weights.mean()}
