"""IQL: asymmetric value regression + in-dataset Q fitting + weighted BC.

Kostrikov et al. (2021), https://arxiv.org/abs/2110.06169, equations 5-7.
Only actions actually present in the dataset enter the value and Q losses.
"""
from copy import deepcopy
import torch
from rl2048.agents.neural import BoardNet, chosen, masked_log_probs, optimize, soft_update


def expectile_loss(error, tau=.7):
    return (torch.where(error>0,tau,1-tau)*error.square()).mean()


class IQL:
    def __init__(self,device='cpu',width=256,lr=3e-4,gamma=.99,tau=.7,beta=3.,architecture="mlp",**_):
        self.policy=BoardNet(4,width,architecture).to(device)
        self.value=BoardNet(1,width,architecture).to(device)
        self.q1=BoardNet(4,width,architecture).to(device); self.q2=BoardNet(4,width,architecture).to(device)
        self.target1=deepcopy(self.q1).requires_grad_(False)
        self.target2=deepcopy(self.q2).requires_grad_(False)
        self.policy_opt=torch.optim.Adam(self.policy.parameters(),lr=lr)
        self.value_opt=torch.optim.Adam(self.value.parameters(),lr=lr)
        self.q_parameters=list(self.q1.parameters())+list(self.q2.parameters())
        self.q_opt=torch.optim.Adam(self.q_parameters,lr=lr)
        self.gamma,self.tau,self.beta=gamma,tau,beta

    def update(self,batch):
        states,actions=batch['states'],batch['actions']
        with torch.no_grad():
            q_data=chosen(torch.minimum(self.target1(states),self.target2(states)),actions)
        value_loss=expectile_loss(q_data-self.value(states).squeeze(1),self.tau)
        optimize(self.value_opt,value_loss,self.value.parameters())
        with torch.no_grad():
            target=batch['rewards']/128+self.gamma*(~batch['terminated'])*self.value(batch['next_states']).squeeze(1)
        q_loss=(chosen(self.q1(states),actions)-target).square().mean()+(chosen(self.q2(states),actions)-target).square().mean()
        optimize(self.q_opt,q_loss,self.q_parameters)
        with torch.no_grad():
            advantage=q_data-self.value(states).squeeze(1)
            weights=(self.beta*advantage).clamp(max=4.60517).exp() # cap100
        policy_loss=-(weights*chosen(masked_log_probs(self.policy(states),batch['masks']),actions)).mean()
        optimize(self.policy_opt,policy_loss,self.policy.parameters())
        soft_update(self.target1,self.q1); soft_update(self.target2,self.q2)
        return {'value_loss':value_loss.detach(),'q_loss':q_loss.detach(),'policy_loss':policy_loss.detach()}
