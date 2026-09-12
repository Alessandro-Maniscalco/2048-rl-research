"""Quantile-regression Double DQN with a Transformer or whole-board MLP.

Dabney et al.: https://arxiv.org/abs/1710.10044
Our adaptation keeps n-step replay, legal masks and soft targets from the
existing learner. It is QR-DQN with Double-Q selection, not full Rainbow.
"""
from copy import deepcopy
import math
import torch
from torch import nn
from rl2048.agents.transformer_q import TransformerQNetwork
from rl2048.agents.neural import optimize, soft_update


class QuantileTransformer(TransformerQNetwork):
    def __init__(self,width=128,input_encoding='exponents',depth=2,heads=4,num_quantiles=51):
        if not isinstance(num_quantiles,int) or num_quantiles<2:
            raise ValueError('Use at least two quantiles')
        super().__init__(width,input_encoding,depth,heads)
        self.architecture='transformer_qr'
        self.num_quantiles=num_quantiles
        scalar=self.readout
        self.readout=nn.Linear(16*width,4*num_quantiles)
        # Start each distribution at the scalar control's prediction. With the
        # same seed, initial torso AND mean Q-values match the Double DQN control.
        # Different quantile gradients separate the initially equal predictions.
        with torch.no_grad():
            self.readout.weight.copy_(scalar.weight.repeat_interleave(num_quantiles,dim=0))
            self.readout.bias.copy_(scalar.bias.repeat_interleave(num_quantiles))

    def quantile_values(self,boards):
        return super().forward(boards).reshape(-1,4,self.num_quantiles)

    def forward(self,boards):
        """The acting interface stays [batch,4]: expected return per action."""
        return self.quantile_values(boards).mean(dim=-1)

    def forward_with_entropy(self,boards):
        values,entropy=super().forward_with_entropy(boards)
        return values.reshape(-1,4,self.num_quantiles).mean(-1),entropy


def quantile_targets(rewards,terminated,next_online,next_target,next_masks,discounts,double=True):
    """Choose by mean; keep ALL target quantiles for the selected action."""
    values=next_online.mean(-1) if double else next_target.mean(-1)
    actions=values.masked_fill(~next_masks,-torch.inf).argmax(-1)
    continuation=next_target[torch.arange(len(actions),device=actions.device),actions]
    continuation=torch.where(terminated[:,None],torch.zeros_like(continuation),continuation)
    discounts=torch.as_tensor(discounts,device=rewards.device,dtype=rewards.dtype)
    if discounts.ndim:discounts=discounts[:,None]
    return rewards[:,None]+discounts*continuation


def quantile_huber_loss(prediction,target,kappa=1.,reduction='mean'):
    """All B*N*N pairs, averaged over batch and both quantile axes.

    delta[b,i,j] = target[b,j] - prediction[b,i]
    tau_i = (i+0.5)/N. Target and indicator are stop-gradient quantities.
    """
    if not math.isfinite(kappa) or kappa<=0:
        raise ValueError('kappa must be finite and positive')
    delta=target.detach()[:,None,:]-prediction[:,:,None]
    absolute=delta.abs()
    huber=torch.where(absolute<=kappa,.5*delta.square(),kappa*(absolute-.5*kappa))
    taus=(torch.arange(prediction.shape[1],device=prediction.device,dtype=prediction.dtype)+.5)/prediction.shape[1]
    weight=(taus[None,:,None]-(delta.detach()<0).to(prediction.dtype)).abs()
    pair_loss=weight*huber/kappa
    if reduction=='mean':return pair_loss.mean()
    if reduction=='none':return pair_loss.mean(dim=(1,2))
    raise ValueError('Quantile loss reduction must be mean or none')


class QRDQN:
    def __init__(self,device='cpu',width=128,input_encoding='exponents',depth=2,heads=4,
                 num_quantiles=51,kappa=1.,lr=1e-4,gamma=.99,target_tau=.005,
                 weight_decay=.01,double=True,architecture='transformer_q',
                 attention_entropy=0.,dueling=False,embedding_dim=16,residual=False,**_):
        if architecture not in ('transformer_q','mlp_q') or attention_entropy or dueling:
            raise ValueError('QR-DQN supports plain Transformer and MLP torsos')
        if not math.isfinite(kappa) or kappa<=0:raise ValueError('Invalid kappa')
        if architecture=='transformer_q':
            self.policy=QuantileTransformer(width,input_encoding,depth,heads,num_quantiles).to(device)
        else:
            from rl2048.agents.qr_mlp import QuantileMLP
            self.policy=QuantileMLP(width,input_encoding,depth,embedding_dim,residual,num_quantiles).to(device)
        self.target=deepcopy(self.policy).requires_grad_(False)
        self.optimizer=torch.optim.AdamW(self.policy.parameters(),lr=lr,weight_decay=weight_decay)
        self.gamma,self.target_tau,self.double,self.kappa=gamma,target_tau,double,kappa
        self.entropy_control=None

    def update(self,batch):
        with torch.no_grad():
            target=quantile_targets(batch['rewards'],batch['terminated'],
                self.policy.quantile_values(batch['next_states']),
                self.target.quantile_values(batch['next_states']),batch['next_masks'],
                batch.get('discounts',self.gamma),self.double)
        all_values=self.policy.quantile_values(batch['states'])
        prediction=all_values[torch.arange(len(all_values),device=all_values.device),batch['actions'].long()]
        if 'weights' in batch:
            per_record=quantile_huber_loss(prediction,target,self.kappa,reduction='none')
            loss=(batch['weights']*per_record).mean()
        else:
            loss=quantile_huber_loss(prediction,target,self.kappa)
        optimize(self.optimizer,loss,self.policy.parameters())
        soft_update(self.target,self.policy,self.target_tau)
        errors=(target.mean(-1)-prediction.detach().mean(-1)).abs()
        metrics=dict(q_loss=loss.detach(),mean_q=prediction.detach().mean(),
            mean_abs_td_error=errors.mean(),
            quantile_spread=prediction.detach().std(dim=-1,unbiased=False).mean())
        if 'weights' in batch:
            metrics['_priorities']=errors.detach()
            metrics['importance_weight_mean']=batch['weights'].mean()
        return metrics
