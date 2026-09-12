"""Matched afterstate value learning and a shared exact depth-1/2 planner.

Values are future raw merge reward / 128 AFTER the current slide. Training
samples the spawn, then bootstraps from the best legal next afterstate. Both
learners see exactly the same transitions. Symmetries are exact D4 sharing.
"""
from copy import deepcopy
import json
from pathlib import Path
import time
import numpy as np
from numba import njit
import torch
from torch import nn
from rl2048.agents.mlp_q import MLPQNetwork
from rl2048.agents.ntuple import make_patterns, row_tables, encode
from rl2048.fast2048 import slide, value, update_value


@njit(cache=True)
def moves(boards, rows, rewards):
    after=np.empty((len(boards),4,16),np.uint8)
    gains=np.empty((len(boards),4),np.float32)
    legal=np.empty((len(boards),4),np.bool_)
    for i in range(len(boards)):
        for a in range(4):
            after[i,a],gains[i,a],legal[i,a]=slide(boards[i],a,rows,rewards)
    return after,gains,legal


@njit(cache=True)
def tuple_values(boards, weights, patterns):
    out=np.empty(len(boards),np.float32)
    for i in range(len(boards)):out[i]=value(boards[i],weights,patterns)
    return out


@njit(cache=True)
def tuple_fit(boards, targets, weights, patterns, alpha):
    dummy=np.empty((1,1),np.float32)
    error=0.
    for i in range(len(boards)):
        e=update_value(boards[i],targets[i],weights,patterns,alpha,dummy,dummy,False,True)
        error+=e*e
    return error/len(boards)


class SymmetricValue(nn.Module):
    def __init__(self,width=512,encoding='relational'):
        super().__init__()
        self.width,self.encoding=width,encoding
        self.base=MLPQNetwork(width,encoding,2)
        self.base.layers[-1]=nn.Linear(width,1)
        nn.init.zeros_(self.base.layers[-1].weight)
        nn.init.zeros_(self.base.layers[-1].bias)
        cells=np.arange(16).reshape(4,4)
        permutations=np.array([np.rot90(cells,k).ravel() for k in range(4)]+
                              [np.rot90(np.fliplr(cells),k).ravel() for k in range(4)])
        self.register_buffer('symmetries',torch.tensor(permutations))

    def forward(self,boards):
        # Same rank cap as the n-tuple representation; game dynamics stay exact.
        b=boards.reshape(-1,16).clamp(0,15)[:,self.symmetries].reshape(-1,16)
        return self.base(b).reshape(-1,8).mean(1)


class NeuralValue:
    def __init__(self,width=512,encoding='relational',device='mps',lr=1e-4):
        self.model=SymmetricValue(width,encoding).to(device)
        self.target=deepcopy(self.model)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=lr)
        self.device=device

    @torch.no_grad()
    def values(self,boards,target=False):
        model=self.target if target else self.model
        output=[]
        for start in range(0,len(boards),2048):
            x=torch.as_tensor(boards[start:start+2048],device=self.device,dtype=torch.long)
            output.append(model(x).cpu().numpy())
        return np.concatenate(output) if output else np.empty(0,np.float32)

    def fit(self,boards,targets):
        x=torch.as_tensor(boards,device=self.device,dtype=torch.long)
        y=torch.as_tensor(targets,device=self.device,dtype=torch.float32)
        loss=((self.model(x)-y)**2).mean()
        self.optimizer.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(),10.)
        self.optimizer.step()
        return float(loss.detach().cpu())

    def sync(self):self.target.load_state_dict(self.model.state_dict())

    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        torch.save({k:v.cpu() for k,v in self.model.state_dict().items()},path/'value.pt')
        (path/'config.json').write_text(json.dumps(dict(width=self.model.width,encoding=self.model.encoding)))

    @classmethod
    def load(cls,path,device='cpu'):
        path=Path(path);obj=cls(**json.loads((path/'config.json').read_text()),device=device)
        obj.model.load_state_dict(torch.load(path/'value.pt',map_location='cpu',weights_only=True));obj.sync()
        return obj


class TupleValue:
    def __init__(self,layout='4x6',alpha=.03):
        self.patterns=make_patterns(layout);self.layout=layout;self.alpha=alpha
        self.weights=np.zeros((len(self.patterns),16**self.patterns.shape[-1]),np.float32)
        self.target=self.weights.copy()

    def values(self,boards,target=False):
        return tuple_values(boards,self.target if target else self.weights,self.patterns)

    def fit(self,boards,targets):return float(tuple_fit(boards,targets,self.weights,self.patterns,self.alpha))
    def sync(self):np.copyto(self.target,self.weights)
    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        np.save(path/'weights.npy',self.weights)
        (path/'config.json').write_text(json.dumps(dict(layout=self.layout,alpha=self.alpha)))

    @classmethod
    def load(cls,path):
        path=Path(path);obj=cls(**json.loads((path/'config.json').read_text()))
        obj.weights=np.load(path/'weights.npy');obj.sync();return obj


def greedy_values(boards,learner,target=False):
    after,reward,legal=moves(boards,*row_tables())
    v=np.maximum(learner.values(after.reshape(-1,16),target),0).reshape(-1,4)
    q=np.where(legal,reward/128+v,-np.inf)
    return q,after,legal


def td_targets(next_states,learner):
    q,_,legal=greedy_values(next_states,learner,target=True)
    # A terminal next state has no legal moves. Time limits do not zero targets.
    return np.where(legal.any(1),q.max(1),0).astype(np.float32)


@njit(cache=True)
def depth2_leaves(boards,rows,rewards):
    # Each root action -> at most32 spawn outcomes -> four next actions.
    count=len(boards);leaves=np.empty((count*4*32*4,16),np.uint8)
    group=np.empty(count*4*32*4,np.int64);gains=np.empty(count*4*32*4,np.float32)
    probabilities=np.zeros(count*4*32,np.float64)
    root_gain=np.zeros((count,4),np.float64);root_legal=np.zeros((count,4),np.bool_)
    used=0
    for i in range(count):
        for a in range(4):
            after,gain,changed=slide(boards[i],a,rows,rewards)
            root_gain[i,a]=gain/128;root_legal[i,a]=changed
            if not changed:continue
            empties=np.where(after==0)[0];outcome=0
            for cell in empties:
                for rank in (1,2):
                    g=(i*4+a)*32+outcome;outcome+=1
                    probabilities[g]=(.9 if rank==1 else .1)/len(empties)
                    spawned=after.copy();spawned[cell]=rank
                    for next_action in range(4):
                        nxt,rew,valid=slide(spawned,next_action,rows,rewards)
                        if valid:
                            leaves[used]=nxt;group[used]=g;gains[used]=rew/128;used+=1
    return leaves[:used],group[:used],gains[:used],probabilities,root_gain,root_legal


def plan(boards,learner,depth=1):
    if depth==1:return greedy_values(boards,learner)[0]
    if depth!=2:raise ValueError('Shared planner supports exact depth1 or2, without downgrading/cutoffs.')
    leaves,group,gains,probabilities,reward,legal=depth2_leaves(boards,*row_tables())
    best=np.zeros(len(probabilities),np.float64) # terminal spawned boards have future value zero
    np.maximum.at(best,group,gains+np.maximum(learner.values(leaves),0))
    expected=(best*probabilities).reshape(len(boards),4,32).sum(2)
    return np.where(legal,reward+expected,-np.inf)


class AfterstateAgent:
    name='afterstate_value'
    def __init__(self,learner,depth=1,label='Afterstate value'):
        self.learner,self.depth=learner,depth;self.display_name=label
        self.rng=np.random.default_rng(0)
    def act(self,board,action_mask):
        return int(plan(encode(board)[None],self.learner,self.depth)[0].argmax())


def evaluate_batch(learner,seeds,depth=1,deadline=float('inf'),max_steps=40000,decision=None,decision_with_ids=None):
    """Independent Gym-compatible spawn RNGs; evaluate active games together."""
    if decision is not None and decision_with_ids is not None:
        raise ValueError('Choose one decision callback')
    seeds=list(seeds);rngs=[np.random.default_rng(s) for s in seeds]
    boards=np.zeros((len(seeds),16),np.uint8)
    scores=np.zeros(len(seeds),np.int64);lengths=np.zeros_like(scores)
    done=np.zeros(len(seeds),bool);truncated=np.zeros_like(done)
    def spawn_one(i):
        empty=np.flatnonzero(boards[i]==0)
        cell=empty[rngs[i].integers(len(empty))]
        boards[i,cell]=1 if rngs[i].random()<.9 else 2
    for i in range(len(seeds)):spawn_one(i);spawn_one(i)
    start=time.time()
    while not done.all() and time.time()<deadline:
        ids=np.flatnonzero(~done)
        for offset in range(0,len(ids),16 if depth==2 else 128):
            selected=ids[offset:offset+(16 if depth==2 else 128)]
            if time.time()>=deadline:break
            if decision_with_ids is not None:
                # Stable game indices let stochastic policies use a separate
                # reproducible action RNG per game, independent of batch layout.
                q=decision_with_ids(boards[selected],selected)
            else:
                q=decision(boards[selected]) if decision is not None else plan(boards[selected],learner,depth)
            for j,i in enumerate(selected):
                if not np.isfinite(q[j]).any():done[i]=True;continue
                action=int(q[j].argmax())
                boards[i],reward,changed=slide(boards[i],action,*row_tables())
                assert changed
                scores[i]+=int(reward);lengths[i]+=1;spawn_one(i)
                if lengths[i]>=max_steps:
                    done[i]=True;truncated[i]=bool(moves(boards[i:i+1],*row_tables())[2].any())
    episodes=[dict(seed=seeds[i],score=int(scores[i]),length=int(lengths[i]),
                   max_tile=1<<int(boards[i].max()),complete=bool(done[i]),truncated=bool(truncated[i])) for i in range(len(seeds))]
    finished=[r for r in episodes if r['complete']]
    summary=dict(complete=bool(done.all()),episodes=len(seeds),completed_episodes=len(finished),
        mean_score=float(scores.mean()) if done.all() else None,
        score_std=float(scores.std(ddof=1)) if done.all() and len(seeds)>1 else None,
        mean_length=float(lengths.mean()) if done.all() else None,
        reaching_2048=float(np.mean([r['max_tile']>=2048 for r in episodes])) if done.all() else None,
        max_tile_distribution={str(t):sum(r['max_tile']==t for r in episodes) for t in sorted(set(r['max_tile'] for r in episodes))},
        transitions=int(lengths.sum()),seconds=time.time()-start,depth=depth,truncated_episodes=int(truncated.sum()))
    return dict(summary=summary,episodes=episodes)
