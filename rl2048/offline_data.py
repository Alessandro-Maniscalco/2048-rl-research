"""Collect a fixed, auditable dataset BEFORE offline learning starts.

Contains full trajectories, original rewards, legal masks, terminal flags,
and discounted Monte Carlo returns. Offline trainers only read these files.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from numba import njit
from rl2048.agents.ntuple import NTupleAgent, row_tables
from rl2048.fast2048 import search, slide, spawn
from rl2048.vector_game import legal_masks


@njit(cache=True, nogil=True)
def collect_game(weights, patterns, rows, gains, seed, random_policy=False, max_steps=40_000):
    np.random.seed(seed)
    board=np.zeros(16,np.uint8); spawn(board); spawn(board)
    states=np.zeros((max_steps,16),np.uint8)
    next_states=np.zeros_like(states)
    actions=np.zeros(max_steps,np.int64)
    rewards=np.zeros(max_steps,np.float32)
    terminated=np.zeros(max_steps,np.bool_)
    count=0
    for t in range(max_steps):
        masks=legal_masks(board.reshape(1,16),rows,gains)[0]
        if not masks.any():
            break
        action=search(board,weights,patterns,rows,gains,1,.0001)
        if random_policy:
            legal=np.where(masks)[0]
            action=legal[np.random.randint(len(legal))]
        states[t]=board
        actions[t]=action
        board,gain,_=slide(board,action,rows,gains); spawn(board)
        next_states[t]=board
        rewards[t]=gain
        terminated[t]=not legal_masks(board.reshape(1,16),rows,gains)[0].any()
        count=t+1
        if terminated[t]:
            break
    return states[:count],actions[:count],rewards[:count],next_states[:count],terminated[:count]


def discounted_returns(rewards, terminated, truncated, gamma):
    returns=np.empty(len(rewards),np.float32)
    carry=0.
    for t in reversed(range(len(rewards))):
        if terminated[t] or truncated[t]:
            carry=0.
        carry=float(rewards[t])/128 + gamma*carry
        returns[t]=carry
    return returns


def collect(args):
    args.out.mkdir(parents=True,exist_ok=False)
    agent=NTupleAgent.load(args.checkpoint,mmap_mode='r')
    rows,gains=row_tables()
    batches=[]; episodes=[]
    for i in range(args.expert_games+args.random_games):
        random_policy=i>=args.expert_games
        sample=collect_game(agent.weights,agent.patterns,rows,gains,args.seed+i,random_policy)
        # Full-return AWR needs complete episodes; don't silently zero-bootstrap
        # a timeout. Exclude capped games from this particular offline dataset.
        if not sample[-1][-1]:
            raise RuntimeError('Dataset game hit time limit; increase collector limit.')
        batches.append(sample)
        episodes.append({'seed':args.seed+i,'policy':'random' if random_policy else 'local_expert',
                         'length':len(sample[1]),'score':float(sample[2].sum())})
        if (i+1)%50==0:
            print('Collected',i+1,'games',flush=True)
    keys=('states','actions','rewards','next_states','terminated')
    data={key:np.concatenate([row[j] for row in batches]) for j,key in enumerate(keys)}
    data['truncated']=np.zeros(len(data['actions']),np.bool_)
    data['masks']=legal_masks(data['states'],rows,gains)
    data['next_masks']=legal_masks(data['next_states'],rows,gains)
    for key,array in data.items():
        np.save(args.out/f'{key}.npy',array)
    digest=hashlib.sha256((args.out/'actions.npy').read_bytes()).hexdigest()
    manifest={'checkpoint':str(args.checkpoint.resolve()),'checkpoint_metadata':agent.metadata,
              'expert_games':args.expert_games,'random_games':args.random_games,'seed':args.seed,
              'transitions':len(data['actions']),'reward_scale':128,'actions_sha256':digest,
              'episodes':episodes,'published_pretrained_data':False}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Dataset saved:',manifest['transitions'],'transitions',flush=True)


def load_dataset(path, gamma=.99):
    path=Path(path)
    data={file.stem:np.load(file,mmap_mode='r') for file in path.glob('*.npy')}
    data['returns']=discounted_returns(data['rewards'],data['terminated'],data['truncated'],gamma)
    return data


class ReplayBuffer:
    """Online SAC's bounded ring buffer; never stores reset states as next states."""
    def __init__(self,capacity=1_000_000):
        self.capacity,self.size,self.position=capacity,0,0
        self.data={key:np.empty((capacity,)+shape,dtype=dtype) for key,shape,dtype in [
            ('states',(16,),np.uint8),('next_states',(16,),np.uint8),('actions',(),np.int64),
            ('rewards',(),np.float32),('terminated',(),np.bool_),('truncated',(),np.bool_),
            ('masks',(4,),np.bool_),('next_masks',(4,),np.bool_)]}

    def add(self,**batch):
        size=len(batch['actions'])
        if size>self.capacity:
            raise ValueError('Insertion batch exceeds replay capacity.')
        index=(np.arange(size)+self.position)%self.capacity
        for key,values in batch.items():
            self.data[key][index]=values
        self.position=(self.position+size)%self.capacity
        self.size=min(self.capacity,self.size+size)

    def sample(self,size,rng):
        indices=rng.integers(self.size,size=size)
        return {key:values[indices] for key,values in self.data.items()}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--expert-games',type=int,default=200); p.add_argument('--random-games',type=int,default=1000)
    p.add_argument('--seed',type=int,default=9_000_000)
    collect(p.parse_args())
