"""Offline training: fixed data -> minibatch -> algorithm.update -> saved policy.

No game is stepped in the training loop. Evaluation happens after training.
Example: python -m rl2048.offline_train --algorithm iql --data runs/research/offline_dataset --out runs/iql --device mps
"""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from rl2048.agents.awr import AWR
from rl2048.agents.iql import IQL
from rl2048.agents.cql import CQL
from rl2048.agents.neural import NeuralAgent, BoardNet, chosen, masked_log_probs, optimize
from rl2048.offline_data import load_dataset
from rl2048.evaluate import evaluate
from rl2048.symmetry import transform_batch


class BehaviorCloning:
    """Control experiment: the same policy network, unweighted imitation."""
    def __init__(self,device='cpu',width=256,lr=3e-4,architecture='mlp',**_):
        self.policy=BoardNet(4,width,architecture).to(device)
        self.optimizer=torch.optim.Adam(self.policy.parameters(),lr=lr)

    def update(self,batch):
        loss=-chosen(masked_log_probs(self.policy(batch['states']),batch['masks']),batch['actions']).mean()
        optimize(self.optimizer,loss,self.policy.parameters())
        return {'policy_loss':loss.detach()}


def tensor_batch(batch,device):
    return {key:torch.as_tensor(value,device=device) for key,value in batch.items()}


def training_state(algorithm):
    """Keep all networks and optimizers, not just the deployed policy."""
    return {key:value.state_dict() for key,value in vars(algorithm).items()
            if isinstance(value,(torch.nn.Module,torch.optim.Optimizer))}


def train(args):
    args.out.mkdir(parents=True,exist_ok=False)
    config={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    config['dataset_manifest']=json.loads((args.data/'manifest.json').read_text())
    (args.out/'config.json').write_text(json.dumps(config,indent=2))
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    data=load_dataset(args.data,args.gamma)
    pool_parts=[];early_parts=[];offset=0
    for episode in config['dataset_manifest']['episodes']:
        length=episode['length']
        if args.data_selection=='all' or episode['policy']=='local_expert':
            pool_parts.append(np.arange(offset,offset+length))
            early_parts.append(np.arange(offset,offset+min(length,128)))
        offset+=length
    pool=np.concatenate(pool_parts);early_pool=np.concatenate(early_parts)
    learner={'awr':AWR,'iql':IQL,'cql':CQL,'bc':BehaviorCloning}[args.algorithm](**vars(args))
    start=time.perf_counter(); updates=0; progress=[]; next_report=0
    while time.perf_counter()-start<args.seconds:
        index=rng.choice(pool,size=args.batch)
        early_count=int(args.batch*args.early_fraction)
        if early_count:
            index[:early_count]=rng.choice(early_pool,size=early_count)
        sample={key:value[index] for key,value in data.items()}
        if args.augment:
            sample=transform_batch(sample,int(rng.integers(4)),bool(rng.integers(2)))
        batch=tensor_batch(sample,args.device)
        losses=learner.update(batch)
        updates+=1
        elapsed=time.perf_counter()-start
        if elapsed>=next_report:
            row={'updates':updates,'training_examples':updates*args.batch,'seconds':elapsed,
                 'environment_transitions_during_training':0,
                 **{key:float(value.cpu()) for key,value in losses.items()}}
            progress.append(row); print(json.dumps(row),flush=True)
            NeuralAgent(learner.policy,args.algorithm,args.device,args.width).save(args.out/'last',config|{'updates':updates})
            with (args.out/'progress.csv').open('w',newline='') as file:
                writer=csv.DictWriter(file,fieldnames=list(row)); writer.writeheader(); writer.writerows(progress)
            next_report=elapsed+15
    NeuralAgent(learner.policy,args.algorithm,args.device,args.width).save(args.out/'last',config|{'updates':updates})
    torch.save(training_state(learner),args.out/'training.pt')
    agent=NeuralAgent(learner.policy.cpu(),args.algorithm,width=args.width)
    summary,episodes=evaluate(agent,seeds=range(6_100_000,6_100_100),max_steps=40_000)
    (args.out/'validation.json').write_text(json.dumps({'summary':summary,'episodes':episodes},indent=2))
    print('VALIDATION',json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--algorithm',choices=['awr','iql','cql','bc'],required=True)
    p.add_argument('--data',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=180); p.add_argument('--device',choices=['cpu','mps'],default='cpu')
    p.add_argument('--seed',type=int,default=0); p.add_argument('--batch',type=int,default=4096)
    p.add_argument('--width',type=int,default=256); p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--gamma',type=float,default=.99); p.add_argument('--tau',type=float,default=.7)
    p.add_argument('--beta',type=float,default=None); p.add_argument('--cql-alpha',type=float,default=1.)
    p.add_argument('--augment',action='store_true',help='Rotate/reflect boards and action labels together')
    p.add_argument('--architecture',choices=['mlp','cnn'],default='mlp')
    p.add_argument('--data-selection',choices=['all','expert'],default='all')
    p.add_argument('--early-fraction',type=float,default=0.,help='Fraction sampled from first 128 moves of selected games')
    args=p.parse_args()
    if args.beta is None:
        args.beta=10. if args.algorithm=='awr' else 3.
    train(args)
