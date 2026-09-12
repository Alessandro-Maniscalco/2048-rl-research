"""Online discrete SAC: collect transitions -> replay buffer -> off-policy updates."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from rl2048.agents.sac import SAC
from rl2048.agents.neural import sample_actions, NeuralAgent, masked_log_probs, tensor_boards
from rl2048.vector_game import VectorGame, legal_masks
from rl2048.offline_data import ReplayBuffer
from rl2048.offline_train import tensor_batch, training_state
from rl2048.evaluate import evaluate


def train(args):
    args.out.mkdir(parents=True,exist_ok=False)
    config={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    (args.out/'config.json').write_text(json.dumps(config,indent=2))
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    learner=SAC(**vars(args))
    env=VectorGame(args.envs,args.seed)
    env.step(np.array([np.flatnonzero(m)[0] for m in env.masks()]))
    env=VectorGame(args.envs,args.seed)
    buffer=ReplayBuffer(args.capacity)
    start=time.perf_counter(); steps=0; updates=0; next_report=0; progress=[]; episodes=[]
    while time.perf_counter()-start<args.seconds:
        states,masks=env.boards.copy(),env.masks()
        if steps<args.warmup:
            actions=np.array([rng.choice(np.flatnonzero(mask)) for mask in masks])
        else:
            with torch.no_grad():
                logs=masked_log_probs(learner.policy(tensor_boards(states,args.device)),torch.as_tensor(masks,device=args.device))
                probs=logs.exp().cpu().numpy()
                actions=sample_actions(probs, rng)
        rewards,next_states,term,trunc,completed=env.step(actions)
        buffer.add(states=states,actions=actions,rewards=rewards,next_states=next_states,
                   masks=masks,next_masks=legal_masks(next_states,env.rows,env.row_rewards),terminated=term,truncated=trunc)
        steps+=args.envs
        for score,length,tile in completed[completed[:,1]>0]:
            episodes.append({'transitions':steps,'score':int(score),'length':int(length),'max_tile':int(tile)})
        losses={}
        if steps>=args.warmup:
            for _ in range(args.updates_per_batch):
                losses=learner.update(tensor_batch(buffer.sample(args.batch,rng),args.device))
                updates+=1
        elapsed=time.perf_counter()-start
        if elapsed>=next_report and losses:
            recent=episodes[-100:]
            row={'transitions':steps,'updates':updates,'seconds':elapsed,'buffer_size':buffer.size,
                 'mean_score_100':float(np.mean([r['score'] for r in recent])) if recent else 0.,
                 **{key:float(value.cpu()) for key,value in losses.items()}}
            progress.append(row); print(json.dumps(row),flush=True)
            NeuralAgent(learner.policy,'sac',args.device,args.width).save(args.out/'last',config|{'transitions':steps})
            for name,rows in [('progress',progress),('episodes',episodes)]:
                if rows:
                    with (args.out/f'{name}.csv').open('w',newline='') as file:
                        writer=csv.DictWriter(file,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            next_report=elapsed+15
    NeuralAgent(learner.policy,'sac',args.device,args.width).save(args.out/'last',config|{'transitions':steps})
    torch.save(training_state(learner),args.out/'training.pt')
    np.savez(args.out/'replay.npz',**{key:val[:buffer.size] for key,val in buffer.data.items()})
    agent=NeuralAgent(learner.policy.cpu(),'sac',width=args.width)
    summary,records=evaluate(agent,seeds=range(6_100_000,6_100_100),max_steps=40_000)
    (args.out/'validation.json').write_text(json.dumps({'summary':summary,'episodes':records},indent=2))
    print('VALIDATION',json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--seconds',type=float,default=180)
    p.add_argument('--device',choices=['cpu','mps'],default='cpu'); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--envs',type=int,default=256); p.add_argument('--batch',type=int,default=4096)
    p.add_argument('--width',type=int,default=256); p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--gamma',type=float,default=.99); p.add_argument('--temperature',type=float,default=.05)
    p.add_argument('--capacity',type=int,default=1_000_000); p.add_argument('--warmup',type=int,default=16384)
    p.add_argument('--updates-per-batch',type=int,default=1)
    p.add_argument('--architecture',choices=['mlp','cnn'],default='mlp')
    train(p.parse_args())
