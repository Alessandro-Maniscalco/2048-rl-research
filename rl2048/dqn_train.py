"""Train the requested 16-input, 4-Q-value network, with visible replay updates.

Inference: current board -> QNetwork -> mask illegal moves -> argmax.
Training adds epsilon-greedy exploration, a replay buffer, and a target network.
"""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from rl2048.agents.dqn import DQN
from rl2048.agents.neural import NeuralAgent,tensor_boards
from rl2048.vector_game import VectorGame,legal_masks
from rl2048.offline_data import ReplayBuffer
from rl2048.offline_train import tensor_batch,training_state
from rl2048.rewards import learning_rewards
from rl2048.evaluate import evaluate


def train(args):
    args.out.mkdir(parents=True,exist_ok=False)
    config={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    (args.out/'config.json').write_text(json.dumps(config,indent=2))
    torch.set_num_threads(args.cpu_threads);torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    learner=DQN(**vars(args))
    prior_steps=0
    if args.resume:
        old_config=json.loads((args.resume/'config.json').read_text())
        for key in ('input_encoding','width','dueling'):
            if old_config.get(key,False if key=='dueling' else None)!=getattr(args,key):
                raise ValueError(f'Resume requires the same {key}')
        if old_config.get('raw_divisor',65536.)!=args.raw_divisor:
            raise ValueError('Resume requires the same raw tile divisor.')
        for key,default in [('architecture','scalar'),('depth',2),('embedding_dim',16),('residual',False)]:
            if old_config.get(key,default)!=getattr(args,key):
                raise ValueError(f'Resume requires the same {key}')
        saved=torch.load(args.resume/'training.pt',map_location='cpu',weights_only=True)
        for key,state in saved.items():
            getattr(learner,key).load_state_dict(state)
        for group in learner.optimizer.param_groups:
            group['lr']=args.lr
        metadata=json.loads((args.resume/'last/metadata.json').read_text())
        prior_steps=metadata['experiment'].get('total_transitions',metadata['experiment'].get('transitions',0))
    name=('dueling_' if args.dueling else '')+('double_dqn' if args.double else 'dqn')
    env=VectorGame(args.envs,args.seed)
    env.step(np.array([np.flatnonzero(mask)[0] for mask in env.masks()]))
    env=VectorGame(args.envs,args.seed)
    buffer=ReplayBuffer(args.capacity)
    # Buffer rewards here are ALREADY scaled/shaped learning rewards. Raw game
    # scores are tracked independently by the environment and episode records.
    start=time.perf_counter();steps=0;updates=0;next_report=0;next_eval=500_000
    progress=[];episodes=[];validation_curve=[];best_score=-np.inf
    while steps<args.steps and time.perf_counter()-start<args.seconds:
        states,masks=env.boards.copy(),env.masks()
        epsilon=1-(1-args.epsilon_end)*min(1.,(steps+prior_steps)/args.epsilon_steps)
        with torch.no_grad():
            q_values=learner.policy(tensor_boards(states,args.device)).cpu().numpy()
        actions=np.where(masks,q_values,-np.inf).argmax(1)
        # Uniform among legal actions; random scores give each legal action
        # the same chance to be largest.
        random_actions=np.where(masks,rng.random(masks.shape),-np.inf).argmax(1)
        explore=rng.random(args.envs)<epsilon
        actions=np.where(explore,random_actions,actions)
        raw,next_states,term,trunc,completed=env.step(actions)
        shaped=learning_rewards(raw,states,next_states,term,args.gamma,args.reward_mode,args.shaping_scale)
        buffer.add(states=states,actions=actions,rewards=shaped,next_states=next_states,
                   masks=masks,next_masks=legal_masks(next_states,env.rows,env.row_rewards),
                   terminated=term,truncated=trunc)
        steps+=args.envs
        for score,length,tile in completed[completed[:,1]>0]:
            episodes.append({'transitions':steps,'score':int(score),'length':int(length),'max_tile':int(tile)})
        losses={}
        if steps>=args.warmup:
            for _ in range(args.updates_per_batch):
                losses=learner.update(tensor_batch(buffer.sample(args.batch,rng),args.device))
                updates+=1
        elapsed=time.perf_counter()-start
        if steps>=next_eval:
            agent=NeuralAgent(deepcopy(learner.policy).cpu(),name,width=args.width)
            summary,validation_games=evaluate(agent,seeds=range(6_300_000,6_300_020),max_steps=40_000)
            if summary['mean_score']>best_score:
                best_score=summary['mean_score']
                agent.save(args.out/'best',config|{'transitions':steps,'prior_transitions':prior_steps,
                                                 'total_transitions':steps+prior_steps})
                (args.out/'best_validation.json').write_text(json.dumps({
                    'summary':summary,'episodes':validation_games,'transitions':steps,
                    'selection':'Highest mean on the fixed 20-game validation suite'},indent=2))
            row={'transitions':steps,'updates':updates,'seconds':time.perf_counter()-start,
                 'validation_score':summary['mean_score'],'mean_length':summary['mean_episode_length'],
                 'reaching_2048':summary['tile_reaching_rates']['2048']}
            validation_curve.append(row)
            with (args.out/'validation_curve.csv').open('w',newline='') as file:
                writer=csv.DictWriter(file,fieldnames=list(row));writer.writeheader();writer.writerows(validation_curve)
            print('EVALUATION',json.dumps(row),flush=True)
            next_eval+=500_000
        if elapsed>=next_report and losses:
            recent=episodes[-100:]
            row={'transitions':steps,'updates':updates,'seconds':elapsed,'epsilon':epsilon,
                 'mean_score_100':float(np.mean([r['score'] for r in recent])) if recent else 0.,
                 **{key:float(value.cpu()) for key,value in losses.items()}}
            progress.append(row);print(json.dumps(row),flush=True)
            NeuralAgent(learner.policy,name,args.device,args.width).save(args.out/'last',config|{'transitions':steps,'prior_transitions':prior_steps,'total_transitions':steps+prior_steps})
            for filename,rows in [('progress',progress),('episodes',episodes)]:
                if rows:
                    with (args.out/f'{filename}.csv').open('w',newline='') as file:
                        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            next_report=elapsed+15
    NeuralAgent(learner.policy,name,args.device,args.width).save(args.out/'last',config|{'transitions':steps,'prior_transitions':prior_steps,'total_transitions':steps+prior_steps})
    torch.save(training_state(learner),args.out/'training.pt')
    agent=NeuralAgent(learner.policy.cpu(),name,width=args.width)
    summary,records=evaluate(agent,seeds=range(6_300_000,6_300_100),max_steps=40_000)
    (args.out/'validation.json').write_text(json.dumps({'summary':summary,'episodes':records},indent=2))
    for filename,rows in [('progress',progress),('episodes',episodes)]:
        if rows:
            with (args.out/f'{filename}.csv').open('w',newline='') as file:
                writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('VALIDATION',json.dumps(summary),flush=True)
    return summary


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume',type=Path,help='Resume networks/optimizer from a prior run; refill replay before updating')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seconds',type=float,default=300)
    p.add_argument('--steps',type=int,default=3_000_000);p.add_argument('--device',choices=['cpu','mps'],default='cpu')
    p.add_argument('--seed',type=int,default=0);p.add_argument('--input-encoding',choices=['raw','exponents','relative','one_hot','embedding','relational'],default='exponents')
    p.add_argument('--architecture',choices=['scalar','mlp_q'],default='scalar')
    p.add_argument('--depth',type=int,default=2);p.add_argument('--embedding-dim',type=int,default=16)
    p.add_argument('--residual',action='store_true');p.add_argument('--target-tau',type=float,default=.005)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--raw-divisor',type=float,default=65536.,help='Fixed positive scale for raw tile inputs only')
    p.add_argument('--dueling',action='store_true',help='Separate shared state value from action advantages')
    p.add_argument('--double',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--reward-mode',choices=['score','corner','snake','corner_snake','dense_corner'],default='score')
    p.add_argument('--shaping-scale',type=float,default=2.)
    p.add_argument('--envs',type=int,default=256);p.add_argument('--batch',type=int,default=4096)
    p.add_argument('--width',type=int,default=256);p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--gamma',type=float,default=.99);p.add_argument('--capacity',type=int,default=1_000_000)
    p.add_argument('--warmup',type=int,default=16384);p.add_argument('--updates-per-batch',type=int,default=1)
    p.add_argument('--epsilon-end',type=float,default=.05);p.add_argument('--epsilon-steps',type=int,default=1_000_000)
    return p


if __name__=='__main__':
    train(parser().parse_args())
