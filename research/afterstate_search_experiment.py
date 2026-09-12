"""Complete, resumable fixed-seed games with an immutable neural afterstate leaf."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from rl2048.agents.afterstate_mlp import AfterstateMLPAgent
from rl2048.agents.afterstate_planning import AfterstateLookahead
from rl2048.game import Game2048, legal_actions


def write(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    config_file = args.out/'config.json'
    depth=getattr(args,'depth',2)
    device=getattr(args,'device','cpu')
    nonnegative_leaf=getattr(args,'nonnegative_leaf',False)
    description=('two exact moves and all intervening2/4spawn outcomes, then frozen afterstate value'
                 if depth==2 else f'{depth} exact moves and {depth-1} intervening full spawn layers, then frozen afterstate value')
    config = dict(checkpoint=str(args.checkpoint.resolve()), games=args.games,depth=depth,device=device,
        nonnegative_leaf=nonnegative_leaf,
        seed_start=args.seed_start, spawn_batch=args.spawn_batch,
        policy=description,
        gamma=json.loads((args.checkpoint/'metadata.json').read_text())['gamma'],
        checkpoint_sha256=hashlib.sha256((args.checkpoint/'policy.pt').read_bytes()).hexdigest())
    if config_file.exists():
        old=json.loads(config_file.read_text());old.setdefault('depth',2);old.setdefault('device','cpu')
        old.setdefault('nonnegative_leaf',False)
        if not args.resume or old != config:
            raise ValueError('Existing experiment requires --resume with unchanged checkpoint, seeds and policy')
    else:
        shutil.copytree(args.checkpoint, args.out/'frozen')
        write(config_file, config)
    # Always load our immutable copy; ongoing training cannot change this run.
    if hashlib.sha256((args.out/'frozen/policy.pt').read_bytes()).hexdigest() != config['checkpoint_sha256']:
        raise ValueError('Frozen weights changed')
    torch.set_num_threads(1)
    actor = AfterstateLookahead(AfterstateMLPAgent.load(args.out/'frozen',device), args.spawn_batch,depth,
                               nonnegative_leaf=nonnegative_leaf)
    end = time.monotonic()+args.seconds
    result_file = args.out/'evaluation.json'
    previous = json.loads(result_file.read_text())['episodes'] if args.resume and result_file.exists() else []
    episodes = []
    for seed in range(args.seed_start, args.seed_start+args.games):
        finished = [e for e in previous if e['seed']==seed and e['complete']]
        if finished:
            episodes.append(finished[0]); continue
        env=Game2048(); board,info=env.reset(seed=seed)
        terminal=False; carry=0.; start=time.monotonic(); next_log=start+10
        state_file=args.out/f'state_seed{seed}.json'
        if args.resume and state_file.exists():
            saved=json.loads(state_file.read_text())
            if saved['checkpoint_sha256'] != config['checkpoint_sha256']:
                raise ValueError('Saved game does not match checkpoint')
            env.board=np.array(saved['board'],np.int64)
            env.score,env.steps=saved['score'],saved['length']
            env.np_random.bit_generator.state=saved['rng_state']
            board=env.board.copy(); info=env._info(legal_actions(board))
            terminal=not info['action_mask'].any();env._terminated=terminal
            carry=saved['seconds']
        def save_state():
            write(state_file,dict(seed=seed,board=board.tolist(),score=env.score,length=env.steps,
                seconds=carry+time.monotonic()-start,rng_state=env.np_random.bit_generator.state,
                checkpoint_sha256=config['checkpoint_sha256']))
        error=None
        while not terminal and env.steps<40000:
            if args.stop_file.exists(): error='user stop requested';break
            if time.monotonic()>=end: error='wall-clock experiment budget reached';break
            action=actor.act(board,info['action_mask'])
            board,_,terminal,_,info=env.step(action)
            if time.monotonic()>=next_log:
                save_state(); write(args.out/'progress.json',dict(seed=seed,score=env.score,length=env.steps))
                next_log=time.monotonic()+10
        save_state()
        episodes.append(dict(seed=seed,score=env.score,length=env.steps,max_tile=int(board.max()),
                             complete=bool(terminal),truncated=not terminal and env.steps>=40000,
                             error=error,seconds=carry+time.monotonic()-start))
        complete=len(episodes)==args.games and all(e['complete'] for e in episodes)
        result=dict(complete=complete,requested_games=args.games,episodes=episodes,depth=depth,device=device,
            nonnegative_leaf=nonnegative_leaf,
            mean_score=float(np.mean([e['score'] for e in episodes])) if complete else None,
            action_selection=config['policy'],gamma=config['gamma'])
        write(result_file,result);print('GAME',json.dumps(episodes[-1]),flush=True)
        if error: break


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--games',type=int,default=8)
    parser.add_argument('--seed-start',type=int,default=8940000)
    parser.add_argument('--seconds',type=float,default=900)
    parser.add_argument('--spawn-batch',type=int,default=256)
    parser.add_argument('--depth',type=int,choices=[1,2,3],default=2)
    parser.add_argument('--device',choices=['cpu','mps'],default='cpu')
    parser.add_argument('--nonnegative-leaf',action='store_true',
                        help='Floor learned future points at zero; preserve known merge rewards')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-file',type=Path,required=True)
    run(parser.parse_args())
