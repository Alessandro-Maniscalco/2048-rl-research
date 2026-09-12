"""Matched frozen-Q lookahead screen. Each result is a complete seeded game.

Example: uv run python research/deeper_q_experiment.py --depths 2 3 --games 8
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from rl2048.agents.deep_q_planning import DeepQPlanner, SearchBudgetExceeded
from rl2048.agents.neural import NeuralAgent
from rl2048.agents.ntuple import encode
from rl2048.game import Game2048, legal_actions


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    config_file = args.out / 'config.json'
    if config_file.exists():
        if not args.resume:
            raise FileExistsError('Use a new output directory, or --resume to continue this experiment.')
        (args.out / f'config_before_resume_{time.time_ns()}.json').write_text(config_file.read_text())
    torch.set_num_threads(getattr(args, 'torch_threads', 2))
    original = NeuralAgent.load(args.checkpoint / 'network', args.device)
    settings = json.loads((args.checkpoint / 'metadata.json').read_text())
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config['checkpoint_sha256'] = hashlib.sha256((args.checkpoint / 'network/policy.pt').read_bytes()).hexdigest()
    config['protocol'] = 'Frozen Q leaves and matched game seeds. Exact search when cutoff=0; otherwise low-probability branches stop at the Q estimate.'
    config_file.write_text(json.dumps(config, indent=2))
    end = time.time() + args.seconds
    for depth in args.depths:
        if getattr(args, 'stop_file', None) and args.stop_file.exists():
            break
        planner = DeepQPlanner(original, settings['gamma'], settings['reward_mode'], settings['shaping_scale'],
                               depth=depth, max_edges=args.max_edges, dedup=args.dedup, probability_cutoff=args.probability_cutoff,
                               network_batch=getattr(args, 'network_batch', 8192))
        rows = []
        previous_file = args.out / f'depth{depth}.json'
        previous = json.loads(previous_file.read_text())['episodes'] if args.resume and previous_file.exists() else []
        for seed in range(args.seed_start, args.seed_start + args.games):
            finished = [r for r in previous if r['seed']==seed and r['complete']]
            if finished:
                rows.append(finished[0]); continue
            env = Game2048()
            board, info = env.reset(seed=seed)
            start = time.time()
            steps = 0
            term = False
            carry_seconds = 0.
            state_file = args.out / f'state_depth{depth}_seed{seed}.json'
            if args.resume and state_file.exists():
                saved = json.loads(state_file.read_text())
                if saved['checkpoint_sha256'] != config['checkpoint_sha256']:
                    raise ValueError('Resume checkpoint does not match the frozen Q network.')
                env.board = np.array(saved['board'], dtype=np.int64)
                env.score, env.steps = saved['score'], saved['steps']
                env.np_random.bit_generator.state = saved['rng_state']
                board, info = env.board.copy(), env._info(legal_actions(env.board))
                steps = env.steps
                term = not bool(info['action_mask'].any())
                env._terminated = term
                carry_seconds = saved['seconds']
            def save_state():
                state = dict(board=board.tolist(), score=info['score'], steps=steps,
                             rng_state=env.np_random.bit_generator.state, seconds=carry_seconds+time.time()-start,
                             checkpoint_sha256=config['checkpoint_sha256'])
                temporary = state_file.with_suffix('.tmp')
                temporary.write_text(json.dumps(state)); temporary.replace(state_file)
            error = None
            next_log = start + 10
            while not term and steps < 40000:
                if getattr(args, 'stop_file', None) and args.stop_file.exists():
                    error = 'user stop requested'; break
                if time.time() >= end:
                    error = 'wall-clock experiment budget reached'; break
                try:
                    q = planner.planned_values(encode(board)[None])[0]
                except SearchBudgetExceeded as exc:
                    error = str(exc); break
                board, reward, term, _, info = env.step(int(q.argmax()))
                steps += 1
                if time.time() >= next_log:
                    save_state()
                    progress = dict(depth=depth, seed=seed, steps=steps, score=info['score'], seconds=time.time()-start)
                    (args.out / 'progress.json').write_text(json.dumps(progress, indent=2))
                    print('PROGRESS', progress, flush=True)
                    next_log = time.time() + 10
            row = dict(seed=seed, score=info['score'], length=steps, max_tile=info['max_tile'],
                       complete=term, truncated=steps >= 40000 and not term,
                       seconds=carry_seconds+time.time()-start, error=error)
            save_state()
            rows.append(row)
            result = dict(depth=depth, episodes=rows, requested_games=args.games,
                          complete=len(rows)==args.games and all(r['complete'] for r in rows))
            result['mean_score'] = float(np.mean([r['score'] for r in rows])) if result['complete'] else None
            (args.out / f'depth{depth}.json').write_text(json.dumps(result, indent=2))
            print('GAME', depth, row, flush=True)
            if error: break
        if len(rows) == args.games and all(r['complete'] for r in rows):
            planner.save(args.out / f'planner_depth{depth}', config)
        if time.time() >= end or (getattr(args, 'stop_file', None) and args.stop_file.exists()): break


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('runs/research/deeper_q/screen'))
    p.add_argument('--checkpoint', type=Path, default=Path('runs/research/dqn16_planning_selected'))
    p.add_argument('--depths', type=int, nargs='+', default=[2, 3])
    p.add_argument('--games', type=int, default=8)
    p.add_argument('--seed-start', type=int, default=8600000)
    p.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    p.add_argument('--max-edges', type=int, default=2_000_000)
    p.add_argument('--network-batch', type=int, default=8192)
    p.add_argument('--torch-threads', type=int, choices=[1,2,4], default=2,
                   help='CPU threads per process; independent game shards can use one each')
    p.add_argument('--dedup', choices=['sort','hash'], default='hash')
    p.add_argument('--resume', action='store_true', help='Continue saved game and RNG state; skip already completed games')
    p.add_argument('--probability-cutoff', type=float, default=0.)
    p.add_argument('--seconds', type=float, default=600)
    p.add_argument('--stop-file', type=Path, help='Save progress and stop between decisions when this file exists')
    run(p.parse_args())
