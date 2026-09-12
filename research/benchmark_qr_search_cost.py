"""Measure search cost and action changes on fixed early/middle/late boards.

This chooses a feasible game-screen budget, not a winning policy. Low-probability
branches use the Q network sooner; their probability mass is never discarded.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from rl2048.agents.deep_q_planning import DeepQPlanner, SearchBudgetExceeded
from rl2048.agents.neural import NeuralAgent
from rl2048.agents.ntuple import encode
from rl2048.game import ACTION_NAMES


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    agent = NeuralAgent.load(args.checkpoint / 'network', 'cpu')
    settings = json.loads((args.checkpoint / 'metadata.json').read_text())
    frames = json.loads(args.replay.read_text())['frames']
    indices = [100, 500, 1000, len(frames) - 5]
    candidates = [(3, 0.), (3, .001), (4, .001), (4, .01)]
    protocol = dict(checkpoint=str(args.checkpoint.resolve()), replay=str(args.replay.resolve()),
        indices=indices, candidates=candidates, network_batch=1024, max_edges=2000000,
        purpose='Find an affordable approximate deeper-search candidate, motivated by exact three-step taking about three hours for eight games. Fixed-state agreement is diagnostic, not proof of strength.',
        budget_seconds=300, stop_file=str(args.stop_file))
    (args.out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    # Exclude first-use Numba compilation from the per-decision timing.
    DeepQPlanner(agent, depth=1, dedup='hash', network_batch=1024).planned_values(encode(np.array(frames[100]['board']))[None])
    deadline = time.time() + 300
    records = []
    for depth, cutoff in candidates:
        planner = DeepQPlanner(agent, gamma=settings['gamma'], reward_mode=settings['reward_mode'],
            shaping_scale=settings['shaping_scale'], depth=depth, probability_cutoff=cutoff,
            network_batch=1024, dedup='hash', max_edges=2000000)
        for index in indices:
            if args.stop_file.exists() or time.time() >= deadline:
                return
            record = dict(depth=depth, cutoff=cutoff, board_index=index)
            start = time.perf_counter()
            try:
                q = planner.planned_values(encode(np.array(frames[index]['board']))[None])[0]
                record.update(action=ACTION_NAMES[int(q.argmax())],
                    values=[float(v) if np.isfinite(v) else None for v in q], stats=planner.last_stats)
            except SearchBudgetExceeded as exc:
                record['error'] = str(exc)
            record['seconds'] = time.perf_counter() - start
            records.append(record)
            (args.out / 'benchmark.json').write_text(json.dumps(records, indent=2))
            print(json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stop-file', type=Path, required=True)
    run(parser.parse_args())
