"""Bounded throughput experiment for CPU search while GPU training continues."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch

from rl2048.agents.deep_q_planning import DeepQPlanner
from rl2048.agents.neural import NeuralAgent
from rl2048.agents.ntuple import encode


def initialize(checkpoint, boards, barrier):
    global planner, bank
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    bank = boards
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / 'metadata.json').read_text())
    agent = NeuralAgent.load(checkpoint / 'network', 'cpu')
    planner = DeepQPlanner(agent, gamma=config['gamma'], reward_mode=config['reward_mode'],
        shaping_scale=config['shaping_scale'], depth=3, probability_cutoff=.01,
        network_batch=1024, dedup='hash')
    planner.planned_values(bank[0][None], depth=1)  # Warm common compiled kernels.
    barrier.wait(timeout=90)


def decision(index):
    values = planner.planned_values(bank[index][None])[0]
    return [float(v) if np.isfinite(v) else None for v in values]


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    frames = json.loads(args.replay.read_text())['frames']
    ids = [100, 500, 1000, len(frames)-5]
    boards = [encode(np.array(frames[i]['board'])) for i in ids]
    tasks = list(range(4)) * 8  # Same 32 decisions for every worker count.
    context = mp.get_context('spawn')
    records = []
    reference = None
    for count in getattr(args, 'workers', (1, 2, 4)):
        if args.stop_file.exists():
            break
        barrier = context.Barrier(count+1)
        with ProcessPoolExecutor(max_workers=count, mp_context=context,
                initializer=initialize, initargs=(str(args.checkpoint),boards,barrier)) as executor:
            # Submitting work starts every child, whose initializer then waits.
            futures = [executor.submit(decision, index) for index in tasks]
            start = time.perf_counter()
            barrier.wait(timeout=90)
            initialized = time.perf_counter()
            values = [f.result(timeout=90) for f in futures]
            elapsed = time.perf_counter()-initialized
        if reference is None:
            reference = values
        assert values == reference, 'Parallel scheduling changed a fixed-board result'
        record = dict(workers=count, torch_threads_per_worker=1, decisions=len(tasks),
            seconds=elapsed, decisions_per_second=len(tasks)/elapsed,
            initialization_seconds=initialized-start, outputs_identical=True)
        records.append(record)
        (args.out / 'benchmark.json').write_text(json.dumps(records,indent=2))
        print(json.dumps(record),flush=True)
    (args.out / 'protocol.json').write_text(json.dumps(dict(
        checkpoint=str(args.checkpoint.resolve()), replay=str(args.replay.resolve()),
        board_indices=ids, repeats=8, depth=3, probability_cutoff=.01,
        workers=getattr(args, 'workers', (1, 2, 4)),
        interpretation='Matched fixed-board throughput with concurrent GPU training and existing CPU searches. Initialization excluded. This is not a score experiment or a universal optimal worker count.'),indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--replay',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stop-file',type=Path,required=True)
    parser.add_argument('--workers',type=int,nargs='+',choices=[1,2,4,6,8],default=[1,2,4])
    run(parser.parse_args())
