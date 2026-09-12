"""Measure this CNN's actual CPU inference and complete-game execution paths."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import torch

from research.afterstate_teacher import digest, write
from rl2048.agents.neural import NeuralAgent, policy_logits, tensor_boards
from rl2048.evaluate import rollout


AGENT = None


def initialize(source):
    global AGENT
    torch.set_num_threads(1)
    AGENT = NeuralAgent.load(source, 'cpu')


def game(seed):
    result, _ = rollout(AGENT, seed=seed, max_steps=100000)
    return result


def main():
    out = Path('runs/research/compute/cnn_cpu_execution')
    out.mkdir(parents=True, exist_ok=False)
    source = Path('runs/research/pretrained_cnn/original').resolve()
    checksum = digest(source/'policy.pt')
    data_path = Path('runs/research/scaled_transformer/cnn_dagger_kl_anchor_seed0/round_001.npz')
    with np.load(data_path) as saved:
        ids = np.linspace(0,len(saved['states'])-1,256,dtype=int)
        boards = saved['states'][ids]
    seeds = list(range(8958450,8958458))
    write(out/'protocol.json', dict(source=str(source),source_sha256=checksum,
        state_source=str(data_path),state_indices=ids.tolist(),complete_game_seeds=seeds,
        reason='The frozen100-game CPU evaluator took681s, whereas a5534move single-game replay took1.9s. '
        'Measure the actual input-batch path instead of assuming larger CPU batches are faster. '
        'Then compare8identical complete games sequentially versus4persistent CPU workers. '
        'The existing GPU learner and CPU search screen remain independent; no claim of isolated hardware peak speed.',
        torch_version=torch.__version__,timing='Forward tests include input conversion and output readback. '
        'Full-game pool timing includes worker startup and checkpoint loading; episode times are recorded separately.'))
    initialize(source)
    records, reference = [], None
    for threads in [1,4]:
        torch.set_num_threads(threads)
        for chunk in [1,8,32,128]:
            durations=[]
            for repeat in range(3):
                start=time.perf_counter()
                with torch.no_grad():
                    output=np.concatenate([policy_logits(AGENT.policy,tensor_boards(boards[i:i+chunk],'cpu')).numpy()
                                           for i in range(0,len(boards),chunk)])
                durations.append(time.perf_counter()-start)
                if reference is None:
                    reference=output.copy()
            record=dict(threads=threads,chunk=chunk,boards=len(boards),
                median_seconds=float(np.median(durations)),
                max_absolute_logit_difference=float(np.max(np.abs(reference-output))),
                raw_argmax_agreement=float((reference.argmax(1)==output.argmax(1)).mean()))
            records.append(record)
            write(out/'forward.json',records)
            print('FORWARD',json.dumps(record),flush=True)
    torch.set_num_threads(1)
    start=time.perf_counter()
    serial=[game(s) for s in seeds]
    serial_seconds=time.perf_counter()-start
    write(out/'serial.json',dict(seconds=serial_seconds,games=serial))
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4,initializer=initialize,initargs=(str(source),)) as pool:
        parallel=list(pool.map(game,seeds))
    parallel_seconds=time.perf_counter()-start
    write(out/'parallel.json',dict(seconds=parallel_seconds,games=parallel))
    for a,b in zip(serial,parallel):
        for key in ['seed','score','length','max_tile','terminated','truncated']:
            assert a[key]==b[key]
        assert a['terminated'] and not a['truncated']
    assert digest(source/'policy.pt')==checksum
    result=dict(complete=True,serial_seconds=serial_seconds,parallel_seconds=parallel_seconds,
        full_loop_speedup=serial_seconds/parallel_seconds,identical_game_results=True,
        source_unchanged=True,transitions=sum(g['length'] for g in serial),
        caution='Eight-game benchmark under concurrent research load; full pool timing includes startup. '
        'Forward-logit agreement does not prove all future game decisions are bit-identical across batch shapes.')
    write(out/'summary.json',result)
    print('COMPLETE',json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
