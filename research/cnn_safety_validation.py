"""Fresh200-game replication of frozen CNN spawn safety, using measured CPU execution."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import time

import numpy as np
import torch

from research.afterstate_teacher import digest, write
from rl2048.agents.neural import NeuralAgent, policy_logits, tensor_boards
from rl2048.agents.spawn_safety import safest_policy_scores
from rl2048.afterstate_compare import evaluate_batch


BASE = Path('runs/research/scaled_transformer')
OUT = BASE/'cnn_spawn_safety_validation200'
AGENT = None


def initialize(source):
    global AGENT
    torch.set_num_threads(1)
    AGENT = NeuralAgent.load(source, 'cpu')


def evaluate_block(label, seeds, deadline):
    counts = dict(active_decisions=0, changed_decisions=0, sum_risk_reduction=0.)

    @torch.no_grad()
    def decision(boards):
        if (BASE/'STOP').exists():
            raise InterruptedError('user_stop')
        # Measured CPU batch-size cliff: chunk8 is faster than32/128 for this CNN.
        logits = np.concatenate([policy_logits(AGENT.policy, tensor_boards(boards[i:i+8], 'cpu')).numpy()
                                 for i in range(0, len(boards), 8)])
        safe, risks, legal = safest_policy_scores(logits, boards)
        plain = np.where(legal, logits, -np.inf)
        if label == 'safety':
            valid = np.flatnonzero(legal.any(1))
            old, new = plain.argmax(1), safe.argmax(1)
            counts['active_decisions'] += len(valid)
            counts['changed_decisions'] += int((old[valid] != new[valid]).sum())
            counts['sum_risk_reduction'] += float((risks[valid, old[valid]]-risks[valid, new[valid]]).sum())
        return safe if label == 'safety' else plain

    result = evaluate_batch(None, seeds, decision=decision, deadline=min(deadline, time.time()+300), max_steps=100000)
    return dict(label=label, seeds=seeds, result=result, intervention_counts=counts)


def summarize(games):
    scores = np.array([g['score'] for g in games])
    tiles = np.array([g['max_tile'] for g in games])
    return dict(complete=len(games)==200 and all(g['complete'] and not g['truncated'] for g in games),
        episodes=len(games), mean_score=float(scores.mean()), score_std=float(scores.std(ddof=1)),
        mean_length=float(np.mean([g['length'] for g in games])), transitions=sum(g['length'] for g in games),
        tile_reaching_rates={str(t):float((tiles>=t).mean()) for t in [2048,4096,8192,16384,32768,65536]},
        max_tile_distribution={str(t):int((tiles==t).sum()) for t in np.unique(tiles)},
        truncated_episodes=sum(g['truncated'] for g in games))


def main():
    OUT.mkdir(exist_ok=False)
    source = Path('runs/research/pretrained_cnn/original').resolve()
    checksum = digest(source/'policy.pt')
    first = json.loads((BASE/'cnn_spawn_safety/comparison.json').read_text())
    assert first['complete']
    benchmark = json.loads(Path('runs/research/compute/cnn_cpu_execution/summary.json').read_text())
    assert benchmark['complete'] and benchmark['identical_game_results']
    seeds = list(range(8958800, 8959000))
    write(OUT/'protocol.json', dict(source=str(source),source_sha256=checksum,seeds=seeds,
        games_per_arm=200, cpu_workers=4, torch_threads_per_worker=1, inference_chunk=8,
        games_per_block=25, max_wall_seconds=1200, per_block_seconds=300,max_steps=100000,
        reason='The first100fresh-pair study improved80568to90082with a positive paired interval. '
        'Replicate the unchanged original CNN versus the same exact one-spawn risk intervention '
        'on200new preselected seeds before treating this as a robust playing improvement. '
        'Use4persistent CPUworkers and chunk8inference based on actual execution benchmarks. '
        'Both arms share that path; no network weights, reward, search depth or threshold change.',
        limitations='Independent spawn seeds, same frozen source and safety rule. Not neural training. '
        'Smaller CPU batches can change rounding, so do not treat different batch executions as '
        'bit-identical policies without trajectory checks. Both arms compute risk statistics.',
        previous_100_game_comparison=first))
    started=time.time()
    blocks=[]
    with ProcessPoolExecutor(max_workers=4, initializer=initialize, initargs=(str(source),)) as pool:
        pending={pool.submit(evaluate_block,label,seeds[i:i+25],started+1200):(label,i)
                 for i in range(0,len(seeds),25) for label in ['baseline','safety']}
        while pending:
            finished,_=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
            for future in finished:
                label,index=pending.pop(future)
                try:
                    block=future.result()
                except Exception as error:
                    block=dict(label=label,seed_start=seeds[index],error=repr(error))
                blocks.append(block)
                write(OUT/f'{label}_{index:03d}.json',block)
            write(OUT/'status.json',dict(phase='evaluating',finished_blocks=len(blocks),
                pending_blocks=len(pending),elapsed_seconds=time.time()-started))
    if any('error' in b or not b['result']['summary']['complete'] or
           b['result']['summary']['truncated_episodes'] for b in blocks):
        write(OUT/'status.json',dict(phase='incomplete',finished_blocks=len(blocks)))
        return
    results={}
    for label in ['baseline','safety']:
        games=sorted([g for b in blocks if b['label']==label for g in b['result']['episodes']],key=lambda g:g['seed'])
        assert [g['seed'] for g in games]==seeds
        results[label]=dict(summary=summarize(games),episodes=games)
        write(OUT/f'{label}.json',results[label])
    delta=np.array([s['score']-b['score'] for b,s in zip(results['baseline']['episodes'],results['safety']['episodes'])])
    boot=delta[np.random.default_rng(8958799).integers(200,size=(20000,200))].mean(1)
    counts={k:sum(b['intervention_counts'][k] for b in blocks if b['label']=='safety')
            for k in ['active_decisions','changed_decisions','sum_risk_reduction']}
    comparison=dict(complete=True,baseline_mean=results['baseline']['summary']['mean_score'],
        safety_mean=results['safety']['summary']['mean_score'],gain=float(delta.mean()),
        wins=int((delta>0).sum()),ties=int((delta==0).sum()),paired_game_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
        intervention_counts=counts,elapsed_seconds=time.time()-started,
        note='Separate200fresh paired-seed replication; original100seeds excluded. Same frozen weights and CPU execution in both arms.')
    assert digest(source/'policy.pt')==checksum
    write(OUT/'comparison.json',comparison)
    write(OUT/'verified.json',dict(complete=True,source_weights_unchanged=True,paired_seeds_match=True))
    write(OUT/'status.json',dict(phase='complete',elapsed_seconds=time.time()-started))
    print('COMPARISON',json.dumps(comparison),flush=True)


if __name__ == '__main__':
    main()
