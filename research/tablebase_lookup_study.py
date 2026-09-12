"""Does search recover better when a table's constrained goal is very unlikely?"""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import argparse
import json
from pathlib import Path
import time

import numpy as np

from research.afterstate_teacher import digest, write
from research.tablebase_pilot import run


def main(threshold=.01, seed_start=8960000, name='lookup_gate_screen'):
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('Gate threshold must be between zero and one')
    if not name or Path(name).name != name:
        raise ValueError('Use a single study directory name')
    base = Path('runs/research/tablebase_search')
    regression = json.loads((base/'lookup_gate_default_regression/verified.json').read_text())
    assert regression['identical_complete_trajectory']
    out = base/name
    out.mkdir(exist_ok=False)
    vendor = Path('third_party/2048-ai-lookup-gate').resolve()
    binary_hash = digest(vendor/'2048')
    hashes = {p.name:digest(p) for p in (base/'cache').glob('tuple_moves.*')}
    seeds = list(range(seed_start, seed_start+12))
    thresholds = dict(original=0., gate=threshold)
    reason = ('The hybrid prioritizes tuple10 whenever its subgoal probability is positive. '
        f'Test search instead when probability<={threshold:.0%}. The1%screen changed eleven '
        'of twelve final scores by zero and the last by only28, so it barely tested whether '
        'general search can recover from a low-probability constrained plan. A10%screen is '
        'motivated by6692tuple10decisions below10% in earlier twelve-game logs. It may damage '
        'the hierarchy, and these probabilities are not immediate death probabilities. '
        'This is one bounded threshold test, not an automatic sweep or promotion.') if threshold != .01 else (
        'Test whether general search improves recovery when tuple10 subgoal probability is at most1%. '
        'Low restricted-goal probability alone is not evidence of a bad move.')
    write(out/'protocol.json', dict(seeds=seeds,depth=5,min_probability=1/512,
        tuple10_min_probabilities=thresholds,planned_games=24,workers=6,
        wall_seconds_budget=2400,per_game_seconds=600,per_child_rss_gib=2.,
        source=str(vendor),binary_sha256=binary_hash,cache_sha256=hashes,
        reason=reason,
        controls='Identical isolated binary and lookup contents, depth5, chance cutoff1/512, '
        f'only -t0 versus -t{threshold:g} changes. Default -t0 reproduces every move of the629908reference '
        'game. Neither original611kvalidation binary nor cache-safe500game binary is edited. '
        'Threshold refers to success of a restricted table goal, not probability of winning2048.',
        limits='All standard two-tile starts and complete-game validation. Preserve incomplete '
        'attempts, no undo, sharedSTOP. No automatic promotion from a12game screening mean.'))
    tasks = [(s,t) for i,s in enumerate(seeds) for t in
             (['original','gate'] if i%2==0 else ['gate','original'])]
    queue = iter(tasks)
    started = time.monotonic()
    results, pending = [], {}
    stop = Path('runs/research/scaled_transformer/STOP')
    with ProcessPoolExecutor(max_workers=6) as pool:
        def submit():
            remaining = 2400-(time.monotonic()-started)
            if stop.exists() or remaining <= 0:
                return
            item = next(queue, None)
            if item is None:
                return
            seed,label = item
            assert digest(vendor/'2048') == binary_hash
            future = pool.submit(run,base/f'{name}_{label}_seed{seed}',5,seed,
                max(1,min(600,int(remaining))),2.,str(vendor),1/512,thresholds[label])
            pending[future] = item
        for _ in range(6):
            submit()
        while pending:
            completed,_ = wait(pending,timeout=10,return_when=FIRST_COMPLETED)
            for future in completed:
                seed,label = pending.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    result = dict(seed=seed,complete=False,error=repr(error))
                results.append(result | dict(setting=label))
                submit()
            summaries = {}
            for label in thresholds:
                good = [r for r in results if r['setting']==label and r.get('complete')]
                summaries[label] = dict(complete_games=len(good),
                    mean_score=float(np.mean([r['score'] for r in good])) if good else None,
                    mean_seconds=float(np.mean([r['elapsed_seconds'] for r in good])) if good else None)
            pairs = []
            for seed in seeds:
                same = {r['setting']:r for r in results if r['seed']==seed and r.get('complete')}
                if len(same)==2:
                    pairs.append(dict(seed=seed,difference=same['gate']['score']-same['original']['score']))
            write(out/'games.json',results)
            write(out/'summary.json',dict(complete=len(results)==24 and all(r.get('complete') for r in results),
                finished_attempts=len(results),active=len(pending),by_setting=summaries,paired_results=pairs,
                mean_paired_difference=float(np.mean([p['difference'] for p in pairs])) if pairs else None,
                elapsed_seconds=time.monotonic()-started))
    final = {p.name:digest(p) for p in (base/'cache').glob('tuple_moves.*')}
    write(out/'verified.json',dict(binary_unchanged=digest(vendor/'2048')==binary_hash,
        cache_unchanged=hashes==final,finished_attempts=len(results)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threshold',type=float,default=.01)
    parser.add_argument('--seed-start',type=int,default=8960000)
    parser.add_argument('--name',default='lookup_gate_screen')
    main(**vars(parser.parse_args()))
