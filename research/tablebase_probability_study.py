"""Separate search horizon from pruning rare future spawn sequences."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import time

import numpy as np

from research.afterstate_teacher import digest, write
from research.tablebase_pilot import run


def main():
    base = Path('runs/research/tablebase_search')
    out = base/'probability_screen'
    out.mkdir(exist_ok=False)
    vendor = Path('third_party/2048-ai-cache-fix').resolve()
    stop = Path('runs/research/scaled_transformer/STOP')
    seeds = list(range(8959000, 8959012))
    cutoffs = dict(standard=1/512, wider=1/4096)
    binary_hash = digest(vendor/'2048')
    cache_hashes = {p.name:digest(p) for p in (base/'cache').glob('tuple_moves.*')}
    write(out/'protocol.json', dict(seeds=seeds, depth=5, min_probabilities=cutoffs,
        planned_games=24, workers=6, wall_seconds_budget=2400, per_game_seconds=600,
        per_child_rss_gib=1.5, binary_sha256=binary_hash, cache_sha256=cache_hashes,
        reason='The earlier depth5/depth8 test changed both search horizon and probability cutoff. '
        'Hold horizon5 and all table logic fixed; lower the chance-path pruning threshold eightfold. '
        'Hypothesis: considering more unlikely spawn sequences gives useful robustness more cheaply than adding three decision levels. '
        'This is a12-paired-game screen, not a final population benchmark. Freeze a promising setting before fresh validation.',
        protocol='Same isolated cache-safe binary, explicit -p after -d, ordinary two-tile starts, '
        'no chosen restarts, every attempt retained. CPU workers stop on shared STOP, time or RSS budget.'))
    tasks = []
    for i,seed in enumerate(seeds):
        for label in (['standard', 'wider'] if i%2 == 0 else ['wider', 'standard']):
            tasks.append((seed,label))
    queue = iter(tasks)
    pending, results = {}, []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=6) as pool:
        def submit():
            remaining = 2400-(time.monotonic()-started)
            if stop.exists() or remaining <= 0:
                return False
            item = next(queue, None)
            if item is None:
                return False
            seed,label = item
            assert digest(vendor/'2048') == binary_hash
            future = pool.submit(run, base/f'probability_{label}_seed{seed}', 5, seed,
                max(1,min(600,int(remaining))), 1.5, str(vendor), cutoffs[label])
            pending[future] = item
            return True
        for _ in range(6):
            submit()
        while pending:
            done,_ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
            for future in done:
                seed,label = pending.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    result = dict(seed=seed, complete=False, error=repr(error))
                results.append(result | dict(setting=label, min_probability=cutoffs[label]))
                submit()
            summaries = {}
            for label in cutoffs:
                good = [r for r in results if r['setting']==label and r.get('complete')]
                summaries[label] = dict(complete_games=len(good),
                    mean_score=float(np.mean([r['score'] for r in good])) if good else None,
                    mean_seconds=float(np.mean([r['elapsed_seconds'] for r in good])) if good else None)
            pairs = []
            for seed in seeds:
                same = {r['setting']:r for r in results if r['seed']==seed and r.get('complete')}
                if len(same)==2:
                    pairs.append(dict(seed=seed, difference=same['wider']['score']-same['standard']['score']))
            write(out/'games.json', results)
            write(out/'summary.json', dict(complete=len(results)==24 and all(r.get('complete') for r in results),
                finished_attempts=len(results), active=len(pending), by_setting=summaries,
                paired_results=pairs, mean_paired_difference=float(np.mean([p['difference'] for p in pairs])) if pairs else None,
                elapsed_seconds=time.monotonic()-started))
    final_hashes = {p.name:digest(p) for p in (base/'cache').glob('tuple_moves.*')}
    write(out/'verified.json', dict(binary_unchanged=digest(vendor/'2048')==binary_hash,
        cache_unchanged=cache_hashes==final_hashes, finished_attempts=len(results)))


if __name__ == '__main__':
    main()
