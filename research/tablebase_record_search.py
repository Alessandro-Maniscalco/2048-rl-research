"""A fixed-budget search for high individual scores, distinct from mean-score validation."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import time
import numpy as np

from research.afterstate_teacher import write, digest
from research.tablebase_pilot import run


def main():
    base = Path('runs/research/tablebase_search')
    gate = json.loads((base/'cachefix_depth5_seed8949000/comparison.json').read_text())
    assert gate['identical_complete_trajectory']
    out = base/'record500'
    out.mkdir(exist_ok=False)
    vendor = Path('third_party/2048-ai-cache-fix').resolve()
    stop = Path('runs/research/scaled_transformer/STOP')
    seeds = list(range(8950000, 8950500))
    binary_hash = digest(vendor/'2048')
    cache_paths = [base/'cache'/f'tuple_moves.{name}' for name in ('10a','11a')]
    cache_hashes = {p.name:digest(p) for p in cache_paths}
    write(out/'protocol.json', dict(seeds=seeds,planned_attempts=500,workers=4,depth=5,
        wall_seconds_budget=7200,per_game_seconds=600,per_child_rss_gib=1.5,
        source=str(vendor),binary_sha256=binary_hash,cache_sha256=cache_hashes,
        objective='Highest individual raw score among complete standard games in this fixed batch. Report attempted-game count with every record; keep the 100-game depth8 mean-score validation separate.',
        reason='The paired screen measured 9.48 seconds/game at depth5 versus322.83 at8. More lower-cost attempts may find rarer high-score outcomes within a wall-time budget. This tests a record-search allocation, not superior population mean. Use four additional CPU workers beside the existing eight-worker validation and single MPS learner.',
        correctness='Isolated binary bypasses the ambiguous four-bit transposition key only when a board contains tile rank16 or greater. Synthetic poisoning regression fails on original and passes here. A complete depth5 pilot exactly matches all22345baseline moves at629908. Existing depth8 validation binary is unchanged.',
        limits='Preselected fresh seeds, ordinary two-tile starts, no undo. Every move/spawn/score validated. Incomplete attempts remain recorded and cannot set a completed-game record. No claim of a world record or a neural training result. Shared STOP halts owned workers.'))
    started = time.monotonic()
    pending, results = {}, []
    seed_queue = iter(seeds)
    with ProcessPoolExecutor(max_workers=4) as pool:
        def submit():
            remaining = 7200-(time.monotonic()-started)
            if stop.exists() or remaining <= 0:
                return False
            seed = next(seed_queue, None)
            if seed is None:
                return False
            assert digest(vendor/'2048')==binary_hash
            task = pool.submit(run,base/f'record_depth5_seed{seed}',5,seed,
                max(1,min(600,int(remaining))),1.5,str(vendor))
            pending[task] = seed
            return True
        for _ in range(4):
            submit()
        while pending:
            done,_ = wait(pending,timeout=10,return_when=FIRST_COMPLETED)
            for task in done:
                seed = pending.pop(task)
                try:
                    result = task.result()
                except Exception as error:
                    result = dict(seed=seed,complete=False,error=repr(error))
                results.append(result)
                submit()
            completed = [r for r in results if r.get('complete')]
            best = max(completed,key=lambda r:r['score']) if completed else None
            summary = dict(finished_attempts=len(results),active=len(pending),
                completed_games=len(completed),incomplete_attempts=len(results)-len(completed),
                planned_attempts=500,batch_complete=len(results)==500,
                all_games_complete=len(completed)==500,
                elapsed_seconds=time.monotonic()-started,
                best_completed_game=best,
                completed_game_mean=float(np.mean([r['score'] for r in completed])) if completed else None,
                reaching_65536=sum(r['max_tile']>=65536 for r in completed),
                note='Record-seeking batch, not a matched mean-score algorithm comparison. Partial means exclude incomplete games and are explicitly provisional; failed attempts remain in games.json. Report attempts alongside a best score.')
            write(out/'games.json',results)
            write(out/'summary.json',summary)
            write(out/'status.json',dict(phase='running',active=len(pending),finished=len(results)))
    final_hashes = {p.name:digest(p) for p in cache_paths}
    write(out/'cache_check.json',dict(unchanged=cache_hashes==final_hashes,sha256=final_hashes))
    write(out/'status.json',dict(phase='complete' if len(results)==500 else 'stopped',finished=len(results)))


if __name__=='__main__':
    main()
