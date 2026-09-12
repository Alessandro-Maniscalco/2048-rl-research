"""Isolate one more search horizon level in the strongest strategy family."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import time

import numpy as np

from research.afterstate_teacher import digest, write
from research.tablebase_pilot import run


def main():
    base=Path('runs/research/tablebase_search')
    out=base/'depth8_vs9_fixed_chance'
    out.mkdir(exist_ok=False)
    vendor=Path('third_party/2048-ai-cache-fix').resolve()
    binary_hash=digest(vendor/'2048')
    tables={p.name:digest(p) for p in (base/'cache').glob('tuple_moves.*')}
    seeds=list(range(8970000,8970008))
    protocol=dict(seeds=seeds,depths=[8,9],min_probability=1/4096,workers=6,planned_games=16,
        max_wall_seconds=5400,per_game_seconds=1800,per_child_rss_gib=2.,
        binary=str(vendor/'2048'),binary_sha256=binary_hash,table_sha256=tables,
        reason='Depth8 substantially beat depth5 in the original pilot but changed probability pruning too. '
        'The separate fixed-depth probability-cutoff test did not establish benefit, and overriding '
        'table recommendations at10% was harmful. Keep the reliable table hierarchy unchanged and '
        'test one more search level at a fixed1/4096chance cutoff. Eight preselected paired games '
        'screen whether extra planning helps the strongest current strategy family. '
        'Measure complete-game score and whole-run cost; do not promote an eight-game mean as '
        'a replacement for the100-game validation or call a selected maximum a world record.',
        controls='Both arms use the same immutable high-rank-cache-safe binary, tables, natural '
        'two-tile starts and fixed chance cutoff. Explicit-p follows-d because-d resets it. '
        'No lookup-confidence gate, new heuristic, weight training, artificial restart or undo.',
        limits='Shared STOP and per-game time/RSS limits apply to every child. All sixteen games '
        'must complete to interpret a paired endpoint. Preserve timeouts rather than silently '
        'discarding difficult seeds. This is external CPU search, separate from MPS neural learning.')
    write(out/'protocol.json',protocol)
    tasks=iter([(seed,depth) for i,seed in enumerate(seeds) for depth in ([8,9] if i%2==0 else [9,8])])
    started=time.monotonic();results=[];pending={}
    stop=Path('runs/research/scaled_transformer/STOP')
    with ProcessPoolExecutor(max_workers=6) as pool:
        def submit():
            remaining=5400-(time.monotonic()-started)
            if stop.exists() or remaining<=0:return
            task=next(tasks,None)
            if task is None:return
            seed,depth=task
            assert digest(vendor/'2048')==binary_hash
            future=pool.submit(run,base/f'deeper_d{depth}_seed{seed}',depth,seed,
                max(1,min(1800,int(remaining))),2.,str(vendor),1/4096)
            pending[future]=task
        for _ in range(6):submit()
        while pending:
            finished,_=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
            for future in finished:
                seed,depth=pending.pop(future)
                try:result=future.result()
                except Exception as error:result=dict(seed=seed,complete=False,error=repr(error))
                results.append(result|dict(depth=depth));submit()
            by_depth={}
            for depth in [8,9]:
                good=[r for r in results if r['depth']==depth and r.get('complete')]
                by_depth[str(depth)]=dict(completed_games=len(good),
                    provisional_mean_score=float(np.mean([r['score'] for r in good])) if good else None,
                    mean_seconds=float(np.mean([r['elapsed_seconds'] for r in good])) if good else None)
            paired=[]
            for seed in seeds:
                same={r['depth']:r for r in results if r['seed']==seed and r.get('complete')}
                if len(same)==2:paired.append(dict(seed=seed,difference=same[9]['score']-same[8]['score']))
            complete=len(results)==16 and all(r.get('complete') for r in results)
            summary=dict(complete=complete,finished_attempts=len(results),active=len(pending),
                by_depth=by_depth,paired=paired,elapsed_seconds=time.monotonic()-started)
            if complete:
                delta=np.array([p['difference'] for p in paired])
                boot=delta[np.random.default_rng(8970020).integers(8,size=(20000,8))].mean(1)
                summary.update(mean_paired_difference=float(delta.mean()),wins=int((delta>0).sum()),
                    ties=int((delta==0).sum()),paired_game_bootstrap95=np.quantile(boot,[.025,.975]).tolist())
            write(out/'games.json',results);write(out/'summary.json',summary)
            write(out/'status.json',dict(phase='complete' if complete else 'running',active=len(pending),finished_attempts=len(results)))
    unchanged=digest(vendor/'2048')==binary_hash and tables=={p.name:digest(p) for p in (base/'cache').glob('tuple_moves.*')}
    write(out/'verified.json',dict(binary_and_tables_unchanged=unchanged,finished_attempts=len(results)))
    write(out/'status.json',dict(phase='complete' if len(results)==16 and all(r.get('complete') for r in results) else 'incomplete',active=0))


if __name__=='__main__':main()
