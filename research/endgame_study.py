"""Eight prespecified compact-player games after compatibility checks.

The first separately running pilot is not duplicated or pooled into this screen.
Separate processes avoid sharing the upstream mutable search tables/cache.
"""
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import json
import time

import numpy as np

from research.afterstate_teacher import write
from research.endgame_experiment import run,BASE,ROOT,fingerprints


def main():
    out=BASE/'screen8';out.mkdir(exist_ok=False)
    start=time.perf_counter();seeds=list(range(8972001,8972009));hashes=fingerprints()
    write(out/'protocol.json',dict(seeds=seeds,workers=8,time_scale=1.,per_game_seconds=1800,
        per_game_rss_gib=2.,fingerprints=hashes,
        reason='Portable core and400movement cases validated; first independent pilot exceeded1280legalmoves '
            'at~0.27GiBpeakRSS. Run8fresh full games in separate processes to measure whether the materially '
            'different compact engine is competitive. Parallel games use CPU throughput without shared mutable '
            'native cache/state. The first compatibility pilot is separate and not duplicated. No neural weights '
            'are trained, no large tables generated, no parameter retuning within this screen.',
        interpretation='All attempts retained, incomplete games never silently excluded from a claimed mean. '
            'Eightgame screening mean is not the100game611854reference. Adaptive wall-time search depends on CPUload.'))
    results=[];future_seeds={}
    with ProcessPoolExecutor(max_workers=8) as pool:
        if not (ROOT/'runs/research/scaled_transformer/STOP').exists():
            for seed in seeds:
                f=pool.submit(run,BASE/f'pilot_seed{seed}',seed,1.,1800,2.)
                future_seeds[f]=seed
        while future_seeds:
            completed,_=wait(future_seeds,timeout=10,return_when=FIRST_COMPLETED)
            for future in completed:
                seed=future_seeds.pop(future)
                try:result=future.result()
                except Exception as error:result=dict(seed=seed,complete=False,error=repr(error))
                results.append(result)
            good=[r for r in results if r.get('complete')]
            summary=dict(complete=len(good)==8,finished=len(results),active=len(future_seeds),planned=8,
                failures=sum(not r.get('complete') for r in results),
                provisional_completed_mean=float(np.mean([r['score'] for r in good])) if good else None,
                mean_score=float(np.mean([r['score'] for r in good])) if len(good)==8 else None,
                elapsed_seconds=time.perf_counter()-start)
            write(out/'games.json',results);write(out/'summary.json',summary)
            write(out/'status.json',dict(phase='running' if future_seeds else 'complete' if summary['complete'] else 'incomplete',**summary))
    write(out/'verified.json',dict(source_binary_unchanged=fingerprints()==hashes,attempts=len(results)))


if __name__=='__main__':main()
