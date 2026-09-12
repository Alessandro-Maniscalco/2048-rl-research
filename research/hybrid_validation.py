"""Evaluate the frozen fast hybrid on 100 prespecified fresh standard games.

Select on the completed eight-pair screen: comparable observed score with wide
uncertainty, substantially lower runtime. Do not retune from partial results.
Keep all attempts, verify exact seeded games, and report a mean only if all100
finish naturally. This is local evaluation of a search/table player, not neural
training, a best-of100 average, or a world-record claim.
"""
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import json
from pathlib import Path
import time

import numpy as np

from research.afterstate_teacher import digest,write
from research.hybrid_endgame_study import BASE,STOP,run,policy_fingerprints,cache_fingerprints
from research.endgame_audit import audit

OUT=BASE/'hybrid_validation100'


def summarize(results,seeds,elapsed):
    complete=(len(results)==len(seeds) and {r['seed'] for r in results}==set(seeds)
              and all(r.get('complete') for r in results))
    finished=[r for r in results if r.get('complete')]
    summary=dict(complete=complete,finished_attempts=len(results),planned=len(seeds),
        completed_games=len(finished),incomplete_attempts=len(results)-len(finished),
        elapsed_wall_seconds=elapsed,mean_score=None,score_std=None,mean_seconds=None,
        best_completed_game=max(finished,key=lambda r:r['score']) if finished else None,
        tile_reaching_rates=None,
        note='Fresh100game frozen-policy evaluation. Partial maximum is a completed individual game, not population strength. '
            'No partial completed-only mean. No result-based restarts, omitted failures or policy changes.')
    if complete:
        scores=np.array([r['score'] for r in results],dtype=np.float64)
        means=np.random.default_rng(40811).choice(scores,(20000,len(scores)),replace=True).mean(1)
        summary.update(mean_score=float(scores.mean()),score_std=float(scores.std(ddof=1)),
            mean_score_bootstrap95=np.quantile(means,[.025,.975]).tolist(),
            mean_seconds=float(np.mean([r['elapsed_seconds'] for r in results])),
            mean_length=float(np.mean([r['length'] for r in results])),
            total_transitions=sum(r['length'] for r in results),
            tile_reaching_rates={str(t):float(np.mean([r['max_tile']>=t for r in results]))
                                 for t in (8192,16384,32768,65536,131072)})
    return summary


def main():
    paired=json.loads((BASE/'hybrid_comparison/summary.json').read_text())
    checked=json.loads((BASE/'hybrid_comparison/verified.json').read_text())
    assert paired['complete'] and checked['policy_unchanged'] and checked['cache_unchanged']
    OUT.mkdir(parents=True,exist_ok=False)
    seeds=list(range(8975000,8975100));policy_hashes=policy_fingerprints();cache_hashes=cache_fingerprints()
    runner_hash=digest(Path(__file__))
    write(OUT/'protocol.json',dict(seeds=seeds,planned_games=100,workers=16,
        seconds_per_game=7200,rss_gib_per_game=3.,max_moves=100000,
        selected_before_evaluation=True,selection_screen=paired,
        selection_reason='Hybrid saved58%ofsearch calls and completed games3.45timesfaster. The8pairs do not establish a score difference. '
            'Freeze the more computationally economical player and measure100freshgames, including rare high-tile transitions.',
        policy='Unmodified HybridEndgameAgent(time_scale=1): frozen oldtables first, then compact CoreAILogic. No risk override.',
        fingerprints=policy_hashes,cache_sha256=cache_hashes,runner_sha256=runner_hash,
        evaluation='Seeds8975000..8975099 are new. Every ordinary two-tile game runs without undo, chosen restart, artificial tile change or planner timing reset. '
            'Adaptive wall-time search depends on CPUload. No parameter changes from partial results. '
            'Do not compare as paired seeds against native-libc611854reference. The original8920000..99set remains untouched.',
        audit='Every completed replay checked for moves/spawns/rawscore/termination; every completed game additionally replayed from its exact seed. '
            'All input/source/cachefingerprints frozen. Record every failed/incomplete attempt.'))
    started=time.perf_counter();queue=iter(seeds);active={};results=[]
    summary=summarize(results,seeds,0.)
    with ProcessPoolExecutor(max_workers=16) as pool:
        def submit():
            if STOP.exists():return
            seed=next(queue,None)
            if seed is not None:active[pool.submit(run,OUT/f'seed{seed}',seed,True)]=seed
        for _ in range(16):submit()
        while active:
            finished,_=wait(active,timeout=10,return_when=FIRST_COMPLETED)
            for future in finished:
                seed=active.pop(future)
                try:
                    result=future.result()
                    if result.get('complete'):
                        audit(OUT/f'seed{seed}')
                        result['exact_seeded_replay_audited']=True
                except Exception as error:
                    result=dict(seed=seed,hybrid=True,complete=False,error=repr(error))
                results.append(result);submit()
            summary=summarize(results,seeds,time.perf_counter()-started)
            write(OUT/'games.json',results);write(OUT/'summary.json',summary)
            write(OUT/'status.json',dict(phase='running',finished=len(results),active=sorted(active.values())))
    verification=dict(policy_unchanged=policy_fingerprints()==policy_hashes,
        cache_unchanged=cache_fingerprints()==cache_hashes,runner_unchanged=digest(Path(__file__))==runner_hash)
    write(OUT/'verified.json',verification)
    if not all(verification.values()):
        summary.update(complete=False,validation_error='Frozen source/cache integrity check failed',
            mean_score=None,score_std=None,mean_seconds=None,tile_reaching_rates=None)
        summary.pop('mean_score_bootstrap95',None)
        write(OUT/'summary.json',summary)
    write(OUT/'status.json',dict(phase='complete' if summary['complete'] else 'incomplete',finished=len(results),active=[]))


if __name__=='__main__':main()
