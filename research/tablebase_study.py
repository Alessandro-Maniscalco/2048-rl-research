"""Preselected paired depth screen; one owned CPU process, reusable exact tables."""
import contextlib
import json
from pathlib import Path
import time

import numpy as np

from research.afterstate_teacher import write
from research.tablebase_pilot import run


def main():
    base = Path('runs/research/tablebase_search')
    out = base/'paired_depth_screen'
    out.mkdir(exist_ok=False)
    seeds = list(range(8949100, 8949110))
    protocol = dict(seeds=seeds, depths=[5,8], planned_games=20,
        wall_seconds_budget=7200, per_game_seconds=1200, rss_gib=8,
        reason='Depth5 and depth8 both scored629908 on the pilot seed, but depth8 cost314s versus14s. Test preselected paired fresh seeds before paying that search cost for a large evaluation.',
        comparison='Same native program and libc RNG seed. The depth option also adjusts probability pruning as specified by the upstream program. Cached subproblem tables are shared across sequential games; cache generation is not neural training.',
        decision='Use all completed paired games, report score and cost. Do not call10seeds a100game estimate or select a lucky record as the best model. Choose the next100game validation only after this bounded screen.',
        source='https://github.com/macroxue/2048-ai',
        limitations='External search and solved tables, not locally trained neural model or GPT-selected moves. This is a screening set, not the reserved final neural test.')
    write(out/'protocol.json', protocol)
    started = time.monotonic()
    results = []
    stop = Path('runs/research/scaled_transformer/STOP')
    for i,seed in enumerate(seeds):
        # Alternate execution order to avoid always favoring a warmed-cache arm.
        for depth in ([5,8] if i%2 == 0 else [8,5]):
            remaining = 7200 - (time.monotonic()-started)
            if stop.exists() or remaining < 1:
                write(out/'status.json', dict(phase='stopped', completed=len(results)))
                return
            job = base/f'screen_depth{depth}_seed{seed}'
            write(out/'status.json', dict(phase='running', depth=depth, seed=seed,
                completed=len(results), elapsed_seconds=time.monotonic()-started))
            try:
                with (out/'runner.log').open('a') as log, contextlib.redirect_stdout(log):
                    result = run(job,depth,seed,min(1200,int(remaining)),8.)
            except Exception as error:
                write(out/'status.json',dict(phase='failed', depth=depth,seed=seed,error=repr(error)))
                raise
            results.append(result | dict(depth=depth, folder=str(job)))
            write(out/'games.json', results)
            summaries = {}
            for d in (5,8):
                games = [r for r in results if r['depth']==d and r['complete']]
                summaries[str(d)] = dict(completed=len(games),
                    mean_score=float(np.mean([r['score'] for r in games])) if games else None,
                    mean_seconds=float(np.mean([r['elapsed_seconds'] for r in games])) if games else None,
                    best_score=max([r['score'] for r in games],default=None))
            pairs = []
            for s in seeds:
                g = {r['depth']:r for r in results if r['seed']==s and r['complete']}
                if len(g)==2:
                    pairs.append(dict(seed=s, difference=g[8]['score']-g[5]['score']))
            write(out/'summary.json',dict(by_depth=summaries, paired_games=pairs,
                mean_paired_difference=float(np.mean([p['difference'] for p in pairs])) if pairs else None,
                complete=len(results)==20 and all(r['complete'] for r in results),
                elapsed_seconds=time.monotonic()-started))
            if not result['complete']:
                write(out/'status.json',dict(phase='stopped',reason='incomplete_game', completed=len(results)))
                return
    write(out/'status.json',dict(phase='complete',completed=len(results)))


if __name__ == '__main__':
    main()
