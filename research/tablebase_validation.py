"""Freeze depth from the completed screen, then evaluate100fresh standard games."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import json
import time

import numpy as np

from research.afterstate_teacher import write, digest
from research.tablebase_pilot import run


def main():
    base=Path('runs/research/tablebase_search')
    out=base/'validation100'
    out.mkdir(exist_ok=False)
    stop=Path('runs/research/scaled_transformer/STOP')
    seeds=list(range(8949200,8949300))
    write(out/'protocol.json',dict(seeds=seeds,planned_games=100,workers=8,
        max_wall_seconds=14400,per_game_seconds=1800,per_child_rss_gib=3,
        selection='Wait for all10paired5/8screen games; choose the depth with higher completed mean score, breaking an exact tie in favor of5. Freeze the choice before any of these100freshgames. Do not retune from their results.',
        reason='The preliminary depth8screen substantially outperforms5. A four-worker benchmark completed110162moves in442.52s with per-childRSS at most0.543GiB, plus an existing CPU screen andGPUlearner. Test eight workers on the fresh100game evaluation to improve useful CPU throughput. Native cache files are immutable after creation; hash check at end.',
        limits='All owned children observe sharedSTOP and individualtime/RSSgates; any incomplete game makes the100game result incomplete. Partial means are explicitly provisional and exclude no failures silently. This is an external search/table player, not newly trained neural weights or GPT choosing actions.'))
    screen=base/'paired_depth_screen/summary.json'
    while True:
        if stop.exists():
            write(out/'status.json',dict(phase='stopped',reason='user_stop'));return
        s=json.loads(screen.read_text()) if screen.exists() else {}
        if s.get('complete'):break
        state=base/'paired_depth_screen/status.json'
        status=json.loads(state.read_text()) if state.exists() else {}
        if status.get('phase') in ('failed','stopped'):
            write(out/'status.json',dict(phase='blocked',reason='screen_incomplete'));return
        write(out/'status.json',dict(phase='waiting_for_completed_screen'))
        time.sleep(10)
    depth=8 if s['by_depth']['8']['mean_score']>s['by_depth']['5']['mean_score'] else 5
    files=[base/'cache'/f'tuple_moves.{n}' for n in ('10a','11a')]
    hashes={p.name:digest(p) for p in files}
    write(out/'selection.json',dict(depth=depth,screen=s,cache_sha256=hashes,
        selected_before_validation=True))
    started=time.monotonic();results=[];queue=iter(seeds);future_seeds={}
    payload=dict(complete=False)  # STOP can arrive before the first job is submitted.
    with ProcessPoolExecutor(max_workers=8) as pool:
        def submit():
            if stop.exists() or time.monotonic()-started>=14400:return False
            seed=next(queue,None)
            if seed is None:return False
            remaining=14400-(time.monotonic()-started)
            future=pool.submit(run,base/f'validation_depth{depth}_seed{seed}',depth,seed,
                max(1,min(1800,int(remaining))),3.)
            future_seeds[future]=seed
            return True
        for _ in range(8):submit()
        while future_seeds:
            completed,_=wait(future_seeds,timeout=10,return_when=FIRST_COMPLETED)
            for future in completed:
                seed=future_seeds.pop(future)
                try:result=future.result()
                except Exception as error:
                    result=dict(seed=seed,complete=False,error=repr(error))
                results.append(result)
                if result.get('complete'):submit()
                # Failure doesn't disappear from the denominator or become a newseed.
            good=[r for r in results if r.get('complete')]
            bad=[r for r in results if not r.get('complete')]
            scores=np.array([r['score'] for r in good])
            payload=dict(depth=depth,finished=len(results),completed_games=len(good),
                incomplete_games=len(bad),planned_games=100,
                complete=len(good)==100 and not bad,
                elapsed_seconds=time.monotonic()-started,
                mean_score=float(scores.mean()) if len(scores) else None,
                score_std=float(scores.std(ddof=1)) if len(scores)>1 else None,
                maximum_score=int(scores.max()) if len(scores) else None,
                mean_length=float(np.mean([r['length'] for r in good])) if good else None,
                tile_reaching_rates={str(tile):float(np.mean([r['max_tile']>=tile for r in good])) if good else None
                    for tile in (2048,4096,8192,16384,32768,65536)},
                note='Partial completed-game mean is provisional; use complete=True before reporting a100game result. Fixed fresh seeds, no chosen restarts or artificialstarts. Includes everyfailedattempt explicitly.')
            write(out/'games.json',results);write(out/'summary.json',payload)
            write(out/'status.json',dict(phase='running',active=len(future_seeds),finished=len(results)))
    final_hashes={p.name:digest(p) for p in files}
    write(out/'cache_check.json',dict(unchanged=hashes==final_hashes,sha256=final_hashes))
    write(out/'status.json',dict(phase='complete' if payload['complete'] else 'incomplete',finished=len(results)))


if __name__=='__main__':
    main()
