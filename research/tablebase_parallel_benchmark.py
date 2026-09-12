"""Four preselected games measure useful CPU concurrency beside the GPU learner."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

from research.afterstate_teacher import write, digest
from research.tablebase_pilot import run


def main():
    base=Path('runs/research/tablebase_search')
    out=base/'parallel4_benchmark'
    out.mkdir(exist_ok=False)
    files=[base/'cache'/f'tuple_moves.{name}' for name in ('10a','11a')]
    assert all(p.exists() for p in files)
    hashes={p.name:digest(p) for p in files}
    seeds=list(range(8949500,8949504))
    write(out/'protocol.json',dict(seeds=seeds,depth=8,workers=4,
        per_game_seconds=1200,per_child_rss_gib=4,cache_sha256=hashes,
        reason='Sequential native search uses one CPU core and around0.4GiB sampledRSS. Measure four independent games concurrently while the existing GPU learner runs; use actual moves per wallsecond rather than utilization percentage.',
        cache_safety='All tables exist before launch. Inspected Array::Save opens with O_CREAT|O_EXCL and returns on EEXIST, so existing table files are never overwritten; MAP_PRIVATE and COW changes remain process-local. Check hashes again after workers exit.',
        limits='At most four benchmark children, plus the existing one-child paired screen. No newGPUlearner. Shared research STOP observed by all wrappers. These seeds are a throughput diagnostic, not the100game final selection suite.'))
    started=time.monotonic();results=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(run,base/f'parallel4_depth8_seed{s}',8,s,1200,4.):s for s in seeds}
        for future in as_completed(futures):
            result=future.result(); results.append(result)
            write(out/'games.json',results)
            write(out/'status.json',dict(completed=len(results),planned=4,
                elapsed_seconds=time.monotonic()-started))
    seconds=time.monotonic()-started
    final_hashes={p.name:digest(p) for p in files}
    assert final_hashes==hashes, 'Shared tables changed unexpectedly'
    completed=[r for r in results if r['complete']]
    write(out/'result.json',dict(complete=len(completed)==4,elapsed_seconds=seconds,
        games_per_minute=len(completed)*60/seconds,
        completed_transitions=sum(r['length'] for r in completed),
        moves_per_second=sum(r['length'] for r in completed)/seconds,
        max_score=max([r['score'] for r in completed],default=None),
        cache_hashes_unchanged=True,
        comparison_note='Concurrent workload includes GPU study and one additional sequentialCPUsearch; different game lengths mean this is not a controlled per-model speed multiplier. Report end-to-end throughput and resource bounds.'))


if __name__=='__main__':
    main()
