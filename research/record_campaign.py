"""Bounded fresh-game record attempts using the unchanged best local player.

Record hunting is separate from held-out mean-score evaluation. Every attempt
is retained. Run from the repository root after installing the native engines.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
from pathlib import Path
import shutil
import time

from research.afterstate_teacher import digest, write
from research.hybrid_endgame_study import ROOT, STOP, cache_fingerprints
from research.work_budget_study import fingerprints, run_game


def main(out, games=1000, workers=16, first_seed=9040000, hours=12.):
    if STOP.exists():
        raise RuntimeError('STOP is present; remove it only after an explicit resume request.')
    if games < 1 or not 1 <= workers <= 16 or hours <= 0:
        raise ValueError('Positive game/time budgets and 1..16 workers required.')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    hashes, caches = fingerprints(), cache_fingerprints()
    runner_hash = digest(Path(__file__))
    seeds = list(range(first_seed, first_seed+games))
    write(out/'protocol.json', dict(planned_games=games, seeds=seeds, workers=workers,
        wall_hours=hours, seconds_per_game=3600, move_cap=200000,
        fingerprints=hashes, cache_sha256=caches, runner_sha256=runner_hash,
        objective='Highest verified individual score across fresh standard games; not model training or a population-strength comparison.',
        policy='Unmodified WorkBudgetHybridAgent(1000000), the policy that produced the verified1357916-point local record.',
        reason='More independent attempts with the strongest economical verified local player can discover rare high-score trajectories. '
               'The4M-work screen had no established strength gain and cost3.25times as much. Keep1M to maximize attempts per hour.',
        rules='Standard4x4 two-tile starts, exact90/10 spawns, no undo, no chosen restarts, raw merge points. '
              'Every failure/partial/unstarted attempt retained. Stop on sharedSTOP, wall budget or disk space below20GiB. '
              'The paused17-complete/16-partial September9 evaluation is preserved separately.'))
    started = time.perf_counter()
    deadline = started + hours*3600
    active, results, queue = {}, [], iter(seeds)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        def submit():
            if STOP.exists() or time.perf_counter() >= deadline or shutil.disk_usage(ROOT).free < 20*1024**3:
                return
            seed = next(queue, None)
            if seed is not None:
                active[pool.submit(run_game, out/f'seed{seed}', seed, 1000000, 200000,
                                   max(1, min(3600, deadline-time.perf_counter())))] = seed
        for _ in range(workers):
            submit()
        while active:
            done, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
            for future in done:
                seed = active.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    result = dict(seed=seed, complete=False, error=repr(error))
                results.append(result)
                submit()
            good = [r for r in results if r.get('complete') and r.get('exact_seeded_replay_audited')
                    and r.get('source_and_binary_unchanged')]
            write(out/'games.json', results)
            write(out/'status.json', dict(phase='running', finished=len(results), planned=games,
                verified_complete=len(good), active=sorted(active.values()),
                elapsed_seconds=time.perf_counter()-started,
                best_verified_game=max(good, key=lambda r: r['score']) if good else None))
    for seed in queue:
        results.append(dict(seed=seed, complete=False, stop_reason='not_started_stop_time_or_disk_limit'))
    verified = dict(policy_unchanged=fingerprints()==hashes, cache_unchanged=cache_fingerprints()==caches,
                    runner_unchanged=digest(Path(__file__))==runner_hash)
    good = [r for r in results if r.get('complete') and r.get('exact_seeded_replay_audited')
            and r.get('source_and_binary_unchanged')]
    write(out/'verified.json', verified)
    write(out/'games.json', results)
    write(out/'status.json', dict(phase='complete' if len(good)==games and all(verified.values()) else 'stopped_or_incomplete',
        finished=len(results), planned=games, verified_complete=len(good), active=[],
        elapsed_seconds=time.perf_counter()-started,
        best_verified_game=max(good, key=lambda r: r['score']) if good and all(verified.values()) else None))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--games', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--first-seed', type=int, default=9040000)
    parser.add_argument('--hours', type=float, default=12.)
    main(**vars(parser.parse_args()))
