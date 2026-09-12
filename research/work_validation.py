"""Freeze the economical repeatable-work player, then evaluate 100 new games.

This estimates the fixed player's strength. It is not a paired comparison with
the older timed controller, and no parameter is selected from these outcomes.
"""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
from pathlib import Path
import time

import numpy as np

from research.afterstate_teacher import digest, write
from research.hybrid_endgame_study import BASE, ROOT, STOP, cache_fingerprints
from research.work_budget_study import fingerprints as base_fingerprints, run_game

OUT = BASE / 'work_validation100'
SEEDS = list(range(9020000, 9020100))


def fingerprints():
    paths = ('research/work_validation.py', 'research/endgame_audit.py',
             'research/endgame_continue.py', 'research/native_policy_data.py',
             'rl2048/agents/spawn_safety.py')
    return dict(base_fingerprints(), **{p: digest(ROOT/p) for p in paths})


def valid_result(r):
    return bool(r.get('complete') and r.get('terminated') and not r.get('truncated')
                and r.get('source_and_binary_unchanged')
                and r.get('exact_seeded_replay_audited') and r.get('saved_audit_checked'))


def summarize(results, seeds, elapsed, integrity_verified=False):
    good = [r for r in results if valid_result(r)]
    complete = (integrity_verified and len(results) == len(seeds)
                and len(set(seeds)) == len(seeds)
                and {r['seed'] for r in results} == set(seeds)
                and len(good) == len(seeds))
    output = dict(complete=complete, finished_attempts=len(results), planned=len(seeds),
        completed_games=len(good), incomplete_attempts=len(results)-len(good),
        elapsed_wall_seconds=elapsed, mean_score=None, score_std=None,
        mean_seconds=None, tile_reaching_rates=None,
        best_completed_game=max(good, key=lambda r: r['score']) if good else None,
        note='Frozen one-million-work hybrid on 100 fresh standard seeded games. '
             'All attempts and final integrity checks are required for means. '
             'An individual maximum is not an average or a world record. '
             'Independent seed set: no paired claim against older timed benchmarks.')
    if complete:
        scores = np.array([r['score'] for r in results], dtype=float)
        boot = np.random.default_rng(90201).choice(scores, (20000, len(scores))).mean(1)
        output.update(mean_score=float(scores.mean()), score_std=float(scores.std(ddof=1)),
            mean_score_bootstrap95=np.quantile(boot, [.025, .975]).tolist(),
            mean_length=float(np.mean([r['length'] for r in results])),
            mean_seconds=float(np.mean([r['elapsed_seconds'] for r in results])),
            total_transitions=sum(r['length'] for r in results),
            tile_reaching_rates={str(t): float(np.mean([r['max_tile'] >= t for r in results]))
                                for t in (8192, 16384, 32768, 65536, 131072)})
    return output


def main():
    if STOP.exists():
        raise RuntimeError('User STOP is present')
    screen = BASE/'work_table_ablation/screen'
    selected = json.loads((screen/'summary.json').read_text())
    assert selected['complete'] and all(json.loads((screen/'verified.json').read_text()).values())
    assert json.loads((BASE/'work_budget_repeatability/summary.json').read_text())['complete']
    OUT.mkdir(exist_ok=False)
    hashes, caches = fingerprints(), cache_fingerprints()
    write(OUT/'protocol.json', dict(seeds=SEEDS, planned_games=100, workers=16,
        per_game_seconds=3600, total_wall_seconds=7200, move_cap=100000, rss_gib_per_worker=3.,
        fingerprints=hashes, cache_sha256=caches, selected_before_evaluation=True,
        selection_screen=selected,
        reason='The completed 16-pair table comparison observed 715240 points with older '
            'tables versus 613404 without, with a wide interval including no difference. '
            'The table player was 2.68 times faster. The earlier 1M/4M screen found no '
            'established score gain from 3.25 times the runtime. Freeze the economical '
            'repeatable-work player and estimate its strength on a larger fresh set.',
        policy='Unmodified WorkBudgetHybridAgent(1000000): older solved formations first, '
            'then compact embedded tables/search. Complete iterative layers from depth3 '
            'until the last layer reaches the soft node target or the configured maximum. '
            'Overshoot is allowed; no wall-clock depth choice, new rollout override or training.',
        evaluation='All 100 standard two-tile starts, exact Python game RNG, raw merge points. '
            'No result-based policy changes, restarts or selected extensions. Every resource '
            'failure and unstarted attempt remains explicit and prevents aggregate means.',
        audit='Frozen source/binaries/caches; every actual board/spawn/reward/termination '
            'reproduced from its seed, plus recorded-decision audits. Existing benchmarks untouched.'))
    started = time.perf_counter()
    deadline = started + 7200
    active, results, queue = {}, [], iter(SEEDS)
    write(OUT/'summary.json', summarize(results, SEEDS, 0.))
    with ProcessPoolExecutor(max_workers=16) as pool:
        def submit():
            if STOP.exists() or time.perf_counter() >= deadline:
                return
            seed = next(queue, None)
            if seed is not None:
                seconds = max(1, min(3600, deadline-time.perf_counter()))
                active[pool.submit(run_game, OUT/f'seed{seed}', seed, 1000000, 100000, seconds)] = seed
        for _ in range(16):
            submit()
        while active:
            done, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
            for future in done:
                seed = active.pop(future)
                try:
                    result = future.result()
                    if result.get('complete'):
                        folder = OUT/f'seed{seed}'
                        audit = json.loads((folder/'audit.json').read_text())
                        result['saved_audit_checked'] = bool(
                            audit['seed'] == seed and audit['score'] == result['score']
                            and audit['moves'] == result['length']
                            and audit['every_transition_rechecked']
                            and audit['exact_seeded_spawn_sequence_rechecked']
                            and audit['unchanged_fingerprints']
                            and audit['replay_sha256'] == digest(folder/'replay.json'))
                except Exception as error:
                    result = dict(seed=seed, complete=False, error=repr(error))
                results.append(result)
                submit()
            elapsed = time.perf_counter()-started
            write(OUT/'games.json', results)
            write(OUT/'summary.json', summarize(results, SEEDS, elapsed))
            write(OUT/'status.json', dict(phase='running', finished=len(results), planned=100,
                active=sorted(active.values()), elapsed_wall_seconds=elapsed))
    for seed in queue:
        results.append(dict(seed=seed, complete=False, stop_reason='not_started_stop_or_wall_budget'))
    verified = dict(policy_unchanged=fingerprints()==hashes,
                    cache_unchanged=cache_fingerprints()==caches)
    write(OUT/'verified.json', verified)
    write(OUT/'games.json', results)
    summary = summarize(results, SEEDS, time.perf_counter()-started, all(verified.values()))
    write(OUT/'summary.json', summary)
    write(OUT/'status.json', dict(phase='complete' if summary['complete'] else 'incomplete',
                                finished=len(results), planned=100, active=[]))


if __name__ == '__main__':
    main()
