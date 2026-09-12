"""Fresh paired games: frozen hybrid versus a selective exact tactical check.

Run only after the frozen 100-game evaluation has finished and passed integrity
checks. This experiment changes one decision rule, not table confidence,
compact search settings, game rewards, or the candidate's eight-move horizon.
"""
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
from pathlib import Path
import resource
import time

import numpy as np

from research.afterstate_teacher import digest, write
from research.endgame_audit import audit
from research.hybrid_endgame_study import BASE, ROOT, STOP, cache_fingerprints, policy_fingerprints
from research.tactical_hybrid import TacticalHybridAgent
from rl2048.agents.hybrid_endgame import HybridEndgameAgent
from rl2048.game import ACTION_NAMES, Game2048
from rl2048.view import COLORS

OUT = BASE / 'tactical_comparison'
SEEDS = list(range(8976000, 8976012))


def fingerprints():
    paths = ['research/tactical_hybrid.py', 'research/exact_endgame_diagnostic.py',
             'research/tactical_hybrid_study.py', 'rl2048/fast2048.py',
             'rl2048/agents/spawn_safety.py', 'rl2048/agents/ntuple.py', 'rl2048/game.py']
    return dict(policy_fingerprints(), **{p: digest(ROOT / p) for p in paths})


def run_game(folder, seed, tactical, seconds=7200, move_cap=100000):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    hashes = fingerprints()
    write(folder / 'protocol.json', dict(seed=seed, tactical=tactical,
        seconds_budget=seconds, move_cap=move_cap, rss_gib_budget=3., fingerprints=hashes,
        policy='Hybrid plus exact eight-move tactical check' if tactical else 'Frozen hybrid control',
        intervention='If the proposed action has greater immediate death risk than another legal '
            'action, enumerate all legal actions and spawns for eight moves. Override only for '
            'strictly greater expected raw points after complete calculation. Preserve proposal '
            'on ties, 500000-state limit or 30-second limit. Risk is a trigger, not the objective.',
        game='Exact Python Game2048; ordinary two-tile start, 90/10 spawns and raw merge rewards. '
            'No undo, chosen restart, artificial tile alteration or planner timing reset.',
        timing='Identical compact time_scale=1 in both arms. Additional tactical cost is included. '
            'Wall-time search can vary with CPU load; paired seeds do not force identical prefixes.'))
    started = time.perf_counter()
    agent = (TacticalHybridAgent if tactical else HybridEndgameAgent)(1.)
    env = Game2048()
    board, info = env.reset(seed=seed)
    frames = [dict(board=board.tolist(), score=0, action=None, reward=0)]
    decisions = []
    done = False
    reason = None
    peak = 0.
    try:
        while not done:
            peak = max(peak, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3)
            if STOP.exists(): reason = 'stop_file'; break
            if time.perf_counter() - started >= seconds: reason = 'time_budget'; break
            if peak > 3.: reason = 'memory_budget'; break
            if len(decisions) >= move_cap: reason = 'move_cap'; break
            try:
                action = agent.act(board, info['action_mask'])
            except Exception as error:
                reason = repr(error)
                break
            board, reward, done, truncated, info = env.step(action)
            assert not truncated
            decisions.append(agent.last_decision)
            frames.append(dict(board=board.tolist(), score=env.score, action=action, reward=reward))
            if len(decisions) % 128 == 0 or done:
                write(folder / 'status.json', dict(phase='complete' if done else 'running',
                    seed=seed, tactical=tactical, score=env.score, moves=len(decisions),
                    max_tile=int(board.max()), elapsed_seconds=time.perf_counter()-started))
    finally:
        agent.tables.close()
    checks = [d['tactical'] for d in decisions if 'tactical' in d]
    result = dict(seed=seed, tactical=tactical, score=env.score, length=len(decisions),
        max_tile=int(board.max()), complete=done, terminated=done, truncated=not done,
        stop_reason=reason, elapsed_seconds=time.perf_counter()-started, peak_rss_gib=peak,
        modes=dict(Counter(d['mode'] for d in decisions)), planner_timing_state_reset=False,
        tactical_checks=len(checks), tactical_overrides=sum(c['overrode'] for c in checks),
        tactical_budget_exhaustions=sum(not c['complete'] for c in checks),
        tactical_seconds=sum(c['seconds'] for c in checks),
        source_and_binary_unchanged=fingerprints() == hashes)
    replay = dict(algorithm=agent.display_name, result=result, frames=frames,
                  actions=ACTION_NAMES, colors=COLORS)
    write(folder / 'decisions.json', decisions)
    write(folder / 'replay.json', replay)
    (folder / 'replay.html').write_text((ROOT / 'rl2048/replay.html').read_text().replace(
        '__REPLAY_DATA__', json.dumps(replay).replace('<', '\\u003c')))
    timing = {k: getattr(agent.logic, k) for k in
              ('last_depth', 'last_sum', 'last_prune', 'last_move', 'time_ratio', 'time_limit_ratio')}
    write(folder / 'planner_timing_state.json',
          {k: v.item() if isinstance(v, np.generic) else v for k, v in timing.items()})
    write(folder / 'result.json', result)
    if done:
        audit(folder)
        result['exact_seeded_replay_audited'] = True
        write(folder / 'result.json', result)
    write(folder / 'status.json', dict(phase='complete' if done else 'incomplete', **result))
    return result


def summarize(results, seeds=SEEDS):
    expected = {(s, t) for s in seeds for t in (False, True)}
    complete = (len(results) == len(expected) and
        {(r['seed'], r['tactical']) for r in results} == expected and
        all(r.get('complete') and r.get('exact_seeded_replay_audited') and
            r.get('source_and_binary_unchanged') for r in results))
    summary = dict(complete=complete, finished=len(results), planned=len(expected),
        incomplete_attempts=sum(not r.get('complete', False) for r in results),
        arms={}, paired_difference=None,
        note='Fresh paired screening, not a benchmark record. No mean until both full endpoints '
            'are audited. Timing-dependent search can vary with machine load. '
            'Do not extend based on a selected maximum or tune from partial results.')
    if not complete:
        return summary
    indexed = {(r['seed'], r['tactical']): r for r in results}
    differences = np.array([indexed[s, True]['score'] - indexed[s, False]['score'] for s in seeds])
    bootstrap = np.random.default_rng(60612).choice(differences, (20000, len(seeds)), replace=True).mean(1)
    summary['paired_difference'] = dict(mean=float(differences.mean()),
        bootstrap95=np.quantile(bootstrap, [.025, .975]).tolist(),
        wins=int((differences > 0).sum()), ties=int((differences == 0).sum()),
        per_seed=differences.tolist())
    for tactical, name in ((False, 'control'), (True, 'tactical')):
        arm = [indexed[s, tactical] for s in seeds]
        summary['arms'][name] = dict(mean_score=float(np.mean([r['score'] for r in arm])),
            best_score=max(r['score'] for r in arm), mean_seconds=float(np.mean([r['elapsed_seconds'] for r in arm])),
            reaching_32768=sum(r['max_tile'] >= 32768 for r in arm),
            reaching_65536=sum(r['max_tile'] >= 65536 for r in arm),
            transitions=sum(r['length'] for r in arm),
            tactical_checks=sum(r['tactical_checks'] for r in arm),
            tactical_overrides=sum(r['tactical_overrides'] for r in arm),
            tactical_budget_exhaustions=sum(r['tactical_budget_exhaustions'] for r in arm))
    return summary


def main():
    if STOP.exists(): raise RuntimeError('User STOP is present')
    previous = BASE / 'hybrid_validation100'
    status = json.loads((previous / 'status.json').read_text())
    checked = json.loads((previous / 'verified.json').read_text())
    assert status['phase'] == 'complete' and all(checked.values())
    assert json.loads((previous / 'summary.json').read_text())['complete']
    OUT.mkdir(parents=True, exist_ok=False)
    hashes, cache_hashes = fingerprints(), cache_fingerprints()
    write(OUT / 'protocol.json', dict(seeds=SEEDS, planned_games=24, workers=16,
        per_game_seconds=7200, per_game_rss_gib=3., fingerprints=hashes, cache_sha256=cache_hashes,
        hypothesis='A solved restricted formation can conflict with near-term raw points. '
            'Test whether selective exact eight-move checks improve full-game outcomes. '
            'Keep profitable risky choices when their finite-horizon score is better.',
        single_intervention='Only the tactical check. Original late-depth code, table thresholds, '
            'raw reward, search time scale and all native binaries remain unchanged.',
        decision='Evaluate all 12 pairs before deciding. Report paired uncertainty, tile rates, '
            'extra computation, intervention counts and audited replays. A short horizon can '
            'damage a long strategy; no automatic promotion or fixed-loop extension.',
        prerequisite='Frozen100evaluation completed with source/cache/runner integrity verified.'))
    queue = iter((s, t) for s in SEEDS for t in (False, True))
    results, active = [], {}
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=16) as pool:
        def submit():
            if STOP.exists(): return
            job = next(queue, None)
            if job is None: return
            seed, tactical = job
            name = ('tactical' if tactical else 'control') + f'_seed{seed}'
            active[pool.submit(run_game, OUT / name, seed, tactical)] = job
        for _ in range(16): submit()
        while active:
            finished, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
            for future in finished:
                seed, tactical = active.pop(future)
                try: result = future.result()
                except Exception as error:
                    result = dict(seed=seed, tactical=tactical, complete=False, error=repr(error))
                results.append(result)
                submit()
            summary = summarize(results)
            summary['elapsed_wall_seconds'] = time.perf_counter() - started
            write(OUT / 'games.json', results)
            write(OUT / 'summary.json', summary)
            write(OUT / 'status.json', dict(phase='running', finished=len(results),
                  active=[dict(seed=s, tactical=t) for s, t in active.values()]))
    verified = dict(policy_unchanged=fingerprints() == hashes,
                    cache_unchanged=cache_fingerprints() == cache_hashes)
    write(OUT / 'verified.json', verified)
    if not all(verified.values()):
        summary.update(complete=False, arms={}, paired_difference=None,
                       validation_error='Frozen source/cache integrity check failed')
        write(OUT / 'summary.json', summary)
    write(OUT / 'status.json', dict(phase='complete' if summary['complete'] else 'incomplete',
                                   finished=len(results), active=[]))


if __name__ == '__main__':
    main()
