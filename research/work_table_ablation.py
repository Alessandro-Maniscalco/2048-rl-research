"""Isolate the old formation tables under one repeatable search-work rule."""
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
from pathlib import Path
import resource
import time

import numpy as np

from research.afterstate_teacher import digest, write
from research.endgame_audit import audit
from research.endgame_continue import restore_environment
from research.hybrid_endgame_study import BASE, ROOT, STOP, cache_fingerprints
from research.work_budget_hybrid import WorkBudgetHybridAgent
from research.work_budget_study import fingerprints as base_fingerprints
from rl2048.agents.endgame import EndgameAgent
from rl2048.game import ACTION_NAMES, Game2048
from rl2048.view import COLORS

OUT = BASE / 'work_table_ablation'
WORK_TARGET = 1000000


class WorkTableAblationAgent(WorkBudgetHybridAgent):
    def __init__(self, use_tables):
        super().__init__(WORK_TARGET)
        self.use_tables = use_tables
        self.display_name = ('Frozen formations + compact search' if use_tables
                             else 'Compact search without old formation tables')

    def act(self, board, action_mask):
        if self.use_tables:
            return super().act(board, action_mask)
        # Keep the exact same native search instance and complete-layer controller.
        # The candidate bypasses only the older formation lookup. Its embedded
        # compact tables and their existing verification search remain enabled.
        self.logic.work_calls = []
        action = EndgameAgent.act(self, board, action_mask)
        self.last_decision.update(table_queries=0, table_hit=False,
            work_target=self.work_target, work_calls=self.logic.work_calls,
            summed_iterative_layer_nodes=sum(c['summed_layer_nodes'] for c in self.logic.work_calls))
        return action


def fingerprints():
    return dict(base_fingerprints(), **{'research/work_table_ablation.py': digest(Path(__file__))})


def play(folder, seed, use_tables, move_cap, seconds):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    hashes = fingerprints()
    write(folder / 'protocol.json', dict(seed=seed, use_tables=use_tables,
        work_target=WORK_TARGET, move_cap=move_cap, seconds_budget=seconds,
        rss_gib_budget=3., fingerprints=hashes,
        difference='Only the older frozen10/11 formation lookup is enabled or bypassed. '
            'Both retain compact embedded tables, heuristic and native search.',
        allocation='Identical one-million completed-layer work target, soft not hard cap. '
            'Start depth3 up to the same upstream maximum. Timing does not select depth.',
        game='Exact Game2048, seeded two-tile starts, raw merge points, no undo or edited tiles. '
            'Capped verification runs are separate from the full-game screen.'))
    agent = WorkTableAblationAgent(use_tables)
    env = Game2048()
    board, info = env.reset(seed=seed)
    frames = [dict(board=board.tolist(), score=0, action=None, reward=0)]
    decisions = []
    done, reason, peak = False, None, 0.
    start = time.perf_counter()
    try:
        while not done:
            peak = max(peak, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3)
            if STOP.exists(): reason = 'stop_file'; break
            if len(decisions) >= move_cap: reason = 'move_cap'; break
            if time.perf_counter()-start >= seconds: reason = 'time_budget'; break
            if peak > 3.: reason = 'memory_budget'; break
            try:
                action = agent.act(board, info['action_mask'])
            except Exception as error:
                reason = repr(error); break
            board, reward, done, truncated, info = env.step(action)
            assert not truncated
            decisions.append(agent.last_decision)
            frames.append(dict(board=board.tolist(), score=env.score, action=action, reward=reward))
            if len(decisions) % 128 == 0 or done:
                write(folder / 'status.json', dict(phase='running', score=env.score,
                    moves=len(decisions), max_tile=int(board.max()), elapsed_seconds=time.perf_counter()-start))
    finally:
        agent.tables.close()
    result = dict(seed=seed, use_tables=use_tables, work_target=WORK_TARGET,
        score=env.score, length=len(decisions), max_tile=int(board.max()),
        complete=done, terminated=done, truncated=not done, stop_reason=reason,
        elapsed_seconds=time.perf_counter()-start, peak_rss_gib=peak,
        modes=dict(Counter(d['mode'] for d in decisions)),
        summed_iterative_layer_nodes=sum(d['summed_iterative_layer_nodes'] for d in decisions),
        source_unchanged=fingerprints() == hashes)
    replay = dict(algorithm=agent.display_name+' (one-million work target)',
        result=result, frames=frames, actions=ACTION_NAMES, colors=COLORS)
    write(folder / 'replay.json', replay)
    write(folder / 'decisions.json', decisions)
    write(folder / 'result.json', result)
    (folder / 'replay.html').write_text((ROOT / 'rl2048/replay.html').read_text().replace(
        '__REPLAY_DATA__', json.dumps(replay).replace('<', '\\u003c')))
    restored, _, _, restored_done = restore_environment(replay)
    assert restored.score == env.score and restored_done == done
    result['exact_seeded_replay_audited'] = True
    if done:
        audit(folder)
    write(folder / 'result.json', result)
    write(folder / 'status.json', dict(phase='complete' if done else 'incomplete', **result))
    return result


def decision_signature(d):
    # Timing and ignored proposed depth are not part of the executed search.
    return dict(action=d['action'], mode=d['mode'], depth=d['depth'],
        search_board_hex=d.get('search_board_hex'), search_scores=d.get('search_scores'),
        work_layers=[c['layers'] for c in d['work_calls']])


def compare_pair(with_folder, without_folder):
    frames = [json.loads((p / 'replay.json').read_text())['frames'] for p in (with_folder, without_folder)]
    decisions = [json.loads((p / 'decisions.json').read_text()) for p in (with_folder, without_folder)]
    intervention = next((i for i, d in enumerate(decisions[0]) if d['mode'].startswith('frozen-')), len(decisions[0]))
    shared = frames[0][:intervention+1] == frames[1][:intervention+1]
    shared = shared and ([decision_signature(d) for d in decisions[0][:intervention]] ==
                         [decision_signature(d) for d in decisions[1][:intervention]])
    divergence = next((i for i, (a, b) in enumerate(zip(decisions[0], decisions[1]))
                       if a['action'] != b['action']), None)
    return dict(identical_before_first_table_intervention=shared,
        first_table_move=intervention+1 if intervention < len(decisions[0]) else None,
        first_action_difference_move=None if divergence is None else divergence+1)


def run_batch(folder, jobs, workers, move_cap, seconds):
    folder.mkdir(parents=True, exist_ok=False)
    hashes, caches = fingerprints(), cache_fingerprints()
    write(folder / 'protocol.json', dict(jobs=jobs, workers=workers, move_cap=move_cap,
        per_game_seconds=seconds, fingerprints=hashes, cache_sha256=caches,
        work_target=WORK_TARGET, hypothesis='Do older solved formation tables improve playing strength '
            'when their compact fallback has repeatable search work? The earlier timed eight-pair '
            'comparison found a runtime advantage but no established score gain; timing is a confound.',
        scope='Keep compact embedded tables in both arms. One factor: older10/11 lookup dispatch. '
            'Same search-work rule when search runs, not equal total runtime or equal total nodes.',
        limits='All attempts retained. No partial means. No policy changes from partial results. '
            'Full-game screen is gated on paired-prefix and duplicate-run verification.'))
    active, results, queue = {}, [], iter(jobs)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        def submit():
            if STOP.exists(): return
            job = next(queue, None)
            if job is not None:
                future = pool.submit(play, folder/job['name'], job['seed'], job['use_tables'], move_cap, seconds)
                active[future] = job
        for _ in range(workers): submit()
        while active:
            completed, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
            for future in completed:
                job = active.pop(future)
                try: result = future.result()
                except Exception as error: result = dict(complete=False, error=repr(error))
                results.append(dict(**result, job=job))
                submit()
            write(folder / 'games.json', results)
            write(folder / 'status.json', dict(phase='running', finished=len(results), planned=len(jobs),
                active=list(active.values()), elapsed_wall_seconds=time.perf_counter()-started))
    verified = dict(policy_unchanged=fingerprints() == hashes, cache_unchanged=cache_fingerprints() == caches)
    write(folder / 'verified.json', verified)
    return results, verified


def verify_repeatability():
    out = OUT / 'verification'
    jobs = [dict(name=f'seed{s}_tables{int(t)}_rep{r}', seed=s, use_tables=t, replicate=r)
            for s in (8981998, 8981999) for t in (True, False) for r in (0, 1)]
    results, verified = run_batch(out, jobs, 8, 4096, 900)
    valid = len(results) == 8 and all(verified.values()) and all(
        r.get('exact_seeded_replay_audited') and r.get('source_unchanged') and
        (r.get('complete') or r.get('stop_reason') == 'move_cap') for r in results)
    duplicates, pairs = [], []
    if valid:
        for seed in (8981998, 8981999):
            for tables in (True, False):
                folders = [out / f'seed{seed}_tables{int(tables)}_rep{r}' for r in (0, 1)]
                replays = [json.loads((f/'replay.json').read_text()) for f in folders]
                ds = [json.loads((f/'decisions.json').read_text()) for f in folders]
                duplicates.append(dict(seed=seed, use_tables=tables,
                    same_frames=replays[0]['frames'] == replays[1]['frames'],
                    same_decisions=[decision_signature(d) for d in ds[0]] == [decision_signature(d) for d in ds[1]]))
            pairs.append(dict(seed=seed, **compare_pair(out/f'seed{seed}_tables1_rep0',out/f'seed{seed}_tables0_rep0')))
        valid = all(d['same_frames'] and d['same_decisions'] for d in duplicates) and all(
            p['identical_before_first_table_intervention'] for p in pairs)
    summary = dict(complete=valid, duplicates=duplicates, pairs=pairs,
        note='Four duplicated4096-move capped trajectories and two intervention-prefix checks. '
            'Not a full-game performance benchmark or proof for every board.')
    write(out/'summary.json', summary)
    write(out/'status.json', dict(phase='complete' if valid else 'failed', finished=len(results), active=[]))
    return valid


def screen():
    out = OUT / 'screen'
    seeds = list(range(8982000, 8982016))
    jobs = [dict(name=f'seed{s}_tables{int(t)}', seed=s, use_tables=t)
            for s in seeds for t in (True, False)]
    results, verified = run_batch(out, jobs, 16, 100000, 3600)
    complete = len(results) == 32 and all(verified.values()) and all(
        r.get('complete') and r.get('exact_seeded_replay_audited') and r.get('source_unchanged') for r in results)
    summary = dict(complete=complete, finished=len(results), planned=32, arms={}, paired_difference=None)
    pairs = []
    if complete:
        pairs = [dict(seed=s, **compare_pair(out/f'seed{s}_tables1',out/f'seed{s}_tables0')) for s in seeds]
        complete = all(p['identical_before_first_table_intervention'] for p in pairs)
        summary['complete'] = complete
    if complete:
        indexed = {(r['seed'], r['use_tables']): r for r in results}
        ds = np.array([indexed[s,False]['score']-indexed[s,True]['score'] for s in seeds])
        bootstrap = np.random.default_rng(9209).choice(ds,(20000,len(seeds)),replace=True).mean(1)
        summary['paired_difference'] = dict(no_old_tables_minus_tables=float(ds.mean()),
            bootstrap95=np.quantile(bootstrap,[.025,.975]).tolist(), per_seed=ds.tolist(),
            wins=int((ds>0).sum()), ties=int((ds==0).sum()))
        for tables, name in ((True,'with_old_tables'),(False,'without_old_tables')):
            arm = [indexed[s,tables] for s in seeds]
            summary['arms'][name] = dict(mean_score=float(np.mean([r['score'] for r in arm])),
                best_score=max(r['score'] for r in arm),mean_seconds=float(np.mean([r['elapsed_seconds'] for r in arm])),
                reaching_32768=sum(r['max_tile']>=32768 for r in arm),reaching_65536=sum(r['max_tile']>=65536 for r in arm))
    summary.update(pairs=pairs, note='Sixteen fresh pairs; only older formation dispatch changes. '
        'Both use the same one-million soft search-work target and keep compact embedded tables. '
        'No partial means or automatic promotion; this screen does not replace100-game benchmarks.')
    write(out/'summary.json', summary)
    write(out/'status.json', dict(phase='complete' if complete else 'incomplete',finished=len(results),active=[]))


def main():
    if STOP.exists(): raise RuntimeError('User STOP is present')
    assert json.loads((BASE/'late_game_comparison/summary.json').read_text())['complete']
    assert all(json.loads((BASE/'late_game_comparison/verified.json').read_text()).values())
    if verify_repeatability() and not STOP.exists():
        screen()


if __name__ == '__main__':
    main()
