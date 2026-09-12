"""Benchmark repeatability first, then compare two completed-layer work targets.

The repeatability assay uses capped 4,096-move trajectories; it is not a score
benchmark. Full-game screening is a separate command, gated on that assay.
"""
import argparse
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
from research.hybrid_endgame_study import BASE, ROOT, STOP, cache_fingerprints, policy_fingerprints
from research.work_budget_hybrid import WorkBudgetHybridAgent
from rl2048.game import ACTION_NAMES, Game2048
from rl2048.view import COLORS


def fingerprints():
    paths = ['research/work_budget_hybrid.py', 'research/work_budget_study.py', 'rl2048/game.py']
    return dict(policy_fingerprints(), **{p:digest(ROOT/p) for p in paths})


def run_game(folder, seed, work_target, move_cap=100000, seconds=7200):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    hashes = fingerprints()
    write(folder/'protocol.json', dict(seed=seed, work_target=work_target,
        move_cap=move_cap, seconds_budget=seconds, rss_gib_budget=3., fingerprints=hashes,
        policy='Read-only solved formations plus compact search with completed-layer work target.',
        allocation='Begin each iterative call at depth3 or a smaller configured maximum; complete '
            'successive layers until the last native node count reaches the target or the upstream '
            'maximum depth is reached. No wall-clock depth selection. Last layer can overshoot. '
            'Fixed-depth embedded-table verification remains. Node counts are engine work counters, '
            'not hardware instructions; iterative-layer sums exclude direct table verification.',
        game='Exact Python Game2048, standard two-tile starts, raw scores and 90/10 spawns. '
            'No undo, chosen restart or artificial tile changes.',
        intervention='No tactical override, late-depth correction, changed table thresholds, '
            'new heuristic or neural update. Fresh search state at the start of each game.'))
    agent = WorkBudgetHybridAgent(work_target)
    env = Game2048()
    board, info = env.reset(seed=seed)
    frames = [dict(board=board.tolist(), score=0, action=None, reward=0)]
    decisions = []
    started = time.perf_counter()
    peak = 0.
    done = False
    reason = None
    try:
        while not done:
            peak = max(peak, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3)
            if STOP.exists(): reason='stop_file'; break
            if len(decisions) >= move_cap: reason='move_cap'; break
            if time.perf_counter()-started >= seconds: reason='time_budget'; break
            if peak > 3.: reason='memory_budget'; break
            try:
                action = agent.act(board, info['action_mask'])
            except Exception as error:
                reason=repr(error); break
            board, reward, done, truncated, info = env.step(action)
            assert not truncated
            decisions.append(agent.last_decision)
            frames.append(dict(board=board.tolist(), score=env.score, action=action, reward=reward))
            if len(decisions)%128==0 or done:
                write(folder/'status.json', dict(phase='complete' if done else 'running', seed=seed,
                    work_target=work_target, score=env.score, moves=len(decisions), max_tile=int(board.max()),
                    elapsed_seconds=time.perf_counter()-started))
    finally:
        agent.tables.close()
    durations = [d['seconds'] for d in decisions]
    result = dict(seed=seed, work_target=work_target, score=env.score, length=len(decisions),
        max_tile=int(board.max()), complete=done, terminated=done, truncated=not done,
        stop_reason=reason, elapsed_seconds=time.perf_counter()-started, peak_rss_gib=peak,
        decision_seconds_p95=float(np.quantile(durations,.95)) if durations else None,
        decision_seconds_max=max(durations,default=None),
        modes=dict(Counter(d['mode'] for d in decisions)),
        summed_iterative_layer_nodes=sum(d['summed_iterative_layer_nodes'] for d in decisions),
        planner_timing_state_reset=False, source_and_binary_unchanged=fingerprints()==hashes)
    replay = dict(algorithm=agent.display_name+f' ({work_target:,})', result=result,
                  frames=frames, actions=ACTION_NAMES, colors=COLORS)
    write(folder/'replay.json', replay)
    write(folder/'decisions.json', decisions)
    (folder/'replay.html').write_text((ROOT/'rl2048/replay.html').read_text().replace(
        '__REPLAY_DATA__',json.dumps(replay).replace('<','\\u003c')))
    write(folder/'result.json',result)
    restored,_,_,restored_done = restore_environment(replay)
    assert restored.score == env.score and restored_done == done
    result['exact_seeded_replay_audited'] = True
    if done: audit(folder)
    write(folder/'result.json',result)
    write(folder/'status.json',dict(phase='complete' if done else 'incomplete',**result))
    return result


def run_batch(out, jobs, move_cap, seconds):
    out.mkdir(parents=True,exist_ok=False)
    hashes, caches = fingerprints(),cache_fingerprints()
    write(out/'protocol.json', dict(jobs=jobs, workers=min(16,len(jobs)), move_cap=move_cap,
        per_game_seconds=seconds, fingerprints=hashes, cache_sha256=caches,
        reason='Search timing caused every prior tactical pair to diverge before the intervention. '
            'Test repeatable completed-layer allocation, then a bounded1M-versus4M work comparison. '
            'No automatic promotion; distinguish reproducibility from playing strength.'))
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=min(16,len(jobs))) as pool:
        active = {pool.submit(run_game,out/j['name'],j['seed'],j['work_target'],move_cap,seconds):j
                  for j in jobs if not STOP.exists()}
        while active:
            done,_ = wait(active,timeout=10,return_when=FIRST_COMPLETED)
            for future in done:
                job = active.pop(future)
                try: result=future.result()
                except Exception as error: result=dict(complete=False,error=repr(error))
                results.append(dict(**result,job=job))
            write(out/'games.json',results)
            write(out/'status.json',dict(phase='running',finished=len(results),planned=len(jobs),
                active=list(active.values()),elapsed_wall_seconds=time.perf_counter()-started))
    verified = dict(policy_unchanged=fingerprints()==hashes,cache_unchanged=cache_fingerprints()==caches)
    write(out/'verified.json',verified)
    return results,verified,time.perf_counter()-started


def repeatability():
    out=BASE/'work_budget_repeatability'
    jobs=[dict(name=f'budget{b}_seed{s}_rep{rep}',seed=s,work_target=b,replicate=rep)
          for s in (8977998,8977999) for b in (1000000,4000000) for rep in (0,1)]
    results,verified,elapsed=run_batch(out,jobs,4096,900)
    valid=(len(results)==len(jobs) and all(verified.values()) and all(
        r.get('exact_seeded_replay_audited') and r.get('source_and_binary_unchanged') and
        (r.get('complete') or r.get('stop_reason')=='move_cap') for r in results))
    pairs=[]
    if valid:
        for seed in (8977998,8977999):
            for budget in (1000000,4000000):
                folders=[out/f'budget{budget}_seed{seed}_rep{rep}' for rep in (0,1)]
                replays=[json.loads((f/'replay.json').read_text()) for f in folders]
                decisions=[json.loads((f/'decisions.json').read_text()) for f in folders]
                # Exclude only timing fields, which should vary with scheduling.
                keys=('action','mode','depth','work_calls','search_board_hex')
                same_decisions=([{k:d.get(k) for k in keys} for d in decisions[0]] ==
                                [{k:d.get(k) for k in keys} for d in decisions[1]])
                same_frames=replays[0]['frames']==replays[1]['frames']
                pairs.append(dict(seed=seed,work_target=budget,identical_frames=same_frames,
                    identical_actions_depths_work=same_decisions,moves=len(decisions[0])))
        valid=all(p['identical_frames'] and p['identical_actions_depths_work'] for p in pairs)
    write(out/'summary.json',dict(complete=valid,pairs=pairs,elapsed_wall_seconds=elapsed,
        note='Four duplicated capped trajectories verify repeatability on these two seeds and budgets. '
            'This is not a full-game performance estimate or a proof for all boards.'))
    write(out/'status.json',dict(phase='complete' if valid else 'failed',finished=len(results),active=[]))


def screen():
    prior=BASE/'work_budget_repeatability'
    assert json.loads((prior/'summary.json').read_text())['complete']
    assert all(json.loads((prior/'verified.json').read_text()).values())
    out=BASE/'work_budget_comparison'
    seeds=list(range(8978000,8978008))
    jobs=[dict(name=f'budget{b}_seed{s}',seed=s,work_target=b)
          for s in seeds for b in (1000000,4000000)]
    results,verified,elapsed=run_batch(out,jobs,100000,7200)
    complete=(len(results)==len(jobs) and all(verified.values()) and all(
        r.get('complete') and r.get('exact_seeded_replay_audited') and
        r.get('source_and_binary_unchanged') for r in results))
    summary=dict(complete=complete,finished=len(results),planned=16,arms={},paired_difference=None,
        elapsed_wall_seconds=elapsed,note='Eight fresh pairs compare work targets, not wall-clock fairness. '
            'No partial means; check both endpoints and uncertainty. No tactical intervention.')
    if complete:
        indexed={(r['seed'],r['work_target']):r for r in results}
        differences=np.array([indexed[s,4000000]['score']-indexed[s,1000000]['score'] for s in seeds])
        bootstrap=np.random.default_rng(9178).choice(differences,(20000,len(seeds)),replace=True).mean(1)
        summary['paired_difference']=dict(mean=float(differences.mean()),
            bootstrap95=np.quantile(bootstrap,[.025,.975]).tolist(),per_seed=differences.tolist(),
            wins=int((differences>0).sum()),ties=int((differences==0).sum()))
        for budget in (1000000,4000000):
            arm=[indexed[s,budget] for s in seeds]
            summary['arms'][str(budget)]=dict(mean_score=float(np.mean([r['score'] for r in arm])),
                best_score=max(r['score'] for r in arm),mean_seconds=float(np.mean([r['elapsed_seconds'] for r in arm])),
                reaching_32768=sum(r['max_tile']>=32768 for r in arm),
                reaching_65536=sum(r['max_tile']>=65536 for r in arm))
    write(out/'summary.json',summary)
    write(out/'status.json',dict(phase='complete' if complete else 'incomplete',finished=len(results),active=[]))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['repeatability','screen'])
    args=parser.parse_args()
    if STOP.exists(): raise RuntimeError('User STOP is present')
    (repeatability if args.mode=='repeatability' else screen)()
