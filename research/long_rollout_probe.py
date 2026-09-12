"""Estimate 512-move raw returns from two recorded decisions, using a base policy.

This is an offline conditional diagnostic. Each first move is forced, then the
existing one-million-work hybrid plays. Finite-horizon Monte Carlo values are
not optimal Q values, full-game returns, or a newly evaluated online player.
"""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
from pathlib import Path
import resource
import time

import numpy as np

from research.afterstate_teacher import digest, write
from research.hybrid_endgame_study import BASE, ROOT, STOP, cache_fingerprints
from research.late_game_study import initialize
from research.work_budget_hybrid import WorkBudgetHybridAgent
from research.work_budget_study import fingerprints as base_fingerprints
from rl2048.game import ACTION_NAMES, legal_actions

OUT = BASE/'long_rollout_probe'
SOURCE = BASE/'work_table_ablation/screen/seed8982012_tables1'
HORIZON = 512


def fingerprints():
    return dict(base_fingerprints(), **{'research/long_rollout_probe.py':digest(Path(__file__)),
        'research/late_game_study.py':digest(ROOT/'research/late_game_study.py')})


def run(job, seconds):
    out=OUT/job['name'];out.mkdir(parents=True,exist_ok=False)
    write(out/'protocol.json',dict(**job,horizon=HORIZON,seconds_budget=seconds,
        conditional=True,excluded_from_full_game_rankings=True))
    start=time.perf_counter()
    agent=WorkBudgetHybridAgent(1000000)
    env,board,info=initialize(job['case'],job['future_seed'])
    frames=[dict(board=board.tolist(),score=env.score,action=None,reward=0)]
    decisions=[];done=False;reason=None
    try:
        for step in range(HORIZON):
            if STOP.exists():reason='stop_file';break
            if time.perf_counter()-start>=seconds:reason='time_budget';break
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>3*1024**3:reason='memory_budget';break
            action=job['first_action'] if step==0 else agent.act(board,info['action_mask'])
            assert info['action_mask'][action]
            if step==0:decision=dict(action=action,mode='forced_first_action')
            else:decision={k:agent.last_decision.get(k) for k in ('action','mode','depth','nodes')}
            board,reward,done,truncated,info=env.step(action)
            assert not truncated
            frames.append(dict(board=board.tolist(),score=env.score,action=action,reward=reward))
            decisions.append(decision)
            if done:break
    finally:agent.tables.close()
    valid=done or len(decisions)==HORIZON
    result=dict(case_id=job['case']['id'],future_seed=job['future_seed'],first_action=job['first_action'],
        horizon=HORIZON,valid_horizon=valid,terminated=done,horizon_capped=not done and valid,
        stop_reason=reason,additional_points=env.score-job['case']['score'],length=len(decisions),
        max_tile=int(board.max()),new_larger_tile=bool(board.max()>np.max(job['case']['board'])),
        elapsed_seconds=time.perf_counter()-start,conditional=True,excluded_from_full_game_rankings=True)
    check,_,_=initialize(job['case'],job['future_seed'])
    for frame in frames[1:]:
        actual,reward,terminal,_,_=check.step(frame['action'])
        assert np.array_equal(actual,frame['board']) and reward==frame['reward'] and check.score==frame['score']
    assert check.score==env.score and check._terminated==done
    result['exact_seeded_transitions_audited']=True
    write(out/'trajectory.json',dict(result=result,frames=frames,decisions=decisions))
    write(out/'result.json',result)
    return result


def summarize(results,jobs,cases):
    expected={(j['case']['id'],j['future_seed'],j['first_action']) for j in jobs}
    complete=(len(results)==len(expected) and
        {(r.get('case_id'),r.get('future_seed'),r.get('first_action')) for r in results}==expected and
        all(r.get('valid_horizon') and r.get('exact_seeded_transitions_audited') for r in results))
    output=dict(complete=complete,finished=len(results),planned=len(jobs),cases=[],
        conditional=True,excluded_from_full_game_rankings=True,
        equation='Qhat_H^pi(s,a) = mean_i sum_{t=0}^{H-1} r[i,t+1], terminal padding=0, H=512',
        note='Sixteen future streams per legal action and two selected fixed boards. '
            'Fresh controller per rollout. Future policy is the existing hybrid, not optimal play. '
            'Zero leaf value at512 excludes all later rewards. Horizon-capped paths are valid512-step '
            'samples; time/memory failures are not. This is not a full-game score or policy-improvement guarantee.')
    if not complete:return output
    indexed={(r['case_id'],r['future_seed'],r['first_action']):r for r in results}
    for case in cases:
        rows=[];seeds=sorted({j['future_seed'] for j in jobs if j['case']['id']==case['id']})
        ref=case['recorded_action']
        for action in np.flatnonzero(legal_actions(np.asarray(case['board']))):
            action=int(action);rs=[indexed[case['id'],s,action] for s in seeds]
            vals=np.array([r['additional_points'] for r in rs],dtype=float)
            diffs=vals-np.array([indexed[case['id'],s,ref]['additional_points'] for s in seeds])
            boot=np.random.default_rng(1123+case['id']*4+action).choice(diffs,(20000,len(seeds)),replace=True).mean(1)
            rows.append(dict(action=action,name=ACTION_NAMES[action],mean_points=float(vals.mean()),
                sample_sd=float(vals.std(ddof=1)),mean_steps=float(np.mean([r['length'] for r in rs])),
                terminated=sum(r['terminated'] for r in rs),horizon_capped=sum(r['horizon_capped'] for r in rs),
                new_larger_tile=sum(r['new_larger_tile'] for r in rs),samples=len(rs),
                paired_difference_from_recorded=float(diffs.mean()),paired_bootstrap95=np.quantile(boot,[.025,.975]).tolist()))
        output['cases'].append(dict(case_id=case['id'],source_move=case['source_move'],
            recorded_action=ACTION_NAMES[ref],actions=rows,
            largest_sample_mean_action=max(rows,key=lambda r:r['mean_points'])['name'],
            caution='Exploratory unadjusted pairwise intervals; selecting the largest sample mean can be optimistic. '
                'A fresh validation sample is required before using a suggested change.'))
    return output


def main():
    if STOP.exists():raise RuntimeError('User STOP is present')
    h=json.loads((SOURCE/'high_tile_audit.json').read_text())
    assert h['exact_seed_audit_present'] and h['policy_unchanged']
    replay=json.loads((SOURCE/'replay.json').read_text())
    assert digest(SOURCE/'replay.json')==h['replay_sha256']
    decisions=json.loads((SOURCE/'decisions.json').read_text())
    indices=[29719,44665]
    cases=[dict(id=i,source=str(SOURCE.relative_to(BASE)),source_move=k+1,
        board=replay['frames'][k]['board'],score=replay['frames'][k]['score'],
        recorded_action=decisions[k]['action']) for i,k in enumerate(indices)]
    jobs=[dict(name=f'case{c["id"]}_action{a}_seed{s}',case=c,first_action=int(a),future_seed=s)
          for c in cases for s in range(8983000+100*c['id'],8983016+100*c['id'])
          for a in np.flatnonzero(legal_actions(np.asarray(c['board'])))]
    OUT.mkdir(exist_ok=False)
    hashes,caches=fingerprints(),cache_fingerprints()
    write(OUT/'protocol.json',dict(cases=cases,jobs=jobs,horizon=HORIZON,workers=8,
        per_rollout_seconds=120,total_seconds=900,fingerprints=hashes,cache_sha256=caches,
        source_replay_sha256=h['replay_sha256'],
        hypothesis='Actual512-step returns under a strong base player may reveal delayed large '
            'merges that an exact eight-move reward calculation cannot see.',
        policy='Force one legal first move, then fresh-controller one-million-work hybrid. '
            'Same16future RNG streams across actions within each board; the two boards have separate streams.',
        equation='Qhat_512^pi(s,a)=mean of undiscounted raw merge rewards over at most512moves including the forced move.',
        limits='Offline diagnostic on two selected recorded boards. No online policy intervention, '
            'full-game score claim, learned Q network, terminal heuristic or guarantee of improvement. '
            'Judge all128samples; incomplete resource-limited attempts prevent aggregate means.',
        reference='https://ocw.mit.edu/courses/6-231-dynamic-programming-and-stochastic-control-fall-2015/resources/mit6_231f15_lec9/'))
    active={};results=[];queue=iter(jobs);deadline=time.perf_counter()+900
    with ProcessPoolExecutor(max_workers=8) as pool:
        def submit():
            if STOP.exists() or time.perf_counter()>=deadline:return
            job=next(queue,None)
            if job is not None:active[pool.submit(run,job,max(1,min(120,deadline-time.perf_counter())))]=job
        for _ in range(8):submit()
        while active:
            done,_=wait(active,timeout=10,return_when=FIRST_COMPLETED)
            for future in done:
                job=active.pop(future)
                try:r=future.result()
                except Exception as error:r=dict(case_id=job['case']['id'],future_seed=job['future_seed'],
                    first_action=job['first_action'],valid_horizon=False,error=repr(error))
                results.append(r);submit()
            write(OUT/'games.json',results)
            write(OUT/'status.json',dict(phase='running',finished=len(results),planned=len(jobs),active=len(active)))
    # Explicitly retain every unstarted attempt when the bounded wall budget expires.
    for job in queue:
        results.append(dict(case_id=job['case']['id'],future_seed=job['future_seed'],
            first_action=job['first_action'],valid_horizon=False,stop_reason='not_started_stop_or_wall_budget'))
    verified=dict(policy_unchanged=fingerprints()==hashes,cache_unchanged=cache_fingerprints()==caches)
    write(OUT/'verified.json',verified)
    summary=summarize(results,jobs,cases)
    if not all(verified.values()):summary.update(complete=False,cases=[],validation_error='Integrity check failed')
    write(OUT/'games.json',results);write(OUT/'summary.json',summary)
    write(OUT/'status.json',dict(phase='complete' if summary['complete'] else 'incomplete',finished=len(results),planned=len(jobs),active=0))


if __name__=='__main__':main()
