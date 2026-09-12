"""Choose actions from discovery rollouts, then test choices on fresh streams.

Six other naturally reached late-game boards. This evaluates conditional
512-step returns, not a new online policy's complete-game strength.
"""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from research.afterstate_teacher import digest, write
from research.hybrid_endgame_study import BASE, ROOT, STOP, cache_fingerprints
import research.long_rollout_probe as probe
from rl2048.game import ACTION_NAMES, legal_actions

OUT=BASE/'rollout_choice_study'


def fingerprints():
    return dict(probe.fingerprints(),**{'research/rollout_choice_study.py':digest(Path(__file__))})


def run(job,seconds,folder):
    # Only output location differs from the frozen simulation implementation.
    with patch.object(probe,'OUT',Path(folder)):
        return probe.run(job,seconds)


def batch(phase,cases,count,offset):
    out=OUT/phase;out.mkdir(exist_ok=False)
    jobs=[dict(name=f'case{c["id"]}_action{a}_seed{s}',case=c,first_action=int(a),future_seed=s)
          for c in cases for s in range(9010000+1000*c['id']+offset,9010000+1000*c['id']+offset+count)
          for a in np.flatnonzero(legal_actions(np.asarray(c['board'])))]
    write(out/'protocol.json',dict(cases=cases,jobs=jobs,streams_per_action=count,horizon=512,
        workers=16,per_rollout_seconds=120,phase_seconds=900,conditional=True,
        excluded_from_full_game_rankings=True,phase=phase))
    active={};results=[];queue=iter(jobs);deadline=time.perf_counter()+900
    with ProcessPoolExecutor(max_workers=16) as pool:
        def submit():
            if STOP.exists() or time.perf_counter()>=deadline:return
            job=next(queue,None)
            if job is not None:active[pool.submit(run,job,max(1,min(120,deadline-time.perf_counter())),out)]=job
        for _ in range(16):submit()
        while active:
            done,_=wait(active,timeout=10,return_when=FIRST_COMPLETED)
            for future in done:
                job=active.pop(future)
                try:r=future.result()
                except Exception as error:r=dict(case_id=job['case']['id'],future_seed=job['future_seed'],
                    first_action=job['first_action'],valid_horizon=False,error=repr(error))
                results.append(r);submit()
            write(out/'games.json',results)
            write(out/'status.json',dict(phase='running',finished=len(results),planned=len(jobs),active=len(active)))
            write(OUT/'status.json',dict(phase=phase,finished=len(results),planned=len(jobs),active=len(active)))
    for job in queue:
        results.append(dict(case_id=job['case']['id'],future_seed=job['future_seed'],
            first_action=job['first_action'],valid_horizon=False,stop_reason='not_started_stop_or_wall_budget'))
    summary=probe.summarize(results,jobs,cases)
    summary['note']=summary['note'].replace('Sixteen future streams',f'{count} future streams').replace('two selected fixed boards',f'{len(cases)} selected fixed boards')
    write(out/'games.json',results);write(out/'summary.json',summary)
    write(out/'status.json',dict(phase='complete' if summary['complete'] else 'incomplete',finished=len(results),planned=len(jobs),active=0))
    return summary,results


def choose(cases):
    choices=[]
    for c in cases:
        means={}
        legal=[int(a) for a in np.flatnonzero(legal_actions(np.asarray(c['board'])))]
        for a in legal:
            pairs=[]
            for p in sorted((OUT/'discovery').glob(f'case{c["id"]}_action{a}_seed*/trajectory.json')):
                fs=json.loads(p.read_text())['frames']
                pairs.append([fs[min(h,len(fs)-1)]['score']-fs[0]['score'] for h in (8,512)])
            assert len(pairs)==16
            means[a]=np.mean(pairs,axis=0).tolist()
        # Prefer the recorded action on an exact tie; otherwise stable action order.
        order=sorted(legal,key=lambda a:(a!=c['recorded_action'],a))
        choices.append(dict(case_id=c['id'],recorded_action=c['recorded_action'],
            short_action=max(order,key=lambda a:means[a][0]),
            long_action=max(order,key=lambda a:means[a][1]),discovery_means=means))
    return choices


def compare(cases,choices,results):
    indexed={(r['case_id'],r['future_seed'],r['first_action']):r['additional_points'] for r in results}
    rows=[]
    for c,ch in zip(cases,choices):
        seeds=sorted({r['future_seed'] for r in results if r['case_id']==c['id']})
        values={name:np.array([indexed[c['id'],s,ch[key]] for s in seeds],dtype=float)
                for name,key in [('recorded','recorded_action'),('short','short_action'),('long','long_action')]}
        diffs={}
        for name,a,b in [('long_minus_recorded','long','recorded'),('short_minus_recorded','short','recorded'),('long_minus_short','long','short')]:
            d=values[a]-values[b]
            boot=np.random.default_rng(8091+c['id']).choice(d,(20000,len(d)),replace=True).mean(1)
            diffs[name]=dict(mean=float(d.mean()),paired_bootstrap95=np.quantile(boot,[.025,.975]).tolist())
        rows.append(dict(case_id=c['id'],source=c['source'],source_move=c['source_move'],
            actions={n:ACTION_NAMES[ch[k]] for n,k in [('recorded','recorded_action'),('short','short_action'),('long','long_action')]},
            mean_points={n:float(v.mean()) for n,v in values.items()},differences=diffs))
    return dict(complete=True,conditional=True,excluded_from_full_game_rankings=True,cases=rows,
        mean_difference_across_fixed_boards={k:float(np.mean([r['differences'][k]['mean'] for r in rows]))
            for k in ('long_minus_recorded','short_minus_recorded','long_minus_short')},
        note='Actions fixed from16-stream discovery before128-stream validation. Both selectors are evaluated '
            'on the SAME512-step future returns. Six selected board clusters, not768 independent boards. '
            'Conditional estimates and exploratory per-board intervals are not full-game scores or a universal ranking. '
            'No automatic online-policy promotion; validation is not used to change selected actions.')


def main():
    if STOP.exists():raise RuntimeError('User STOP is present')
    source=BASE/'work_table_ablation/screen'
    assert json.loads((source/'summary.json').read_text())['complete']
    assert all(json.loads((source/'verified.json').read_text()).values())
    cases=[]
    for path in sorted(source.glob('seed*_tables1/audit.json')):
        if path.parent.name=='seed8982012_tables1':continue
        audit=json.loads(path.read_text())
        assert audit['replay_sha256']==digest(path.parent/'replay.json')
        for d in audit['decisions']:
            risks=[v for v in d['risks'].values() if v is not None]
            if (max(map(max,d['board']))>=16384 and len(risks)>1 and
                    d['risks'][d['actual_action']]>min(risks)+1e-12):
                cases.append(dict(id=len(cases),source=str(path.parent.relative_to(BASE)),
                    source_replay_sha256=audit['replay_sha256'],source_move=d['board_index']+1,
                    board=d['board'],score=d['score_before'],recorded_action=ACTION_NAMES.index(d['actual_action'])))
                break
    assert len(cases)==6
    OUT.mkdir(exist_ok=False)
    hashes,caches=fingerprints(),cache_fingerprints()
    write(OUT/'protocol.json',dict(cases=cases,fingerprints=hashes,cache_sha256=caches,
        selection='First recorded decision per other with-table game with max tile>=16384, at least two legal '
            'actions, and chosen immediate death risk above the minimum. Exclude the earlier two-board source. '
            'All six eligible games included; cases fixed before simulations.',
        hypothesis='Does selecting by512-step rather than8-step estimated returns improve held-out '
            '512-step continuation reward, and does either beat the recorded strong policy?',
        design='16fresh streams per legal action for discovery. Choose largest8/512-step mean, ties prefer '
            'recorded action. Freeze choices. Then128disjoint future streams per legal action for validation. '
            'Same trajectories for every horizon and same128validation streams for every selected action.',
        seeds='Discovery9010000+1000*case_id+[0..15]; validation9010100+1000*case_id+[0..127].',
        limits='Offline conditional study only,16CPUworkers,900seconds per phase,120seconds/3GiB per rollout. '
            'No artificial full-game record, no online intervention, no guarantee of policy improvement.'))
    s,_=batch('discovery',cases,16,0)
    if not s['complete']:
        write(OUT/'status.json',dict(phase='incomplete_discovery',active=0));return
    choices=choose(cases)
    write(OUT/'choices.json',dict(choices=choices,frozen_before_validation=True))
    choices_hash=digest(OUT/'choices.json')
    s,results=batch('validation',cases,128,100)
    verified=dict(policy_unchanged=fingerprints()==hashes,cache_unchanged=cache_fingerprints()==caches,
        choices_unchanged=digest(OUT/'choices.json')==choices_hash,
        source_replays_unchanged=all(digest(BASE/c['source']/'replay.json')==c['source_replay_sha256'] for c in cases))
    write(OUT/'verified.json',verified)
    result=compare(cases,choices,results) if s['complete'] and all(verified.values()) else dict(complete=False,cases=[])
    write(OUT/'summary.json',result)
    write(OUT/'status.json',dict(phase='complete' if result['complete'] else 'incomplete',active=0))


if __name__=='__main__':main()
