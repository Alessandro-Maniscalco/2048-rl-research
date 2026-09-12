"""Validate the frozen rollout diagnostic with128 new streams per action."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from research.afterstate_teacher import digest, write
from research.hybrid_endgame_study import BASE, ROOT, STOP, cache_fingerprints
import research.long_rollout_probe as probe
from rl2048.game import legal_actions

OUT=BASE/'long_rollout_validation'


def run(job,seconds):
    # Reuse the frozen simulation verbatim. This process-local override changes
    # only the destination directory; it is restored before another job begins.
    with patch.object(probe,'OUT',OUT):
        return probe.run(job,seconds)


def fingerprints():
    return dict(probe.fingerprints(),**{'research/long_rollout_validation.py':digest(Path(__file__))})


def main():
    if STOP.exists():raise RuntimeError('User STOP is present')
    prior=BASE/'long_rollout_probe'
    assert json.loads((prior/'summary.json').read_text())['complete']
    assert all(json.loads((prior/'verified.json').read_text()).values())
    protocol=json.loads((prior/'protocol.json').read_text())
    cases=protocol['cases']
    jobs=[dict(name=f'case{c["id"]}_action{a}_seed{s}',case=c,first_action=int(a),future_seed=s)
          for c in cases for s in range(8984000+1000*c['id'],8984128+1000*c['id'])
          for a in np.flatnonzero(legal_actions(np.asarray(c['board'])))]
    OUT.mkdir(exist_ok=False)
    hashes,caches=fingerprints(),cache_fingerprints()
    write(OUT/'protocol.json',dict(cases=cases,jobs=jobs,horizon=512,workers=16,
        per_rollout_seconds=120,total_seconds=600,streams_per_action=128,
        fingerprints=hashes,cache_sha256=caches,discovery_protocol_sha256=digest(prior/'protocol.json'),
        source_replay_sha256=protocol['source_replay_sha256'],
        hypothesis='The recorded actions had the highest512-step sample returns despite lower '
            'exact8-step values. Check every action again on128 independent fresh streams per board '
            'to measure whether rare high-return branches caused a misleading16-stream result.',
        locked='Same two boards, all four actions, horizon512, one-million-work hybrid, '
            'fresh controller, raw rewards and zero terminal leaf. Only future samples and sample count change.',
        scope='Separate validation results, never pooled with discovery. Offline conditional diagnostic, '
            'not a full-game score, trained policy or online intervention. Multiple action comparisons '
            'remain exploratory; no automatic policy promotion.'))
    active={};results=[];queue=iter(jobs);start=time.perf_counter();deadline=start+600
    with ProcessPoolExecutor(max_workers=16) as pool:
        def submit():
            if STOP.exists() or time.perf_counter()>=deadline:return
            job=next(queue,None)
            if job is not None:active[pool.submit(run,job,max(1,min(120,deadline-time.perf_counter())))]=job
        for _ in range(16):submit()
        while active:
            done,_=wait(active,timeout=10,return_when=FIRST_COMPLETED)
            for future in done:
                job=active.pop(future)
                try:r=future.result()
                except Exception as error:r=dict(case_id=job['case']['id'],future_seed=job['future_seed'],
                    first_action=job['first_action'],valid_horizon=False,error=repr(error))
                results.append(r);submit()
            write(OUT/'games.json',results)
            write(OUT/'status.json',dict(phase='running',finished=len(results),planned=len(jobs),active=len(active),elapsed_wall_seconds=time.perf_counter()-start))
    for job in queue:
        results.append(dict(case_id=job['case']['id'],future_seed=job['future_seed'],
            first_action=job['first_action'],valid_horizon=False,stop_reason='not_started_stop_or_wall_budget'))
    verified=dict(policy_unchanged=fingerprints()==hashes,cache_unchanged=cache_fingerprints()==caches)
    write(OUT/'verified.json',verified)
    summary=probe.summarize(results,jobs,cases)
    summary['note']=summary['note'].replace('Sixteen future streams','128 fresh validation streams')
    summary.update(validation=True,not_pooled_with_discovery=True,elapsed_wall_seconds=time.perf_counter()-start)
    if not all(verified.values()):summary.update(complete=False,cases=[],validation_error='Integrity check failed')
    write(OUT/'games.json',results);write(OUT/'summary.json',summary)
    write(OUT/'status.json',dict(phase='complete' if summary['complete'] else 'incomplete',finished=len(results),planned=len(jobs),active=0))


if __name__=='__main__':main()
