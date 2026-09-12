"""Fresh paired screen of compact search versus added frozen solved formations.

The only policy intervention is querying the old solved tables before the same
compact fallback. Table misses never compute new entries. Full exact Python
games, no chosen restart, no planner reset, and no partial completed-only means.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import json
from pathlib import Path
import resource
import time

import numpy as np

from research.afterstate_teacher import digest,write
from research.endgame_experiment import BASE,ROOT,fingerprints
from research.native_policy_data import extract_game
from rl2048.agents.endgame import EndgameAgent
from rl2048.agents.frozen_tablebase import CACHE,LIBRARY
from rl2048.agents.hybrid_endgame import HybridEndgameAgent
from rl2048.game import Game2048,ACTION_NAMES
from rl2048.view import COLORS

OUT=BASE/'hybrid_comparison'
STOP=ROOT/'runs/research/scaled_transformer/STOP'


def policy_fingerprints():
    paths=[LIBRARY,ROOT/'rl2048/agents/frozen_tablebase.py',ROOT/'rl2048/agents/hybrid_endgame.py',
           ROOT/'research/native/tablebase_bridge.cc',Path(__file__).resolve(),
           *sorted((LIBRARY.parent/'src').glob('*'))]
    return dict(fingerprints(),**{str(p.relative_to(ROOT)):digest(p) for p in paths})


def cache_fingerprints():
    return {p.name:digest(p) for p in sorted(CACHE.glob('tuple_moves.*'))}


def run(out,seed,hybrid,seconds=7200,move_cap=100000):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    hashes=policy_fingerprints();started=time.perf_counter()
    write(out/'protocol.json',dict(seed=seed,hybrid=hybrid,time_scale=1.,seconds_budget=seconds,
        rss_gib_budget=3.,move_cap=move_cap,fingerprints=hashes,
        policy='Frozen MacroXue tables then compact fallback' if hybrid else 'Unchanged compact CoreAILogic',
        table_selection='Read-only cached 11-cell success probability>=0.9, then10-cell>0, otherwise compact engine. '
            'No generation or saving; missing cached states fall through. Probabilities are restricted subgoal success, not game-score Q values.',
        game='Exact original Python Game2048, two starting tiles, unchanged90/10spawns and raw merge rewards.',
        reason='Test whether the previous engine\'s detailed solved formations complement the compact engine\'s broader planning, improving score and reducing search cost.',
        timing='Time-based adaptive search depends on CPUload. Same seeds pair environmental random streams, but do not force identical actions or CPUtiming. No planner timing resets during a game.'))
    agent=(HybridEndgameAgent if hybrid else EndgameAgent)(1.)
    env=Game2048();board,info=env.reset(seed=seed)
    frames=[dict(board=board.tolist(),score=0,action=None,reward=0)]
    decisions=[];done=False;reason=None;peak=0
    while not done:
        peak=max(peak,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3)
        if STOP.exists():reason='stop_file';break
        if time.perf_counter()-started>=seconds:reason='time_budget';break
        if peak>3:reason='memory_budget';break
        if len(decisions)>=move_cap:reason='move_cap';break
        try:action=agent.act(board,info['action_mask'])
        except Exception as error:reason=repr(error);break
        board,reward,done,truncated,info=env.step(action)
        assert not truncated
        decisions.append(agent.last_decision)
        frames.append(dict(board=board.tolist(),score=env.score,action=action,reward=reward))
        if len(decisions)%128==0 or done:
            write(out/'status.json',dict(phase='complete' if done else 'running',seed=seed,hybrid=hybrid,
                score=env.score,moves=len(decisions),max_tile=int(board.max()),
                elapsed_seconds=time.perf_counter()-started,peak_rss_gib=peak,
                modes=dict(Counter(d['mode'] for d in decisions))))
    result=dict(seed=seed,hybrid=hybrid,score=env.score,length=len(decisions),max_tile=int(board.max()),
        complete=done,terminated=done,truncated=not done,stop_reason=reason,
        elapsed_seconds=time.perf_counter()-started,peak_rss_gib=peak,
        modes=dict(Counter(d['mode'] for d in decisions)),
        source_and_binary_unchanged=policy_fingerprints()==hashes,
        planner_timing_state_reset=False)
    replay=dict(algorithm=agent.display_name,result=result,frames=frames,actions=ACTION_NAMES,colors=COLORS)
    write(out/'replay.json',replay);write(out/'decisions.json',decisions)
    (out/'replay.html').write_text((ROOT/'rl2048/replay.html').read_text().replace('__REPLAY_DATA__',json.dumps(replay).replace('<','\\u003c')))
    timing={key:getattr(agent.logic,key) for key in
        ('last_depth','last_sum','last_prune','last_move','time_ratio','time_limit_ratio')}
    write(out/'planner_timing_state.json',{k:v.item() if isinstance(v,np.generic) else v for k,v in timing.items()})
    if hybrid:agent.tables.close()
    if done:extract_game(out/'replay.json');result['all_transitions_rechecked']=True
    write(out/'result.json',result);write(out/'status.json',dict(phase='complete' if done else 'incomplete',**result))
    return result


def summarize(results,seeds):
    expected={(s,h) for s in seeds for h in (False,True)}
    actual={(r['seed'],r['hybrid']) for r in results}
    complete=len(results)==len(expected) and actual==expected and all(r.get('complete') for r in results)
    summary=dict(complete=complete,finished=len(results),planned=len(expected),
        incomplete_attempts=sum(not r.get('complete',False) for r in results),
        arms={},paired_difference=None,
        note='Fresh paired screening only. No partial completed-only means. Same game seeds, adaptive wall-time search. '
            'Eight pairs give wide uncertainty; a higher selected maximum does not establish higher average strength.')
    for hybrid,label in [(False,'compact'),(True,'hybrid')]:
        arm=[r for r in results if r['hybrid']==hybrid]
        finished=len(arm)==len(seeds) and all(r.get('complete') for r in arm)
        summary['arms'][label]=dict(complete=finished,finished=len(arm),
            mean_score=float(np.mean([r['score'] for r in arm])) if finished else None,
            best_completed_score=max((r['score'] for r in arm if r.get('complete')),default=None),
            mean_seconds=float(np.mean([r['elapsed_seconds'] for r in arm])) if finished else None,
            reaching_32768=sum(r.get('complete',False) and r.get('max_tile',0)>=32768 for r in arm),
            reaching_65536=sum(r.get('complete',False) and r.get('max_tile',0)>=65536 for r in arm))
    if complete:
        indexed={(r['seed'],r['hybrid']):r for r in results}
        differences=np.array([indexed[s,True]['score']-indexed[s,False]['score'] for s in seeds])
        bootstrap=np.random.default_rng(7819).choice(differences,(20000,len(seeds)),replace=True).mean(1)
        summary['paired_difference']=dict(mean=float(differences.mean()),
            bootstrap95=np.quantile(bootstrap,[.025,.975]).tolist(),
            wins=int((differences>0).sum()),ties=int((differences==0).sum()),per_seed=differences.tolist())
    return summary


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    seeds=list(range(8973000,8973008));hashes=policy_fingerprints();cache_hashes=cache_fingerprints()
    jobs=[(seed,hybrid) for seed in seeds for hybrid in (False,True)]
    write(OUT/'protocol.json',dict(seeds=seeds,workers=16,planned_games=16,
        per_game_seconds=7200,per_game_rss_gib=3.,move_cap=100000,
        fingerprints=hashes,cache_sha256=cache_hashes,source_cache='Existing validated small tables,read-only.',
        hypothesis='Solved strategic formations may complement compact adaptive search and reduce expensive search calls. '
            'Compare compact alone versus adding frozen MacroXue tables before the SAME compact fallback.',
        single_intervention='Old table recommendation takes priority when cached and qualifying. No new reward, safety rule, search-time multiplier or learned network.',
        compute='Sixteen independent single-thread games on the18-coreMac, paired arms start together. '
            'Compact pilots peaked at0.27GiB; hybrid smoke at0.32GiB. Read-only table mappings share file pages. '
            'Actual wall-time search depth depends on CPUload; record per-move durations and compare full outcomes/cost.',
        limits='Fresh seeds, no artificial start/restart/undo. Two-hour limit avoids prior30mincensoring. '
            'Any incomplete game stays explicit. Do not substitute continuation games. Both full endpoints before promotion. '
            'Time-dependent search can vary with CPUload; paired seeds do not prove identical prefixes.'))
    started=time.perf_counter();results=[];active={};queue=iter(jobs)
    summary=summarize(results,seeds)
    with ProcessPoolExecutor(max_workers=16) as pool:
        def submit():
            if STOP.exists():return
            job=next(queue,None)
            if job is None:return
            seed,hybrid=job;name=('hybrid' if hybrid else 'compact')+f'_seed{seed}'
            active[pool.submit(run,OUT/name,seed,hybrid)]=job
        for _ in range(16):submit()
        while active:
            done,_=wait(active,timeout=10,return_when=FIRST_COMPLETED)
            for f in done:
                seed,hybrid=active.pop(f)
                try:result=f.result()
                except Exception as error:result=dict(seed=seed,hybrid=hybrid,complete=False,error=repr(error))
                results.append(result);submit()
            summary=summarize(results,seeds);summary['elapsed_seconds']=time.perf_counter()-started
            write(OUT/'games.json',results);write(OUT/'summary.json',summary)
            write(OUT/'status.json',dict(phase='running',active=[dict(seed=s,hybrid=h) for s,h in active.values()],finished=len(results)))
    write(OUT/'verified.json',dict(policy_unchanged=policy_fingerprints()==hashes,
        cache_unchanged=cache_fingerprints()==cache_hashes))
    write(OUT/'status.json',dict(phase='complete' if summary['complete'] else 'incomplete',finished=len(results)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    if args.smoke:
        print(json.dumps(run(BASE/'hybrid_smoke',8972999,True,seconds=60,move_cap=256)))
    else:main()
