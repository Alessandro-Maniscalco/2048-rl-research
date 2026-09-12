"""Conditional late-game test from three recorded first-65,536 positions.

These are saved-position continuations with four fresh future RNG streams per
position. They are not full games from two tiles and are excluded from score
records and full-game mean rankings. Compare additional points, not start score.
"""
from collections import Counter
from concurrent.futures import FIRST_COMPLETED,ProcessPoolExecutor,wait
import json
from pathlib import Path
import resource
import time

import numpy as np

from research.afterstate_teacher import digest,write
from research.full_rank_search import LIBRARY
from research.hybrid_endgame_study import BASE,ROOT,STOP,cache_fingerprints
from research.late_full_rank import LateFullRankAgent
from research.work_budget_hybrid import WorkBudgetHybridAgent
from research.work_budget_study import fingerprints as work_fingerprints
from rl2048.game import Game2048,ACTION_NAMES
from rl2048.view import COLORS

OUT=BASE/'late_game_comparison'
SOURCES=['hybrid_validation100/seed8975052','tactical_comparison/tactical_seed8976007',
         'work_budget_comparison/budget4000000_seed8978003']


def fingerprints():
    files=[ROOT/'research/late_game_study.py',ROOT/'research/late_full_rank.py',
           ROOT/'research/full_rank_search.py',LIBRARY,
           ROOT/'research/native/full_rank_search.cc',*sorted((LIBRARY.parent/'src').glob('*'))]
    return dict(work_fingerprints(),**{str(p.relative_to(ROOT)):digest(p) for p in files})


def initialize(case,seed):
    env=Game2048()
    env.reset(seed=seed)
    # Retain a naturally reached saved board, with new explicit future draws.
    # Initialization's two draws are discarded: future stream starts at seed.
    env.np_random=np.random.default_rng(seed)
    env.board=np.asarray(case['board'],dtype=np.int64).copy()
    env.score=case['score']
    env.steps=0
    env._terminated=False
    from rl2048.game import legal_actions
    return env,env.board.copy(),dict(action_mask=legal_actions(env.board))


def run(folder,case,seed,full_rank,seconds=7200,move_cap=50000):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    hashes=fingerprints()
    write(folder/'protocol.json',dict(case=case,future_seed=seed,full_rank=full_rank,
        seconds_budget=seconds,move_cap=move_cap,rss_gib_budget=3.,fingerprints=hashes,
        conditional=True,excluded_from_full_game_rankings=True,
        reason='Test the late-stage representation where it matters, without spending99outof100 '
            'full games on trajectories that never reach65536.',
        policy='Frozen tables first; then '+('full-rank depth8 search with orientation-adaptive '
            'heuristic and probability pruning' if full_rank else 'unmodified four-million-work compact search'),
        initialization='Three preselected recorded first65536boards, four fresh future RNG streams each. '
            'The first frame already has a large tile and score. No claim of natural full-game performance. '
            'Each pair has the same starting board, score and future seed.'))
    agent=LateFullRankAgent(4000000,8) if full_rank else WorkBudgetHybridAgent(4000000)
    env,board,info=initialize(case,seed)
    frames=[dict(board=board.tolist(),score=env.score,action=None,reward=0)]
    decisions=[];done=False;reason=None;peak=0.;started=time.perf_counter()
    try:
        while not done:
            peak=max(peak,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3)
            if STOP.exists():reason='stop_file';break
            if time.perf_counter()-started>=seconds:reason='time_budget';break
            if peak>3.:reason='memory_budget';break
            if len(decisions)>=move_cap:reason='move_cap';break
            try:action=agent.act(board,info['action_mask'])
            except Exception as error:reason=repr(error);break
            board,reward,done,truncated,info=env.step(action)
            assert not truncated
            decisions.append(agent.last_decision)
            frames.append(dict(board=board.tolist(),score=env.score,action=action,reward=reward))
            if len(decisions)%128==0 or done:
                write(folder/'status.json',dict(phase='complete' if done else 'running',
                    additional_points=env.score-case['score'],final_score_so_far=env.score,
                    max_tile=int(board.max()),moves=len(decisions),elapsed_seconds=time.perf_counter()-started))
    finally:agent.tables.close()
    result=dict(case_id=case['id'],future_seed=seed,full_rank=full_rank,
        conditional=True,excluded_from_full_game_rankings=True,start_score=case['score'],
        additional_points=env.score-case['score'],score=env.score,length=len(decisions),
        max_tile=int(board.max()),complete=done,terminated=done,truncated=not done,stop_reason=reason,
        elapsed_seconds=time.perf_counter()-started,peak_rss_gib=peak,
        modes=dict(Counter(d['mode'] for d in decisions)),source_unchanged=fingerprints()==hashes)
    replay=dict(algorithm='SAVED-POSITION CONTINUATION — '+agent.display_name,
        result=result,frames=frames,actions=ACTION_NAMES,colors=COLORS)
    write(folder/'replay.json',replay);write(folder/'decisions.json',decisions)
    template=(ROOT/'rl2048/replay.html').read_text()
    disclaimer='<p style="padding:12px;background:#fff0ca;color:#222">Saved-position continuation starting with a65,536tile. Excluded from full-game score records. Compare additional points only.</p>'
    template=template.replace('<body>','<body>'+disclaimer)
    detail=(f'Future spawn seed {seed} · {len(decisions):,} continuation moves · '
            f'Start score {case["score"]:,} · Additional points {env.score-case["score"]:,} · '
            +('Natural end' if done else f'Incomplete: {reason}'))
    template=template.replace('</body>',
        '<script>document.getElementById("details").textContent='+json.dumps(detail)+';</script></body>')
    (folder/'replay.html').write_text(template.replace('__REPLAY_DATA__',json.dumps(replay).replace('<','\\u003c')))
    # Replay every actual action from the same saved board and fresh future seed.
    checked,_,_=initialize(case,seed)
    for frame in frames[1:]:
        actual,reward,terminal,truncated,_=checked.step(frame['action'])
        assert np.array_equal(actual,frame['board']) and checked.score==frame['score'] and reward==frame['reward']
    assert checked.score==env.score and checked._terminated==done
    result['all_conditional_transitions_and_future_rng_audited']=True
    write(folder/'result.json',result)
    write(folder/'audit.json',dict(conditional=True,exact_future_seed=seed,
        every_transition_score_and_spawn_verified=True,source_case=case,replay_sha256=digest(folder/'replay.json')))
    write(folder/'status.json',dict(phase='complete' if done else 'incomplete',**result))
    return result


def summarize(results,jobs):
    expected={(j['case']['id'],j['future_seed'],j['full_rank']) for j in jobs}
    complete=(len(results)==len(expected) and
        {(r.get('case_id'),r.get('future_seed'),r.get('full_rank')) for r in results}==expected and
        all(r.get('complete') and r.get('source_unchanged') and
            r.get('all_conditional_transitions_and_future_rng_audited') for r in results))
    summary=dict(complete=complete,conditional=True,excluded_from_full_game_rankings=True,
        finished=len(results),planned=len(jobs),arms={},paired_mean_difference=None,per_case=[],
        note='Conditional late-game assay: only three starting positions, four future streams each. '
            'No full-game mean or score record. Treat positions as clusters, not12independent board samples. '
            'No bootstrap interval pretending12independent boards; report per-position differences.')
    if not complete:return summary
    indexed={(r['case_id'],r['future_seed'],r['full_rank']):r for r in results}
    differences=[]
    for case_id in range(3):
        ds=[]
        for seed in sorted({j['future_seed'] for j in jobs if j['case']['id']==case_id}):
            ds.append(indexed[case_id,seed,True]['additional_points']-indexed[case_id,seed,False]['additional_points'])
        differences.extend(ds)
        summary['per_case'].append(dict(case_id=case_id,paired_mean_difference=float(np.mean(ds)),per_stream=ds))
    summary['paired_mean_difference']=float(np.mean(differences))
    for full_rank,label in ((False,'compact'),(True,'full_rank')):
        arm=[r for r in results if r['full_rank']==full_rank]
        summary['arms'][label]=dict(mean_additional_points=float(np.mean([r['additional_points'] for r in arm])),
            mean_seconds=float(np.mean([r['elapsed_seconds'] for r in arm])),
            reached_131072=sum(r['max_tile']>=131072 for r in arm))
    return summary


def main():
    if STOP.exists():raise RuntimeError('User STOP is present')
    assert json.loads((BASE/'work_budget_comparison/summary.json').read_text())['complete']
    assert all(json.loads((BASE/'work_budget_comparison/verified.json').read_text()).values())
    probe=json.loads((BASE/'full_rank_probe.json').read_text())
    assert probe['complete'] and probe['repeated_queries_match'] and probe['max_seconds']<5
    cases=[]
    for i,name in enumerate(SOURCES):
        folder=BASE/name
        replay=json.loads((folder/'replay.json').read_text())
        a=json.loads((folder/'audit.json').read_text())
        assert a['exact_seeded_spawn_sequence_rechecked']
        assert digest(folder/'replay.json')==a['replay_sha256']
        index=next(j for j,f in enumerate(replay['frames']) if np.max(f['board'])>=65536)
        cases.append(dict(id=i,source=name,source_replay_sha256=a['replay_sha256'],
            source_move=index,board=replay['frames'][index]['board'],score=replay['frames'][index]['score']))
    jobs=[dict(name=f'case{c["id"]}_seed{s}_'+('full' if full else 'compact'),case=c,
        future_seed=s,full_rank=full) for c in cases
        for s in range(8981000+c['id']*4,8981004+c['id']*4) for full in (False,True)]
    OUT.mkdir(parents=True,exist_ok=False)
    hashes,caches=fingerprints(),cache_fingerprints()
    write(OUT/'protocol.json',dict(conditional=True,excluded_from_full_game_rankings=True,
        cases=cases,jobs=jobs,workers=16,fingerprints=hashes,cache_sha256=caches,
        hypothesis='Preserving actual high ranks may improve late-game continuation and enable '
            'another large merge. This changes the late search implementation and representation, '
            'not just depth. Original compact four-million-work controller is the control.',
        limits='24conditional continuations,7200seconds/3GiB/50000moves each. No selected restarts. '
            'Judge all12pairs and consistency across all3starting positions; no automatic promotion.'))
    active={};results=[];queue=iter(jobs);started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=16) as pool:
        def submit():
            if STOP.exists():return
            j=next(queue,None)
            if j is not None:active[pool.submit(run,OUT/j['name'],j['case'],j['future_seed'],j['full_rank'])]=j
        for _ in range(16):submit()
        while active:
            done,_=wait(active,timeout=10,return_when=FIRST_COMPLETED)
            for future in done:
                j=active.pop(future)
                try:r=future.result()
                except Exception as error:r=dict(case_id=j['case']['id'],future_seed=j['future_seed'],full_rank=j['full_rank'],complete=False,error=repr(error))
                results.append(r);submit()
            summary=summarize(results,jobs);summary['elapsed_wall_seconds']=time.perf_counter()-started
            write(OUT/'games.json',results);write(OUT/'summary.json',summary)
            write(OUT/'status.json',dict(phase='running',finished=len(results),active=list(active.values())))
    verified=dict(policy_unchanged=fingerprints()==hashes,cache_unchanged=cache_fingerprints()==caches)
    write(OUT/'verified.json',verified)
    if not all(verified.values()):
        summary.update(complete=False,arms={},paired_mean_difference=None,per_case=[],validation_error='Integrity check failed')
        write(OUT/'summary.json',summary)
    write(OUT/'status.json',dict(phase='complete' if summary['complete'] else 'incomplete',finished=len(results),active=[]))


if __name__=='__main__':main()
