"""Finish every time-limited game without replaying its expensive searches.

This is a separate, disclosed continuation protocol, not a retroactive edit of
the original1800-second screen. Restore the exact environment by replaying its
recorded legal actions from its original seed and verifying every frame/reward.
Original pilots did not save the adaptive timing state, so the first resumed
segment resets that planner state. Do NOT call these identical frozen-policy
evaluations or silently substitute them into screen8's mean.
"""
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from collections import Counter
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np

from research.afterstate_teacher import digest,write
from research.endgame_experiment import fingerprints,BASE,ROOT
from research.native_policy_data import extract_game
from rl2048.agents.endgame import EndgameAgent
from rl2048.game import Game2048,ACTION_NAMES
from rl2048.view import COLORS


def restore_environment(replay):
    """Replay actions, never search again; exact seed also restores spawn RNG."""
    env=Game2048();board,info=env.reset(seed=replay['result']['seed'])
    frames=replay['frames']
    if frames[0]['score']!=0 or not np.array_equal(board,frames[0]['board']):
        raise ValueError('Initial seeded board mismatch')
    terminated=False
    for frame in frames[1:]:
        if terminated:raise ValueError('Actions after natural game termination')
        board,reward,terminated,truncated,info=env.step(frame['action'])
        if (truncated or reward!=frame['reward'] or env.score!=frame['score']
                or not np.array_equal(board,frame['board'])):
            raise ValueError('Recorded prefix differs from the exact seeded trajectory')
    return env,board,info,terminated


def run(prefix,out,seconds=3600):
    prefix,out=Path(prefix),Path(out)
    out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    original=json.loads((prefix/'result.json').read_text())
    if original['complete'] or original['stop_reason']!='time_budget':
        raise ValueError('Only naturally unfinished time-limited games may resume here')
    protocol=json.loads((prefix/'protocol.json').read_text())
    hashes=fingerprints()
    if hashes!=protocol['fingerprints']:
        raise ValueError('Planner source/binary changed since prefix')
    replay=json.loads((prefix/'replay.json').read_text())
    env,board,info,done=restore_environment(replay)
    if done:raise ValueError('Prefix already terminal')
    frames=replay['frames'];decisions=json.loads((prefix/'decisions.json').read_text())
    if len(decisions)!=len(frames)-1:raise ValueError('Missing recorded decisions')
    agent=EndgameAgent(protocol['time_scale'])
    write(out/'protocol.json',dict(seed=original['seed'],prefix=str(prefix.resolve()),
        prefix_sha256=digest(prefix/'replay.json'),prefix_moves=original['length'],
        fingerprints=hashes,time_scale=protocol['time_scale'],seconds_budget=seconds,
        environment_restored_from_exact_seed_and_actions=True,planner_timing_state_reset=True,
        reason='Finish all time-limited attempts regardless of score. Preserve the original incomplete1800-second screen. '
            'Original adaptive timing state was not checkpointed; a fresh planner is explicitly part of this separate continuation. '
            'No move is undone, no spawn resampled, no chosen restart, no artificial game score. '
            'Report complete individual games and costs separately, not as unchanged screen8policy results.'))
    count_before=len(decisions);reason=None;peak=0
    while not done:
        peak=max(peak,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3)
        if (ROOT/'runs/research/scaled_transformer/STOP').exists():reason='stop_file';break
        if time.perf_counter()-started>=seconds:reason='time_budget';break
        if peak>2:reason='memory_budget';break
        if len(decisions)>=100000:reason='move_cap';break
        try:action=agent.act(board,info['action_mask'])
        except Exception as error:reason=repr(error);break
        board,reward,done,truncated,info=env.step(action)
        assert not truncated
        decisions.append(agent.last_decision)
        frames.append(dict(board=board.tolist(),score=env.score,action=action,reward=reward))
        if len(decisions)%128==0 or done:
            write(out/'status.json',dict(phase='complete' if done else 'running',score=env.score,
                moves=len(decisions),max_tile=int(board.max()),prefix_moves=count_before,
                elapsed_seconds=time.perf_counter()-started+original['elapsed_seconds'],
                segment_seconds=time.perf_counter()-started,planner_timing_state_reset=True))
    result=dict(seed=original['seed'],score=env.score,length=len(decisions),max_tile=int(board.max()),
        complete=done,terminated=done,truncated=not done,stop_reason=reason,
        elapsed_seconds=original['elapsed_seconds']+time.perf_counter()-started,
        segment_seconds=time.perf_counter()-started,prefix_moves=count_before,
        source_and_binary_unchanged=fingerprints()==hashes,
        planner_timing_state_reset=True,environment_restored_from_exact_seed_and_actions=True,
        modes=dict(Counter(d['mode'] for d in decisions)),peak_rss_gib=peak)
    replay=dict(algorithm=agent.display_name+' · continued with planner timing reset',
        result=result,frames=frames,actions=ACTION_NAMES,colors=COLORS)
    write(out/'replay.json',replay);write(out/'decisions.json',decisions)
    (out/'replay.html').write_text((ROOT/'rl2048/replay.html').read_text().replace('__REPLAY_DATA__',json.dumps(replay).replace('<','\\u003c')))
    # Future segments can restore the explicit Python timing state; native caches
    # are ephemeral and recreated for root searches. This first rescue disclosed
    # a reset because its original prefix had no timing checkpoint.
    state={key:getattr(agent.logic,key) for key in
           ('last_depth','last_sum','last_prune','last_move','time_ratio','time_limit_ratio')}
    state={k:v.item() if isinstance(v,np.generic) else v for k,v in state.items()}
    write(out/'planner_timing_state.json',state)
    if done:extract_game(out/'replay.json');result['all_transitions_rechecked']=True
    write(out/'result.json',result);write(out/'status.json',dict(phase='complete' if done else 'incomplete',**result))
    return result


def supervise():
    out=BASE/'continuation_protocol';out.mkdir(exist_ok=False)
    seeds=list(range(8972000,8972009));waiting=set(seeds);futures={};results=[]
    write(out/'protocol.json',dict(seeds=seeds,workers=8,seconds_per_resumed_segment=3600,
        original_screen_preserved=True,selection_rule='Resume every time_budget exit irrespective of its score; keep naturally completed games untouched.',
        reason='The first completed game took936seconds while others were still active above1000seconds. '
            'Some long games may exceed the prespecified1800secondlimit. Finish their exact existing trajectories '
            'without paying for old searches again. The original timing state was not saved, so resumed games '
            'have a disclosed planner timing reset and are separate from the original fixed screen.'))
    with ProcessPoolExecutor(max_workers=8) as pool:
        while waiting or futures:
            if (ROOT/'runs/research/scaled_transformer/STOP').exists():waiting.clear()
            for seed in sorted(list(waiting)):
                p=BASE/f'pilot_seed{seed}'/'result.json'
                if not p.exists():continue
                original=json.loads(p.read_text());waiting.remove(seed)
                if not original['complete'] and original['stop_reason']=='time_budget':
                    f=pool.submit(run,p.parent,BASE/f'continued_seed{seed}',3600);futures[f]=seed
                else:results.append(dict(seed=seed,original_result=original,resumed=False))
            if futures:
                completed,_=wait(futures,timeout=5,return_when=FIRST_COMPLETED)
                for f in completed:
                    seed=futures.pop(f)
                    try:r=f.result()
                    except Exception as error:r=dict(complete=False,error=repr(error))
                    results.append(dict(seed=seed,resumed=True,result=r))
            else:time.sleep(5)
            write(out/'status.json',dict(phase='running' if waiting or futures else 'finished',
                waiting_for_original=sorted(waiting),active=list(futures.values()),processed=len(results)))
            write(out/'results.json',results)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args();supervise()
