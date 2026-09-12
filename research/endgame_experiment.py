"""Bounded standard-game evaluation of the compact external endgame engine.

We do not run upstream AItest's altered high-tile simulation. All real moves,
spawns, rewards and termination come from our unchanged Game2048. Adaptive
wall-time search may vary by machine load; save every actual decision and frame.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import subprocess
import time

import numpy as np

from research.afterstate_teacher import digest,write
from research.native_policy_data import extract_game
from rl2048.agents.endgame import EndgameAgent,VENDOR,ROOT
from rl2048.game import Game2048,ACTION_NAMES
from rl2048.view import COLORS


BASE=ROOT/'runs/research/endgame_tablebase'


def fingerprints():
    paths=list((VENDOR/'native_core').glob('*core*.so'))+[VENDOR/'engine_core/AIPlayer.py',
        VENDOR/'native_core/src/AIPlayer.cpp',VENDOR/'native_core/egtb_data.7z',
        ROOT/'rl2048/agents/endgame.py']
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}


def run(out,seed=8972000,time_scale=1.,seconds=1800,rss_gib=2.):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    hashes=fingerprints();started=time.perf_counter()
    write(out/'protocol.json',dict(seed=seed,time_scale=time_scale,seconds_budget=seconds,
        rss_gib_budget=rss_gib,source='https://github.com/game-difficulty/2048EndgameTablebase',
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=VENDOR,text=True).strip(),
        fingerprints=hashes,search_threads=1,game='Unchanged exact Python Game2048, standard two-tile start,2/4spawn90/10,no undo.',
        reason='Test a materially different stronger compact planning recipe: broader formation heuristics, adaptive search, embedded compressed subgoal tables. '
            'Published772353average uses large external tables which are NOT included here. This tests the compact standalone CoreAILogic.',
        portability='AppleClangC++17O3,serial search per process,matching aligned new/delete,guard no-move root score indexing. '
            'Separate task-local CMake4.4.3andpy-cpuinfo9.0.0; no large table generation or global dependency changes.',
        limitations='Upstream planner treats ranks>=15as opaque large tiles and downgrades two32768onlyin search before65536. '
            'Our game never downgrades or invents points. Wall-time iterative deepening can vary by CPUload; compare actual saved decisions.'))
    (out/'portability.patch').write_text(subprocess.check_output(['git','diff'],cwd=VENDOR,text=True))
    env=Game2048();board,info=env.reset(seed=seed)
    agent=EndgameAgent(time_scale)
    frames=[dict(board=board.tolist(),score=0,action=None,reward=0)]
    decisions=[];done=False;reason=None;peak=0
    while not done:
        peak=max(peak,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3)
        if (ROOT/'runs/research/scaled_transformer/STOP').exists():reason='stop_file';break
        if time.perf_counter()-started>=seconds:reason='time_budget';break
        if peak>rss_gib:reason='memory_budget';break
        if len(decisions)>=100000:reason='move_cap';break
        try:action=agent.act(board,info['action_mask'])
        except Exception as error:
            reason=repr(error);break
        before=board.copy()
        board,reward,done,truncated,info=env.step(action)
        assert not truncated and not np.array_equal(before,board)
        decisions.append(agent.last_decision)
        frames.append(dict(board=board.tolist(),score=env.score,action=action,reward=reward))
        if len(decisions)%128==0 or done:
            write(out/'status.json',dict(phase='complete' if done else 'running',moves=len(decisions),
                score=env.score,max_tile=int(board.max()),elapsed_seconds=time.perf_counter()-started,
                peak_rss_gib=peak,last_decision=agent.last_decision))
    result=dict(seed=seed,score=env.score,length=len(decisions),max_tile=int(board.max()),
        terminated=done,truncated=not done,complete=done,stop_reason=reason,
        elapsed_seconds=time.perf_counter()-started,peak_rss_gib=peak,
        modes=dict(Counter(d['mode'] for d in decisions)),
        high_rank_abstracted_decisions=sum(d['high_rank_abstraction'] for d in decisions))
    replay=dict(algorithm=agent.display_name,result=result,frames=frames,colors=COLORS,actions=ACTION_NAMES)
    write(out/'replay.json',replay);write(out/'decisions.json',decisions)
    (out/'replay.html').write_text((ROOT/'rl2048/replay.html').read_text().replace('__REPLAY_DATA__',json.dumps(replay).replace('<','\\u003c')))
    result['source_and_binary_unchanged']=fingerprints()==hashes
    if done:
        extract_game(out/'replay.json');result['all_transitions_rechecked']=True
    write(out/'result.json',result)
    write(out/'status.json',dict(phase='complete' if done else 'incomplete',**result))
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--seed',type=int,default=8972000)
    parser.add_argument('--time-scale',type=float,default=1.)
    parser.add_argument('--seconds',type=int,default=1800);parser.add_argument('--rss-gib',type=float,default=2.)
    run(**vars(parser.parse_args()))
