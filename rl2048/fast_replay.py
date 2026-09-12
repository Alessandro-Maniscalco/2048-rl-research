"""Record the exact compiled-RNG game selected from a fast evaluation.

Unlike save_replay's reference Game2048 stream, this matches evaluate_game.
The title explicitly labels a selected high-score game rather than a fixed demo.
"""
import json
from pathlib import Path
import time
import numpy as np
from numba import njit
from rl2048.fast2048 import spawn,slide,search,downgrade_root
from rl2048.agents.ntuple import decode,row_tables
from rl2048.view import COLORS
from rl2048.game import ACTION_NAMES


@njit(cache=True)
def record_game(weights,patterns,rows,gains,seed,depth,cutoff,threshold,max_steps=40_000):
    np.random.seed(seed)
    board=np.zeros(16,np.uint8);spawn(board);spawn(board)
    boards=np.zeros((max_steps+1,16),np.uint8);boards[0]=board
    actions=np.zeros(max_steps,np.int64);rewards=np.zeros(max_steps,np.int64)
    count=0;terminated=False
    for t in range(max_steps):
        root=downgrade_root(board,threshold) if threshold else board
        action=search(root,weights,patterns,rows,gains,depth,cutoff)
        if action<0:
            terminated=True;break
        board,gain,_=slide(board,action,rows,gains);spawn(board)
        boards[t+1]=board;actions[t]=action;rewards[t]=gain;count=t+1
    return boards[:count+1],actions[:count],rewards[:count],terminated


def save_fast_replay(agent,path,seed,*,selected=True):
    start=time.perf_counter()
    boards,actions,rewards,terminated=record_game(agent.weights,agent.patterns,*row_tables(),
        seed,agent.depth,agent.cutoff,agent.downgrade_threshold)
    frames=[{'board':decode(boards[0]).tolist(),'score':0,'action':None,'reward':0}]
    score=0
    for board,action,reward in zip(boards[1:],actions,rewards,strict=True):
        score+=int(reward)
        frames.append({'board':decode(board).tolist(),'score':score,'action':int(action),'reward':int(reward)})
    result={'seed':seed,'score':score,'length':len(actions),'max_tile':int(decode(boards[-1]).max()),
            'terminated':bool(terminated),'truncated':not bool(terminated),'elapsed_seconds':time.perf_counter()-start,
            'rng':'Numba MT19937','selected_high_score':selected}
    label=agent.display_name+(' · selected high-score game' if selected else '')
    data={'algorithm':label,'result':result,'frames':frames,'colors':COLORS,'actions':ACTION_NAMES}
    path=Path(path)
    path.with_suffix('.json').write_text(json.dumps(data))
    template=Path(__file__).with_name('replay.html').read_text()
    path.write_text(template.replace('__REPLAY_DATA__',json.dumps(data).replace('<','\\u003c')))
    return result
