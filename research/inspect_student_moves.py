"""Reproduce Q predictions and exact immediate Left/Right counterfactuals.

Counterfactual rewards average ALL possible spawns, not an invented shared
spawn location. This does not establish which move has the better full return.
"""
import json
from pathlib import Path
import numpy as np
import torch
from rl2048.agents.neural import NeuralAgent, tensor_boards
from rl2048.agents.ntuple import encode
from rl2048.agents.deep_q_planning import DeepQPlanner
from rl2048.game import move, legal_actions, ACTION_NAMES
from rl2048.rewards import learning_rewards, snake_potential, corner_potential
from rl2048.view import board_html


def inspect(out=Path('runs/research/deeper_q')):
    replay=json.loads((out/'replay_latest_student.json').read_text())
    student=NeuralAgent.load(out/'stronger_students/student_depth3')
    teacher=NeuralAgent.load('runs/research/dqn16_planning_selected/network')
    planner=DeepQPlanner(teacher,.99,'corner_snake',2,depth=3)
    records=[];parts=[]
    for i in (500,501):
        b=np.array(replay['frames'][i]['board']);encoded=encode(b).reshape(1,16)
        with torch.no_grad():q=student.policy(tensor_boards(encoded,'cpu')).numpy()[0]
        tq=planner.planned_values(encoded)[0]
        actual=replay['frames'][i+1]
        actual_shaped=float(learning_rewards(np.array([actual['reward']]),encoded,
            encode(np.array(actual['board'])).reshape(1,16),np.array([False]),.99,'corner_snake',2)[0])
        record=dict(board_index=i,board=b.tolist(),student_q=q.tolist(),
            legal_mask=legal_actions(b).tolist(),teacher_depth3_q=[float(x) if np.isfinite(x) else None for x in tq],
            actual_next_move=ACTION_NAMES[actual['action']],actual_shaped_reward=actual_shaped,alternatives=[])
        parts += [f'<h2>Board {i} → move {i+1}</h2>',board_html(b,replay['frames'][i]['score'],'Before choosing the next move'),
            '<p>Student Q-values in Up / Right / Down / Left order: '+ ' / '.join(f'{x:.4f}' for x in q)+
            '. Illegal moves are excluded.</p>', '<div class="alternatives">']
        for action in (1,3):
            after,raw,valid=move(b,action);assert valid
            values=[];weights=[]
            empty=np.argwhere(after==0)
            for pos in empty:
                for tile,prob in ((2,.9),(4,.1)):
                    successor=after.copy();successor[tuple(pos)]=tile
                    reward=learning_rewards(np.array([raw]),encoded,encode(successor).reshape(1,16),
                        np.array([not legal_actions(successor).any()]),.99,'corner_snake',2)[0]
                    values.append(float(reward));weights.append(prob/len(empty))
            assert np.isclose(sum(weights),1)
            a=dict(action=ACTION_NAMES[action],afterstate_before_spawn=after.tolist(),merge_points=raw,
                expected_learning_reward=float(np.dot(values,weights)),min_learning_reward=min(values),
                max_learning_reward=max(values),corner_potential_before_spawn=float(corner_potential(encode(after)[None])[0]),
                snake_potential_before_spawn=float(snake_potential(encode(after)[None])[0]))
            record['alternatives'].append(a)
            parts += ['<section>',board_html(after,0,f'{ACTION_NAMES[action].title()} · before the random spawn'),
                f"<p>Merge points: <b>{raw}</b><br>Expected shaped reward: <b>{a['expected_learning_reward']:.4f}</b><br>"
                f"Possible range after spawning: {min(values):.4f}–{max(values):.4f}</p></section>"]
        parts += ['</div>',f'<p>Actual recorded shaped reward for {ACTION_NAMES[actual["action"]]}: {actual_shaped:.4f}. '
            'The counterfactual is averaged over possible new tile positions and values, so it is not a full-game score prediction.</p>']
        records.append(record)
    (out/'moves_500_502.json').write_text(json.dumps(records,indent=2,allow_nan=False))
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Why did the student go right?</title><style>body{font:17px/1.6 system-ui;background:#faf8ef;color:#544c44;max-width:950px;margin:40px auto;padding:0 24px}.alternatives{display:flex;gap:40px;flex-wrap:wrap}.alternatives section{flex:1;min-width:280px}a{color:#197970}</style>
<h1>Why did the student go right?</h1><p>The displayed move number describes the board after that many moves. Board 500 precedes the Right move displayed at 501; board 501 precedes the Right move displayed at 502.</p>
<p>The student chooses its largest legal predicted Q-value. Those estimates can be wrong. At board 500, Left has higher immediate shaped reward and the three-step teacher narrowly prefers Left. At board 501, the same teacher prefers Right. Neither comparison proves the optimal full-game action.</p>
<p>The old shaping reward is merge points / 128 + 2 × (0.99 × next potential − current potential). Its snake potential is a weighted sum of tile exponents, not a descending-order test. A broken snake does not set reward to zero. Corner potential is separate and vanishes when no largest tile is in the bottom-left cell.</p>
<p><a href="replay_latest_student.html">Open the replay</a></p>'''+''.join(parts)+'</html>'
    (out/'moves_500_502.html').write_text(html)
    return records


if __name__=='__main__':
    torch.set_num_threads(2)
    print(json.dumps(inspect(),indent=2))
