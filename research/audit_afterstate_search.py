"""A fixed planner replay with exact same-board action and spawn-risk checks."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rl2048.agents.afterstate_mlp import AfterstateMLPAgent
from rl2048.agents.afterstate_planning import AfterstateLookahead
from rl2048.agents.ntuple import encode, row_tables
from rl2048.afterstate_expectation import spawn_outcomes
from rl2048.game import ACTION_NAMES, move
from rl2048.vector_game import legal_masks
from rl2048.view import save_replay, board_html


def audit(checkpoint, out, seed=8930100):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    agent = AfterstateMLPAgent.load(checkpoint)
    actor = AfterstateLookahead(agent, depth=2)
    deeper = AfterstateLookahead(agent, depth=3)
    game = save_replay(actor, out/'replay.html', seed=seed)
    frames = json.loads((out/'replay.json').read_text())['frames']
    count = len(frames)-1
    selected = sorted(set(np.linspace(0,count-1,8,dtype=int)) | set(range(max(0,count-8),count)))
    records = []
    for index in selected:
        board = np.array(frames[index]['board'])
        encoded = encode(board)[None]
        root, after, legal = agent.learner.decision(encoded)
        q = actor.planned_values(encoded)[0]
        chosen = int(q.argmax())
        assert chosen == frames[index+1]['action'], 'Replay and frozen planner disagree'
        q3 = deeper.planned_values(encoded)[0] if index >= count-4 else None
        actions = np.flatnonzero(legal[0])
        states, owners, probabilities = spawn_outcomes(after[0,actions])
        masks = legal_masks(states,*row_tables())
        alternatives = []
        for row, action in enumerate(actions):
            after_board, gain, _ = move(board,int(action))
            included = owners == row
            assert np.isclose(probabilities[included].sum(),1.)
            alternatives.append(dict(action=ACTION_NAMES[action],root_q=float(root[0,action]),
                two_move_q=float(q[action]),three_move_q=float(q3[action]) if q3 is not None else None,
                merge_points=int(gain),afterstate=after_board.tolist(),
                empty_cells_before_spawn=int((after_board==0).sum()),
                immediate_death_probability=float(np.dot(probabilities[included],~masks[included].any(1)))))
        records.append(dict(board_index=int(index),board=board.tolist(),chosen=ACTION_NAMES[chosen],
            root_choice=ACTION_NAMES[int(root[0].argmax())],
            three_move_choice=ACTION_NAMES[int(q3.argmax())] if q3 is not None else None,
            alternatives=alternatives))
    summary=dict(source=str(checkpoint.resolve()),seed=seed,game=game,planning_depth=2,
        checked_decisions=len(records),root_disagreements=sum(r['chosen']!=r['root_choice'] for r in records),
        third_move_disagreements=sum(r['three_move_choice'] is not None and r['chosen']!=r['three_move_choice'] for r in records),
        interpretation='Eight regularly spaced and final eight decisions, deduplicated. Three-move checks only on the final four. Exact spawn risks; future values remain learned estimates. Disagreement is not proof of optimality or a representative error rate.')
    (out/'audit.json').write_text(json.dumps(dict(summary=summary,decisions=records),indent=2,allow_nan=False))
    html=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2048 planner decisions</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:30px auto;padding:20px;background:#faf8ef;color:#544c44}td,th{padding:8px;text-align:left;border-bottom:1px solid #ccc}a{color:#197970}.scroll{overflow:auto}</style>',
        '<h1>Two-move planner: fixed replay and decisions</h1><a href="replay.html">Watch the complete game</a>',
        f'<p>Score {game["score"]:,}. {summary["interpretation"]}</p>']
    for r in records:
        html += [f'<h2>Board {r["board_index"]}: chosen {r["chosen"]}</h2>',
            board_html(np.array(r['board']),frames[r['board_index']]['score'],'Before the decision'),
            '<div class="scroll"><table><tr><th>Move</th><th>Root value</th><th>Two moves</th><th>Three moves</th><th>Merge points</th><th>Immediate death risk</th></tr>']
        for a in r['alternatives']:
            deeper_value='Not checked' if a['three_move_q'] is None else f'{a["three_move_q"]:.3f}'
            html.append(f'<tr><td>{a["action"]}</td><td>{a["root_q"]:.3f}</td><td>{a["two_move_q"]:.3f}</td><td>{deeper_value}</td><td>{a["merge_points"]}</td><td>{a["immediate_death_probability"]:.1%}</td></tr>')
        html.append('</table></div>')
    (out/'index.html').write_text(''.join(html)+'</html>')
    print(json.dumps(summary),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=8930100)
    a=p.parse_args(); audit(a.checkpoint,a.out,a.seed)
