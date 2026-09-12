"""Audit saved policy-logit replays without pretending logits are Q-values.

Exact spawn probabilities give immediate-death risk for every legal move.
Optional frozen n-tuple values provide a strong but imperfect reference only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.afterstate_teacher import write, digest
from rl2048.agents.neural import NeuralAgent, tensor_boards, masked_log_probs
from rl2048.agents.ntuple import NTupleAgent, encode, row_tables
from rl2048.agents.q_planning import outcomes
from rl2048.fast2048 import value
from rl2048.game import ACTION_NAMES, move, legal_actions
from rl2048.vector_game import legal_masks
from rl2048.view import board_html


def audit(checkpoint, replay, out, teacher_checkpoint=None):
    torch.set_num_threads(1)
    checkpoint, replay, out = Path(checkpoint), Path(replay), Path(out)
    if out.exists():
        raise FileExistsError('Preserve completed audits; choose a new output directory')
    out.mkdir(parents=True)
    saved = json.loads(replay.read_text())
    frames = saved['frames']
    count = len(frames)-1
    agent = NeuralAgent.load(checkpoint, 'cpu')
    teacher = NTupleAgent.load(teacher_checkpoint, mmap_mode='r') if teacher_checkpoint else None
    selected = set(np.linspace(0, count-1, min(8, count), dtype=int).tolist())
    selected.update(range(max(0, count-8), count))
    lost_corners = []
    corners = lambda b: b[[0, 0, 3, 3], [0, 3, 0, 3]]
    for i in range(count):
        board = np.asarray(frames[i]['board'])
        after, _, _ = move(board, frames[i+1]['action'])
        if board.max() >= 128 and board.max() in corners(board) and after.max() not in corners(after):
            lost_corners.append(i)
    selected.update(lost_corners[-4:])
    records = []
    for i in sorted(selected):
        board = np.asarray(frames[i]['board'])
        encoded = encode(board)[None]
        mask = legal_actions(board)
        with torch.no_grad():
            logits = agent.policy(tensor_boards(encoded, 'cpu'))[:, :4]
            policy_mask = agent.policy_mask(encoded, mask[None])
            probabilities = masked_log_probs(logits, torch.as_tensor(policy_mask)).exp().numpy()[0]
        actual = int(frames[i+1]['action'])
        states, actions, probability, _ = outcomes(encoded.reshape(16), *row_tables())
        next_masks = legal_masks(states, *row_tables())
        alternatives = []
        for action in np.flatnonzero(mask):
            after, reward, _ = move(board, int(action))
            pick = actions == action
            assert np.isclose(probability[pick].sum(), 1.)
            q = float(reward+value(encode(after), teacher.weights, teacher.patterns)) if teacher else None
            alternatives.append(dict(action=ACTION_NAMES[action], action_index=int(action),
                policy_probability=float(probabilities[action]), logit=float(logits[0, action]),
                allowed_by_policy_mask=bool(policy_mask[0, action]),
                merge_points=int(reward), teacher_q_points=q,
                immediate_death_probability=float(np.dot(probability[pick], ~next_masks[pick].any(1))),
                expected_legal_next_actions=float(np.dot(probability[pick], next_masks[pick].sum(1))),
                empty_cells_before_spawn=int((after == 0).sum()), afterstate=after.tolist()))
        chosen = next(a for a in alternatives if a['action_index'] == actual)
        best_risk = min(a['immediate_death_probability'] for a in alternatives)
        records.append(dict(board_index=i, board=board.tolist(), score=frames[i]['score'],
            recorded_action=ACTION_NAMES[actual], cpu_reproduces_recorded_action=int(probabilities.argmax()) == actual,
            chosen_death_probability=chosen['immediate_death_probability'],
            lowest_available_death_probability=best_risk,
            avoidable_immediate_risk=chosen['immediate_death_probability'] > best_risk+1e-9,
            teacher_choice=max(alternatives, key=lambda a: a['teacher_q_points'])['action'] if teacher else None,
            alternatives=alternatives))
    summary = dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=digest(checkpoint/'policy.pt'),
        replay=str(replay.resolve()), game=saved['result'], checked_decisions=len(records),
        cpu_action_matches=sum(r['cpu_reproduces_recorded_action'] for r in records),
        avoidable_immediate_risk_decisions=sum(r['avoidable_immediate_risk'] for r in records),
        lost_any_corner_events=len(lost_corners),
        interpretation='Recorded MPS moves remain the source of truth. Regular, final-eight and recent lost-corner boards are selected diagnostics, not a representative error rate. Exact spawn-weighted death risk is a one-move fact, not the long-term optimal action. Teacher values are imperfect estimates; policy logits are not Q-values. No training data or rewards changed.')
    write(out/'audit.json', dict(summary=summary, decisions=records))
    html = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Actor transition audit</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:30px auto;padding:20px;background:#faf8ef;color:#544c44}table{width:100%}th,td{padding:8px;text-align:left;border-bottom:1px solid #ccc}.scroll{overflow:auto}a{color:#197970}</style>',
        '<h1>Inspecting actual actor decisions</h1>', f'<p>{summary["interpretation"]}</p>',
        f'<p>Replay score {saved["result"]["score"]:,}; {count} moves. {summary["avoidable_immediate_risk_decisions"]} selected decisions had a lower immediate-risk alternative.</p>']
    for r in records:
        html.extend([f'<h2>Board {r["board_index"]}: recorded {r["recorded_action"]}</h2>',
            board_html(np.array(r['board']), r['score']), '<div class="scroll"><table><tr><th>Action</th><th>Policy probability</th><th>Merge points</th><th>Immediate death probability</th><th>Teacher Q points</th></tr>'])
        for a in r['alternatives']:
            q = f'{a["teacher_q_points"]:,.1f}' if a['teacher_q_points'] is not None else '—'
            html.append(f'<tr><td>{a["action"]}</td><td>{a["policy_probability"]:.1%}</td><td>{a["merge_points"]}</td><td>{a["immediate_death_probability"]:.1%}</td><td>{q}</td></tr>')
        html.append('</table></div>')
    (out/'index.html').write_text(''.join(html)+'</html>')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'replay', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--teacher-checkpoint', type=Path)
    print(json.dumps(audit(**vars(p.parse_args())), indent=2))
