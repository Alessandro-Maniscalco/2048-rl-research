"""Save a game and inspect selected decisions against exact short lookahead.

Diagnostics do not prove the optimal action. Sampling emphasizes late failures
and lost corners, so disagreement rates are not estimates over all play.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from rl2048.agents.neural import NeuralAgent, tensor_boards
from rl2048.agents.ntuple import encode, row_tables
from rl2048.agents.deep_q_planning import DeepQPlanner
from rl2048.agents.q_planning import outcomes
from rl2048.game import ACTION_NAMES, move, legal_actions
from rl2048.vector_game import legal_masks
from rl2048.view import save_replay, board_html


def audit(checkpoint, out, seed, acting_depth=0):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    agent = NeuralAgent.load(checkpoint, 'cpu')
    metadata = json.loads((checkpoint / 'metadata.json').read_text())
    config = metadata.get('experiment', {})
    agent.save(out / 'checkpoint', config)
    settings=dict(gamma=config.get('gamma',.99),reward_mode=config.get('reward_mode','score'),
                  shaping_scale=config.get('shaping_scale',2.))
    actor=DeepQPlanner(agent,depth=1,network_batch=1024,**settings) if acting_depth else agent
    game = save_replay(actor, out / 'replay.html', seed=seed)
    frames = json.loads((out / 'replay.json').read_text())['frames']
    count = len(frames) - 1
    selected = set(np.linspace(0, count - 1, 8, dtype=int).tolist())
    selected.update(range(max(0, count - 8), count))
    lost_corners = [];lost_any_corners=[];first_large_corner_loss=None
    for i in range(count):
        board = np.array(frames[i]['board'])
        after, _, _ = move(board, frames[i+1]['action'])
        if board[3, 0] == board.max() and after[3, 0] != after.max():
            lost_corners.append(i)
        corners=lambda b:b[[0,0,3,3],[0,3,0,3]]
        if board.max()>=128 and board.max() in corners(board) and after.max() not in corners(after):
            lost_any_corners.append(i)
            if board.max()>=2048 and first_large_corner_loss is None:first_large_corner_loss=i
        if board.max()<2048<=after.max():selected.add(i)
    selected.update(lost_corners[-4:])
    selected.update(lost_any_corners[-4:])
    if first_large_corner_loss is not None:selected.add(first_large_corner_loss)
    planner = DeepQPlanner(agent, gamma=config.get('gamma', .99),
        reward_mode=config.get('reward_mode', 'score'), shaping_scale=config.get('shaping_scale', 2.),
        depth=2, network_batch=1024)
    records = []
    for i in sorted(selected):
        board = np.array(frames[i]['board']); encoded = encode(board)[None]
        mask = legal_actions(board)
        with torch.no_grad():
            q = agent.policy(tensor_boards(encoded, 'cpu')).numpy()[0]
        acting_q=actor.planned_values(encoded)[0] if acting_depth else np.where(mask,q,-np.inf)
        choice = int(acting_q.argmax())
        assert choice == frames[i+1]['action']
        planned = planner.planned_values(encoded)[0]
        search_choice = int(planned.argmax())
        successors, actions, probabilities, raw = outcomes(encoded.reshape(16), *row_tables())
        successor_masks = legal_masks(successors, *row_tables())
        alternatives = []
        for action in np.flatnonzero(mask):
            after, reward, _ = move(board, int(action))
            pick = actions == action
            assert np.isclose(probabilities[pick].sum(), 1)
            alternatives.append(dict(action=ACTION_NAMES[action], q=float(q[action]),
                acting_q=float(acting_q[action]),
                planned_q=float(planned[action]), merge_points=reward,
                afterstate=after.tolist(), empty_cells_before_spawn=int((after == 0).sum()),
                largest_in_bottom_left=bool(after[3, 0] == after.max()),
                immediate_death_probability=float(np.dot(probabilities[pick], ~successor_masks[pick].any(1))),
                expected_legal_next_actions=float(np.dot(probabilities[pick], successor_masks[pick].sum(1)))))
        records.append(dict(board_index=i, board=board.tolist(), chosen=ACTION_NAMES[choice],
            planner_choice=ACTION_NAMES[search_choice], disagreement=choice != search_choice,
            planner_gap=float(planned[search_choice]-planned[choice]), alternatives=alternatives))
    summary = dict(seed=seed, acting_depth=acting_depth, diagnostic_depth=2, game=game, checked_decisions=len(records),
        disagreements=sum(r['disagreement'] for r in records), lost_bottom_left_events=len(lost_corners),
        lost_any_corner_events=len(lost_any_corners),first_2048_corner_loss=first_large_corner_loss,
        interpretation='Selected-state diagnostics only. A two-step planner also has imperfect Q leaves; disagreement does not prove an error. Corner placement is diagnostic, not a rule.',
        source=str(checkpoint.resolve()))
    (out / 'audit.json').write_text(json.dumps(dict(summary=summary, decisions=records), indent=2, allow_nan=False))
    parts = []
    for r in sorted(records, key=lambda r: r['planner_gap'], reverse=True):
        parts.append(f'<h2>Board {r["board_index"]} → {r["chosen"]}; planner prefers {r["planner_choice"]}</h2>')
        parts.append(board_html(np.array(r['board']), frames[r['board_index']]['score'], 'Before the decision'))
        parts.append('<table><tr><th>Move</th><th>Model Q</th><th>Acting Q</th><th>Two-step Q</th><th>Merge points</th><th>Empty cells before spawn</th><th>Immediate game-over probability</th><th>Expected legal next actions</th></tr>')
        for a in r['alternatives']:
            parts.append(f'<tr><td>{a["action"]}</td><td>{a["q"]:.3f}</td><td>{a["acting_q"]:.3f}</td><td>{a["planned_q"]:.3f}</td><td>{a["merge_points"]}</td><td>{a["empty_cells_before_spawn"]}</td><td>{a["immediate_death_probability"]:.1%}</td><td>{a["expected_legal_next_actions"]:.2f}</td></tr>')
        parts.append('</table>')
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2048 · Policy decision audit</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:30px auto;padding:20px;background:#faf8ef;color:#544c44}table{width:100%;font-size:14px}td,th{text-align:left;padding:8px;border-bottom:1px solid #ccc}a{color:#197970}</style><h1>Inspecting actual policy decisions</h1><a href="replay.html">Watch this game</a><p>Regularly sampled boards, the final eight decisions, and up to four recent lost-corner events. All random spawns are probability-weighted in the counterfactuals. Two-step search is a diagnostic, not an oracle; these selected cases do not establish overall policy quality.</p>'''
    html += f'<p>Acting policy: {acting_depth}-step planning (0 means direct model). Game score: {game["score"]:,}. Checked {len(records)} decisions; {summary["disagreements"]} disagree with two-step search.</p>'
    (out / 'index.html').write_text(html + ''.join(parts) + '</html>')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=8930100)
    parser.add_argument('--acting-depth',type=int,choices=[0,1],default=0)
    args = parser.parse_args()
    audit(args.checkpoint, args.out, args.seed,args.acting_depth)
