"""Inspect a fixed afterstate-policy game and all legal alternatives at selected moves.

One further exact chance backup uses this same imperfect learned value. It is
a counterfactual diagnostic, not an optimal-play oracle or a representative rate.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rl2048.afterstate_expectation import spawn_outcomes
from rl2048.agents.afterstate_mlp import AfterstateMLPAgent
from rl2048.agents.ntuple import encode, row_tables
from rl2048.game import ACTION_NAMES, move
from rl2048.vector_game import legal_masks
from rl2048.view import save_replay, board_html


def audit(checkpoint, out, seed=8930100, symmetry_average=False):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    agent = AfterstateMLPAgent.load(checkpoint, 'cpu')
    native_views = 8 if agent.learner.policy.architecture == 'afterstate_sym_mlp' else 1
    metadata = json.loads((checkpoint/'metadata.json').read_text())
    agent.save(out/'checkpoint', metadata['experiment'])
    if symmetry_average:
        from research.afterstate_symmetry_screen import SymmetryAverage
        agent.learner.policy=SymmetryAverage(agent.learner.policy)
        agent.learner.target=SymmetryAverage(agent.learner.target)
        agent.display_name+=' · eight-view value average'
    game = save_replay(agent, out/'replay.html', seed=seed)
    frames = json.loads((out/'replay.json').read_text())['frames']
    count = len(frames)-1
    selected = set(np.linspace(0, count-1, 8, dtype=int).tolist())
    selected.update(range(max(0, count-8), count))
    records = []
    for index in sorted(selected):
        board = np.array(frames[index]['board'])
        q, after, legal = agent.learner.decision(encode(board)[None])
        actions = np.flatnonzero(legal[0])
        chosen = int(q[0].argmax())
        assert chosen == frames[index+1]['action']
        afterstates = after[0, actions]
        expected, _, _ = agent.learner.expected_targets(afterstates)
        states, owners, probabilities = spawn_outcomes(afterstates)
        masks = legal_masks(states, *row_tables())
        alternatives = []
        for row, action in enumerate(actions):
            after_board, gain, _ = move(board, int(action))
            included = owners == row
            assert np.isclose(probabilities[included].sum(), 1.)
            alternatives.append(dict(action=ACTION_NAMES[action], q=float(q[0, action]),
                lookahead_q=float(gain/128 + agent.learner.gamma*expected[row]),
                merge_points=int(gain), afterstate=after_board.tolist(),
                empty_cells_before_spawn=int((after_board==0).sum()),
                immediate_death_probability=float(np.dot(probabilities[included], ~masks[included].any(1))),
                expected_legal_next_actions=float(np.dot(probabilities[included], masks[included].sum(1)))))
        expanded = max(alternatives, key=lambda a: a['lookahead_q'])['action']
        records.append(dict(board_index=index, board=board.tolist(), chosen=ACTION_NAMES[chosen],
                            lookahead_choice=expanded, disagreement=expanded!=ACTION_NAMES[chosen],
                            alternatives=alternatives))
    summary = dict(source=str(checkpoint.resolve()), seed=seed, game=game,
        inference_view_count=native_views * (8 if symmetry_average else 1),
        checked_decisions=len(records), disagreements=sum(r['disagreement'] for r in records),
        interpretation='Eight regular and final eight decisions, deduplicated. Exact probability-weighted extra spawn/slide backup uses the same imperfect network. Disagreement is not proof of an error; selected decisions are not an overall error-rate estimate.')
    (out/'audit.json').write_text(json.dumps(dict(summary=summary, decisions=records), indent=2, allow_nan=False))
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>2048 afterstate decision audit</title>',
        '<style>body{font:16px/1.6 system-ui;max-width:1100px;margin:30px auto;padding:20px;background:#faf8ef;color:#544c44}table{width:100%}td,th{padding:8px;text-align:left;border-bottom:1px solid #ccc}a{color:#197970}</style>',
        '<h1>Inspecting afterstate policy decisions</h1><a href="replay.html">Watch the complete fixed-seed game</a>',
        f'<p>Game score: {game["score"]:,}. {summary["interpretation"]}</p>']
    for record in records:
        parts += [f'<h2>Board {record["board_index"]}: {record["chosen"]}; extra lookahead prefers {record["lookahead_choice"]}</h2>',
            board_html(np.array(record['board']), frames[record['board_index']]['score'], 'Before the decision'),
            '<table><tr><th>Move</th><th>Current Q</th><th>Extra lookahead Q</th><th>Merge points</th><th>Empty cells</th><th>Immediate game-over probability</th></tr>']
        for a in record['alternatives']:
            parts.append(f'<tr><td>{a["action"]}</td><td>{a["q"]:.3f}</td><td>{a["lookahead_q"]:.3f}</td><td>{a["merge_points"]}</td><td>{a["empty_cells_before_spawn"]}</td><td>{a["immediate_death_probability"]:.1%}</td></tr>')
        parts.append('</table>')
    (out/'index.html').write_text(''.join(parts)+'</html>')
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=8930100)
    parser.add_argument('--symmetry-average',action='store_true')
    args = parser.parse_args()
    audit(args.checkpoint, args.out, args.seed,args.symmetry_average)
