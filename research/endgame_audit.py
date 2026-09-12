"""Audit completed compact-engine games without rerunning adaptive searches."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from research.afterstate_teacher import digest, write
from research.endgame_continue import restore_environment
from research.endgame_experiment import ROOT
from research.native_policy_data import extract_game
from rl2048.agents.spawn_safety import immediate_death_risks
from rl2048.game import ACTION_NAMES


def audit(folder):
    folder = Path(folder)
    replay = json.loads((folder/'replay.json').read_text())
    result = json.loads((folder/'result.json').read_text())
    if not result['complete']:
        raise ValueError('Audit completed games; retain unfinished attempts separately')
    boards, actions, legal, _ = extract_game(folder/'replay.json')
    env, _, _, done = restore_environment(replay)
    if not done or env.score != result['score']:
        raise ValueError('Exact seeded replay does not match the completed result')
    protocol = json.loads((folder/'protocol.json').read_text())
    unchanged = all((ROOT/name).exists() and digest(ROOT/name)==value
                    for name,value in protocol['fingerprints'].items())
    if not unchanged:
        raise ValueError('Planner source/binary fingerprint changed')
    decisions = json.loads((folder/'decisions.json').read_text())
    if len(decisions) != len(actions) or any(d['action'] != int(a) for d,a in zip(decisions, actions)):
        raise ValueError('Saved decisions differ from executed actions')
    risks, risk_legal = immediate_death_risks(boards)
    np.testing.assert_array_equal(risk_legal, legal)
    chosen = risks[np.arange(len(actions)), actions]
    avoidable = np.flatnonzero(chosen > risks.min(1) + 1e-12)
    ranks = boards.max(1)
    increases = np.flatnonzero((ranks[1:] > ranks[:-1]) & (ranks[1:] >= 14)) + 1
    selected = set(np.linspace(0, len(actions)-1, 16, dtype=int))
    selected.update(range(max(0, len(actions)-12), len(actions)))
    selected.update(int(i) for i in avoidable)
    for i in increases:
        selected.update(range(max(0, int(i)-2), min(len(actions), int(i)+2)))
    samples = []
    for i in sorted(selected):
        samples.append(dict(board_index=int(i), board=replay['frames'][i]['board'],
            actual_action=ACTION_NAMES[int(actions[i])],
            risks={name:float(risks[i,a]) if legal[i,a] else None for a,name in enumerate(ACTION_NAMES)},
            score_before=replay['frames'][i]['score'],
            actual_reward=replay['frames'][i+1]['reward'], saved=decisions[i]))
    seconds = np.array([d['seconds'] for d in decisions])
    summary = dict(complete=True, seed=result['seed'], score=result['score'], moves=len(actions),
        max_tile=result['max_tile'], every_transition_rechecked=True,
        exact_seeded_spawn_sequence_rechecked=True, unchanged_fingerprints=unchanged,
        replay_sha256=digest(folder/'replay.json'),
        planner_timing_state_reset=result.get('planner_timing_state_reset',False),
        modes=dict(Counter(d['mode'] for d in decisions)),
        avoidable_immediate_risk_decisions=len(avoidable),
        positive_risk_decisions=int((chosen > 0).sum()),
        high_rank_abstracted_decisions=sum(d['high_rank_abstraction'] for d in decisions),
        search_input_double32768_changes=sum(d['two32768_search_downgrade'] for d in decisions),
        decision_seconds_median=float(np.median(seconds)),
        decision_seconds_p95=float(np.quantile(seconds,.95)),
        saved_node_count_sum=sum(d['nodes'] for d in decisions),
        note='Risk is exact next-spawn death probability, not eventual failure probability. '
            'Zero avoidable immediate risk does not prove optimal long-term decisions. '
            'Adaptive searches are not rerun; actual recorded actions and exact RNG trajectory are checked.')
    write(folder/'audit.json', dict(**summary, decisions=samples))
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders',nargs='+',type=Path)
    args=parser.parse_args()
    for folder in args.folders:
        audit(folder)
