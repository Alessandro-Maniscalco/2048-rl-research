"""Recompute selected rollout decisions; verification only, not new samples."""
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from research.afterstate_teacher import digest, write
from research.hybrid_endgame_study import BASE
from research.late_game_study import initialize
from research.long_rollout_probe import fingerprints
from research.work_budget_hybrid import WorkBudgetHybridAgent


def main():
    # Two opposed choices in the failed selection case, and the first future
    # (by seed) that rebuilt a 32768 after the original record's final position.
    paths = [BASE/f'rollout_choice_study/validation/case0_action{a}_seed9010100' for a in (0, 1)]
    root = BASE/'long_rollout_validation'
    for path in sorted(root.glob('case1_action1_seed*/trajectory.json')):
        frames = json.loads(path.read_text())['frames']
        if any(32768 in np.asarray(f['board']) for f in frames):
            paths.append(path.parent)
            break
    assert len(paths) == 3
    hashes = fingerprints()
    rows = []
    for folder in paths:
        start = time.perf_counter()
        replay_hash = digest(folder/'trajectory.json')
        p = json.loads((folder/'protocol.json').read_text())
        replay = json.loads((folder/'trajectory.json').read_text())
        env, board, info = initialize(p['case'], p['future_seed'])
        agent = WorkBudgetHybridAgent(1000000)
        modes = Counter()
        try:
            for i, (frame, decision) in enumerate(zip(replay['frames'][1:], replay['decisions'])):
                if i == 0:
                    action = p['first_action']
                    modes['forced_first_action'] += 1
                else:
                    action = agent.act(board, info['action_mask'])
                    for key in ('action', 'mode', 'depth', 'nodes'):
                        assert agent.last_decision.get(key) == decision.get(key), (folder.name, i, key)
                    modes[agent.last_decision['mode']] += 1
                assert action == frame['action']
                board, reward, done, truncated, info = env.step(action)
                assert not truncated and np.array_equal(board, frame['board'])
                assert reward == frame['reward'] and env.score == frame['score']
            assert done == replay['result']['terminated']
            assert len(replay['decisions']) == len(replay['frames'])-1
        finally:
            agent.tables.close()
        assert digest(folder/'trajectory.json') == replay_hash
        rows.append(dict(folder=str(folder.relative_to(BASE)), moves=len(replay['decisions']),
            trajectory_sha256=replay_hash, every_action_mode_depth_node_and_transition_matched=True,
            modes=dict(modes), elapsed_seconds=time.perf_counter()-start))
    assert fingerprints() == hashes
    write(BASE/'rollout_choice_study/policy_recheck.json', dict(complete=True, rows=rows,
        unchanged_policy=True, verification_only=True,
        note='Repeated three already counted conditional trajectories from fresh controllers. '
             'Not new random samples or complete-game score evidence.'))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
