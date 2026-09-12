"""Inspect the first 65,536 merge and re-query all subsequent table decisions.

This reads a completed, already seed-audited game. It never changes the player
or reruns its timing-dependent search. Finite-horizon diagnostics are separate
from both the recorded game and the frozen evaluation.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from research.afterstate_teacher import digest, write
from research.exact_endgame_diagnostic import ExactEndgame
from research.hybrid_endgame_study import policy_fingerprints
from rl2048.agents.frozen_tablebase import FrozenTablebase
from rl2048.game import ACTION_NAMES, move


def inspect(folder):
    folder = Path(folder)
    replay_path = folder / 'replay.json'
    replay = json.loads(replay_path.read_text())
    audit = json.loads((folder / 'audit.json').read_text())
    result = json.loads((folder / 'result.json').read_text())
    decisions = json.loads((folder / 'decisions.json').read_text())
    assert result['complete'] and result['terminated'] and not result['truncated']
    assert audit['exact_seeded_spawn_sequence_rechecked'] and audit['every_transition_rechecked']
    assert digest(replay_path) == audit['replay_sha256']
    frames = replay['frames']
    first = next(i for i, frame in enumerate(frames)
                 if np.max(frame['board']) >= 65536)
    before = np.asarray(frames[first - 1]['board'], dtype=np.int64)
    after, reward, changed = move(before, decisions[first - 1]['action'])
    assert changed and int(after.max()) == 65536
    assert reward == frames[first]['reward']
    hashes = policy_fingerprints()
    tables = FrozenTablebase()
    checks = Counter()
    try:
        for i in range(first, len(decisions)):
            recommendation = tables.query(frames[i]['board'])
            saved = decisions[i]
            if saved['mode'].startswith('frozen-'):
                assert recommendation['kind'] == int(saved['mode'].split('-')[1])
                assert recommendation['action'] == saved['action']
                assert abs(recommendation['probability'] - saved['table_probability']) < 1e-12
                checks[saved['mode']] += 1
            else:
                assert recommendation['kind'] == 0
                checks['confirmed_table_miss'] += 1
    finally:
        tables.close()
    assert policy_fingerprints() == hashes

    diagnostics = []
    for sample in audit['decisions']:
        legal_risks = [v for v in sample['risks'].values() if v is not None]
        if sample['risks'][sample['actual_action']] <= min(legal_risks) + 1e-12:
            continue
        solver = ExactEndgame(max_nodes=500000, seconds=30)
        horizons = []
        error = None
        try:
            for horizon in (1, 2, 4, 8):
                try:
                    values = solver.query(sample['board'], horizon)
                except RuntimeError as exc:
                    error = str(exc)
                    break
                horizons.append(dict(horizon=horizon, actions=values))
            diagnostics.append(dict(board_index=sample['board_index'], board=sample['board'],
                recorded_action=sample['actual_action'], mode=sample['saved']['mode'],
                immediate_risks=sample['risks'], horizons=horizons,
                complete_eight=error is None, error=error, states=solver.nodes,
                seconds=time.perf_counter()-solver.started))
        finally:
            solver.legal.cache_clear()
            solver.value.cache_clear()
    output = dict(seed=result['seed'], score=result['score'], max_tile=result['max_tile'],
        replay_sha256=audit['replay_sha256'], exact_seed_audit_present=True,
        first_65536_move=first, score_before=frames[first-1]['score'],
        score_after=frames[first]['score'], actual_merge_reward=reward,
        actual_action=ACTION_NAMES[decisions[first-1]['action']], before=before.tolist(),
        after_move_before_spawn=after.tolist(), after_spawn=frames[first]['board'],
        saved_decision=decisions[first-1], subsequent_moves=len(decisions)-first,
        subsequent_points=result['score']-frames[first]['score'],
        post_65536_table_rechecks=dict(checks), policy_unchanged=True,
        higher_immediate_risk_diagnostics=diagnostics,
        interpretation='The real game retains 65536 exactly. The compact search may abstract '
            'large ranks only in its input; frozen-table queries use actual ranks. '
            'Re-querying verifies cache recommendations, not adaptive search reproducibility. '
            'Short-horizon expected points are not full-game values; a higher immediate risk '
            'is not automatically a mistake. This selected game is not an average.')
    write(folder / 'high_tile_audit.json', output)
    print(json.dumps({k:v for k,v in output.items() if k not in
                     ('higher_immediate_risk_diagnostics', 'before', 'after_spawn',
                      'after_move_before_spawn', 'saved_decision')}), flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    inspect(parser.parse_args().folder)
