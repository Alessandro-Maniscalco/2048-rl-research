"""Audit completed conditional controls and reproduce a saved search prefix."""
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from research.afterstate_teacher import digest, write
from research.cached_full_rank import CachedFullRankSearch
from research.late_game_study import initialize, fingerprints
from rl2048.agents.frozen_tablebase import FrozenTablebase
from rl2048.game import legal_actions

BASE = Path('runs/research/endgame_tablebase')


def audit_controls(tables):
    root = BASE / 'late_game_comparison'
    paths = sorted(root.glob('case*_compact/result.json'))
    assert len(paths) == 12, 'Wait for every compact continuation'
    reports = []
    for path in paths:
        result = json.loads(path.read_text())
        assert result['complete'] and result['all_conditional_transitions_and_future_rng_audited']
        folder = path.parent
        replay = json.loads((folder / 'replay.json').read_text())
        frames = replay['frames']
        decisions = json.loads((folder / 'decisions.json').read_text())
        audited = json.loads((folder / 'audit.json').read_text())
        assert digest(folder / 'replay.json') == audited['replay_sha256']
        max_other = 0
        boards_with_other_32768 = 0
        for f in frames:
            flat = [x for row in f['board'] for x in row]
            other = max((x for x in flat if x < 65536), default=0)
            max_other = max(other, max_other)
            boards_with_other_32768 += other >= 32768
        checks = []
        for i in range(max(0, len(decisions)-20), len(decisions)):
            d = decisions[i]
            board = np.array(frames[i]['board'])
            assert legal_actions(board)[d['action']]
            answer = tables.query(board)
            if d['mode'].startswith('frozen-'):
                assert d['mode'] == f'frozen-{answer["kind"]}'
                assert d['action'] == answer['action']
                assert d['table_probability'] == answer['probability']
            else:
                assert answer['kind'] == 0
            checks.append(dict(continuation_move=i+1, recorded=d, table_query=answer))
        assert not legal_actions(np.array(frames[-1]['board'])).any()
        reports.append(dict(folder=folder.name, replay_sha256=audited['replay_sha256'],
            length=result['length'], additional_points=result['additional_points'],
            max_tile_other_than_original_65536=max_other,
            frames_with_another_32768=boards_with_other_32768,
            final_board=frames[-1]['board'], terminal_tail_checks=checks))
    output = dict(complete=True, conditional=True, excluded_from_full_game_rankings=True,
        controls=12, completed_move_count=sum(r['length'] for r in reports),
        final_twenty_decisions_rechecked=sum(len(r['terminal_tail_checks']) for r in reports),
        largest_other_tile_counts=dict(Counter(r['max_tile_other_than_original_65536'] for r in reports)),
        none_reached_another_32768=all(r['frames_with_another_32768'] == 0 for r in reports),
        reports=reports,
        inference='These controls died before building a second32768 tile. The immediate '
            '32768+32768/65536+65536 representational limitation was not reached on these actual '
            'boards. Deeper imagined futures and heuristic rank scaling may still differ. '
            'This does not prove what caused each loss or compare incomplete candidate outcomes.')
    write(root / 'control_transition_audit.json', output)
    return {k: v for k, v in output.items() if k != 'reports'}


def reproduce_prefix(tables, folder=None):
    folder = Path(folder) if folder else BASE / 'late_full_rank_smoke'
    protocol = json.loads((folder / 'protocol.json').read_text())
    reference = json.loads((folder / 'replay.json').read_text())
    decisions = json.loads((folder / 'decisions.json').read_text())
    fingerprint = fingerprints()
    player = CachedFullRankSearch()
    env, board, info = initialize(protocol['case'], protocol['future_seed'])
    begin = time.perf_counter()
    records = []
    for i, d in enumerate(decisions):
        if time.perf_counter()-begin > 600:
            raise RuntimeError('Reproduction exceeded its ten-minute verification budget')
        table = tables.query(board)
        if table['kind']:
            action = table['action']
            assert d['mode'] == f'frozen-{table["kind"]}'
            assert d['table_probability'] == table['probability']
        else:
            query = player.query(board, 8)
            assert d['mode'] == 'full-rank-search'
            assert d['heuristic_value'] == query['answer']['heuristic_value']
            action = query['answer']['action']
            records.append(dict(move=i+1, **query))
        assert action == d['action'] and info['action_mask'][action]
        board, reward, terminal, truncated, info = env.step(action)
        expected = reference['frames'][i+1]
        assert np.array_equal(board, expected['board'])
        assert reward == expected['reward'] and env.score == expected['score']
        assert terminal == (i == len(decisions)-1 and reference['result']['complete'])
        assert not truncated
    source_complete = reference['result']['complete']
    output = dict(complete=True, conditional=True, explicitly_truncated_prefix=not source_complete,
        source_complete=source_complete, source=str(folder),
        source_replay_sha256=digest(folder / 'replay.json'),
        source_future_seed=protocol['future_seed'], moves=len(decisions),
        all_actions_values_states_spawns_rewards_match=True,
        full_rank_queries=len(records), table_hits=len(decisions)-len(records),
        prefix_seconds=time.perf_counter()-begin, additional_points=env.score-protocol['case']['score'],
        live_sources_unchanged=fingerprints() == fingerprint, queries=records,
        note='Exact reproduction of an existing saved-position continuation. '
            'No new full game, trained policy, score record or controlled full-loop speed comparison.')
    filename = 'complete_continuation_check.json' if source_complete else 'trajectory_check.json'
    write(BASE / 'cached_full_rank_probe' / filename, output)
    return {k: v for k, v in output.items() if k != 'queries'}


def main():
    tables = FrozenTablebase()
    try:
        print(json.dumps(audit_controls(tables)), flush=True)
        print(json.dumps(reproduce_prefix(tables)), flush=True)
    finally:
        tables.close()


if __name__ == '__main__':
    main()
