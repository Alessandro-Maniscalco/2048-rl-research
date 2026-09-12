"""Close the conditional rank comparison with terminal decision checks."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from research.afterstate_teacher import digest, write
from research.cached_full_rank import CachedFullRankSearch
from rl2048.agents.frozen_tablebase import FrozenTablebase
from rl2048.game import legal_actions


def main():
    root = Path('runs/research/endgame_tablebase/late_game_comparison')
    summary = json.loads((root/'summary.json').read_text())
    assert summary['complete'] and all(json.loads((root/'verified.json').read_text()).values())
    tables, search = FrozenTablebase(), CachedFullRankSearch()
    reports = []
    try:
        for folder in sorted(root.glob('case*_full')):
            result = json.loads((folder/'result.json').read_text())
            replay = json.loads((folder/'replay.json').read_text())
            frames = replay['frames']
            decisions = json.loads((folder/'decisions.json').read_text())
            original_audit = json.loads((folder/'audit.json').read_text())
            assert digest(folder/'replay.json') == original_audit['replay_sha256']
            assert result['complete'] and result['source_unchanged']
            assert result['all_conditional_transitions_and_future_rng_audited']
            maximum_other = max(max((x for row in f['board'] for x in row if x<65536),default=0) for f in frames)
            tail = []
            for i in range(max(0,len(decisions)-20),len(decisions)):
                board = np.asarray(frames[i]['board'])
                d = decisions[i]
                assert legal_actions(board)[d['action']]
                table = tables.query(board)
                if table['kind']:
                    assert d['mode'] == f'frozen-{table["kind"]}'
                    assert d['action'] == table['action'] and d['table_probability'] == table['probability']
                    answer = table
                else:
                    answer = search.query(board,8)['answer']
                    assert d['mode'] == 'full-rank-search' and d['action'] == answer['action']
                    assert d['heuristic_value'] == answer['heuristic_value']
                tail.append(dict(continuation_move=i+1,recorded=d,rechecked=answer))
            assert not legal_actions(np.asarray(frames[-1]['board'])).any()
            reports.append(dict(folder=folder.name,moves=result['length'],
                max_other_tile=maximum_other,final_board=frames[-1]['board'],tail=tail,
                source_replay_sha256=original_audit['replay_sha256']))
    finally:
        tables.close()
    assert len(reports)==12
    output = dict(complete=True, conditional=True, excluded_from_full_game_rankings=True,
        full_rank_moves=sum(r['moves'] for r in reports),tail_decisions_rechecked=sum(len(r['tail']) for r in reports),
        maximum_other_tile_counts=dict(Counter(r['max_other_tile'] for r in reports)),
        none_built_another_32768=all(r['max_other_tile']<32768 for r in reports),reports=reports,
        conclusion='No demonstrated benefit from replacing compact late search with this full-rank '
            'heuristic. Full-rank mean additional score is lower; only three starting-position '
            'clusters, so do not claim a universal ranking. Both arms failed before another32768. '
            'Close this score experiment without promotion. Keep exact caching as a compute tool for future work.')
    write(root/'full_rank_transition_audit.json',output)
    print(json.dumps({k:v for k,v in output.items() if k!='reports'}),flush=True)


if __name__=='__main__':
    main()
