"""Locate paired divergence and independently recompute recorded interventions."""
import json
from pathlib import Path

import numpy as np

from research.afterstate_teacher import write
from research.tactical_hybrid import tactical_choice
from rl2048.game import ACTION_NAMES


def main():
    out=Path('runs/research/endgame_tablebase/tactical_comparison')
    assert json.loads((out/'summary.json').read_text())['complete']
    assert all(json.loads((out/'verified.json').read_text()).values())
    pairs=[]
    checks=[]
    for seed in range(8976000,8976012):
        folders=[out/f'{name}_seed{seed}' for name in ('control','tactical')]
        decisions=[json.loads((p/'decisions.json').read_text()) for p in folders]
        replays=[json.loads((p/'replay.json').read_text()) for p in folders]
        audits=[json.loads((p/'audit.json').read_text()) for p in folders]
        assert all(a['exact_seeded_spawn_sequence_rechecked'] and a['every_transition_rechecked'] for a in audits)
        first_difference=next((i for i,(a,b) in enumerate(zip(*decisions)) if a['action']!=b['action']),None)
        overrides=[i for i,d in enumerate(decisions[1]) if d.get('tactical',{}).get('overrode')]
        first_override=min(overrides,default=None)
        prefix=(first_difference is not None and
                replays[0]['frames'][:first_difference+1]==replays[1]['frames'][:first_difference+1])
        first_65536=next((i for i,f in enumerate(replays[1]['frames']) if np.max(f['board'])>=65536),None)
        pairs.append(dict(seed=seed,first_different_action_board_index=first_difference,
            first_tactical_override_board_index=first_override,identical_board_prefix=prefix,
            divergence_before_override=first_difference is not None and
                (first_override is None or first_difference<first_override),
            first_divergence_modes=[d[first_difference]['mode'] for d in decisions] if first_difference is not None else None,
            first_divergence_depths=[d[first_difference]['depth'] for d in decisions] if first_difference is not None else None,
            first_65536_frame=first_65536,
            reached_65536_before_override=first_65536 is not None and
                (first_override is None or first_65536<=first_override)))
        for i in overrides:
            saved=decisions[1][i]['tactical']
            board=replays[1]['frames'][i]['board']
            action,recomputed=tactical_choice(board,saved['proposed'])
            assert recomputed['complete'] and recomputed['overrode'] and action==saved['selected']
            for name,values in saved['values'].items():
                for key,value in values.items():
                    assert np.isclose(recomputed['values'][name][key],value,rtol=0,atol=1e-8)
            selected=ACTION_NAMES[action]
            original=ACTION_NAMES[saved['proposed']]
            checks.append(dict(seed=seed,board_index=i,board=board,original_action=original,
                selected_action=selected,mode_of_original_proposal=decisions[1][i]['proposal']['mode'],
                exact_values_matched=True,original_expected_eight=saved['values'][original]['expected_additional_raw_score'],
                selected_expected_eight=saved['values'][selected]['expected_additional_raw_score'],
                original_optimal_eight_survival=saved['values'][original]['probability_still_alive_after_h_moves'],
                selected_optimal_eight_survival=saved['values'][selected]['probability_still_alive_after_h_moves'],
                actual_remaining_moves=len(decisions[1])-i,
                actual_remaining_points=replays[1]['frames'][-1]['score']-replays[1]['frames'][i]['score']))
    result=dict(complete=True,pairs=pairs,overrides=checks,overrides_recomputed=len(checks),
        pairs_diverging_before_any_override=sum(p['divergence_before_override'] for p in pairs),
        note='All full replays were seed-audited. Same seeds do not isolate tactical causality when '
            'time-dependent search already chooses different actions before intervention. '
            'Recomputed eight-move values verify the intended calculation, not full-game optimality. '
            'Optimal survival is a separate objective and may choose different future actions. '
            'Actual remaining points are one realized trajectory, not an action-value estimate.')
    write(out/'transition_diagnostic.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('pairs','overrides')}),flush=True)


if __name__=='__main__':main()
