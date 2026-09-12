import json
from pathlib import Path

import numpy as np

from research.tactical_hybrid import tactical_choice
from rl2048.game import ACTION_NAMES


def test_profitable_risky_move_is_preserved():
    board=np.array([[256,1024,4096,32768],[64,512,2048,8192],[8,16,32,128],[2,2,8,2]])
    action,check=tactical_choice(board,ACTION_NAMES.index('right'))
    assert action==ACTION_NAMES.index('right') and check['complete'] and not check['overrode']
    assert check['values']['right']['expected_additional_raw_score']>100
    assert check['values']['left']['expected_additional_raw_score']<55


def test_bad_table_recommendation_changes_only_after_completed_calculation():
    path=Path('runs/research/endgame_tablebase/hybrid_comparison/risk_horizon_diagnostic.json')
    cases=json.loads(path.read_text())['results']
    case=next(c for c in cases if c['seed']==8973000)
    proposed=ACTION_NAMES.index(case['actual_action'])
    action,check=tactical_choice(case['board'],proposed)
    assert check['complete'] and check['overrode'] and action==ACTION_NAMES.index('up')
    action,check=tactical_choice(case['board'],proposed,max_nodes=1)
    assert action==proposed and not check['complete'] and not check['overrode']
