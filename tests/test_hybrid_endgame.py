import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from research.hybrid_endgame_study import summarize
from rl2048.agents.endgame import EndgameAgent
from rl2048.agents.frozen_tablebase import LIBRARY,ROOT
from rl2048.agents.hybrid_endgame import HybridEndgameAgent
from rl2048.game import legal_actions


def test_incomplete_results_cannot_be_reported_as_a_complete_mean():
    finished=dict(seed=1,hybrid=False,complete=True,score=100,elapsed_seconds=1,max_tile=16)
    timed_out=dict(seed=1,hybrid=True,complete=False,score=1000,elapsed_seconds=20,max_tile=64)
    result=summarize([finished,timed_out],[1])
    assert not result['complete'] and result['paired_difference'] is None
    assert result['arms']['hybrid']['mean_score'] is None
    assert result['incomplete_attempts']==1


@pytest.mark.skipif(not LIBRARY.exists(),reason='Build frozen table adapter first')
def test_hybrid_uses_exact_table_action_without_running_fallback_search():
    game=json.loads((ROOT/'runs/research/tablebase_search/validation_depth8_seed8949281/replay.json').read_text())
    board=np.array(game['frames'][24000]['board']);before=board.copy()
    agent=HybridEndgameAgent()
    try:
        with patch.object(EndgameAgent,'act',side_effect=AssertionError('Table hit must avoid compact search')):
            action=agent.act(board,legal_actions(board))
        assert action==game['frames'][24001]['action']
        assert agent.last_decision['mode']=='frozen-11' and agent.last_decision['nodes']==0
        np.testing.assert_array_equal(board,before)
    finally:agent.tables.close()
