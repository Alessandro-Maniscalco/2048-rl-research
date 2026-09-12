import numpy as np

from research.late_game_study import initialize,summarize
from rl2048.game import legal_actions


def test_saved_position_uses_reproducible_fresh_future_draws():
    case=dict(id=0,score=971000,board=[[0,0,0,2],[0,4,8,16],[2,8,16,32],[0,2,4,65536]])
    a,_,_=initialize(case,88);b,_,_=initialize(case,88)
    for _ in range(20):
        action=int(np.flatnonzero(legal_actions(a.board))[0])
        x=a.step(action);y=b.step(action)
        np.testing.assert_array_equal(x[0],y[0])
        assert x[1:4]==y[1:4]
    assert case['board'][3][3]==65536 and case['score']==971000


def test_conditional_summary_does_not_present_incomplete_mean():
    jobs=[dict(case={'id':i},future_seed=i*4+j,full_rank=f)
          for i in range(3) for j in range(4) for f in (False,True)]
    rows=[dict(case_id=j['case']['id'],future_seed=j['future_seed'],full_rank=j['full_rank'],
        complete=True,source_unchanged=True,all_conditional_transitions_and_future_rng_audited=True,
        additional_points=100+20*j['full_rank'],elapsed_seconds=1,max_tile=65536) for j in jobs]
    assert not summarize(rows[:-1],jobs)['complete']
    assert summarize(rows[:-1],jobs)['paired_mean_difference'] is None
    result=summarize(rows,jobs)
    assert result['complete'] and result['conditional'] and result['excluded_from_full_game_rankings']
    assert result['paired_mean_difference']==20
    assert len(result['per_case'])==3
