from copy import deepcopy

from research.long_rollout_probe import summarize


def sample_data():
    case=dict(id=0,source_move=8,recorded_action=1,
        board=[[2,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    jobs=[dict(case=case,future_seed=s,first_action=a) for s in (1,2) for a in (1,2)]
    results=[dict(case_id=0,future_seed=s,first_action=a,valid_horizon=True,
        exact_seeded_transitions_audited=True,additional_points=10+20*(s-1)+10*(a-1),
        length=512 if s==1 else 20,terminated=s==2,horizon_capped=s==1,new_larger_tile=False)
        for s in (1,2) for a in (1,2)]
    return case,jobs,results


def test_horizon_caps_are_valid_but_resource_failures_block_means():
    case,jobs,results=sample_data()
    assert summarize(results,jobs,[case])['complete']
    assert not summarize(results[:-1],jobs,[case])['complete']
    failed=deepcopy(results);failed[0]['valid_horizon']=False
    result=summarize(failed,jobs,[case])
    assert not result['complete'] and result['cases']==[]


def test_raw_returns_and_paired_seed_differences_match_hand_calculation():
    case,jobs,results=sample_data()
    s=summarize(results,jobs,[case])
    rows=s['cases'][0]['actions']
    assert [r['mean_points'] for r in rows]==[20,30]
    assert rows[1]['paired_difference_from_recorded']==10
    assert rows[1]['paired_bootstrap95']==[10,10]
    assert rows[0]['terminated']==1 and rows[0]['horizon_capped']==1
    assert s['cases'][0]['largest_sample_mean_action']=='down'
