import json

import research.rollout_choice_study as study


def test_selector_uses_declared_horizon_and_prefers_recorded_action_on_tie(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'OUT',tmp_path)
    case=dict(id=0,recorded_action=1,board=[[2,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    for action in (1,2):
        for seed in range(16):
            folder=tmp_path/'discovery'/f'case0_action{action}_seed{seed}'
            folder.mkdir(parents=True)
            frames=[dict(score=0)]*10
            frames[8]=dict(score=10*action)
            frames[9]=dict(score=30)
            (folder/'trajectory.json').write_text(json.dumps(dict(frames=frames)))
    choice=study.choose([case])[0]
    assert choice['short_action']==2
    assert choice['long_action']==1


def test_independent_validation_compares_same_future_seed_returns():
    cases=[dict(id=0,source='test',source_move=1)]
    choices=[dict(recorded_action=1,short_action=2,long_action=3)]
    results=[dict(case_id=0,future_seed=s,first_action=a,additional_points=v)
        for s,vals in ((1,(10,20,30)),(2,(20,40,50))) for a,v in zip((1,2,3),vals)]
    out=study.compare(cases,choices,results)
    assert out['mean_difference_across_fixed_boards']==dict(long_minus_recorded=25.,short_minus_recorded=15.,long_minus_short=10.)
    assert out['cases'][0]['differences']['long_minus_short']['paired_bootstrap95']==[10.,10.]
