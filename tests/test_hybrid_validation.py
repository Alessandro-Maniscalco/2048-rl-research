from research.hybrid_validation import summarize


def test_no_mean_until_all_prespecified_seeds_finish():
    seeds=list(range(100))
    games=[dict(seed=s,complete=True,score=640000,elapsed_seconds=500,length=22000,max_tile=32768) for s in seeds]
    assert summarize(games[:-1],seeds,1)['mean_score'] is None
    games[-1]['complete']=False
    assert not summarize(games,seeds,1)['complete']
    assert summarize(games,seeds,1)['mean_score'] is None
    games[-1]['complete']=True;games[-1]['seed']=101
    assert summarize(games,seeds,1)['mean_score'] is None
    games[-1]['seed']=99
    result=summarize(games,seeds,1)
    assert result['complete'] and result['mean_score']==640000
    assert result['mean_score_bootstrap95']==[640000,640000]
    assert result['total_transitions']==2200000
