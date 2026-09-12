from research.plateau import plateau_status


def curve(scores):
    return [dict(transitions=i*100,mean_score=s) for i,s in enumerate(scores)]


def test_plateau_minimum_patience_and_robust_spikes():
    flat=curve([100]*12)
    assert not plateau_status(flat,min_transitions=1200)['reached']
    assert plateau_status(flat,min_transitions=1000)['reached']
    flat[7]['mean_score']=10000 # one lucky checkpoint cannot reset patience
    assert plateau_status(flat,min_transitions=1000)['reached']


def test_improvement_resets_patience_and_small_gains_accumulate():
    increasing=curve([100+i for i in range(25)])
    assert not plateau_status(increasing,min_transitions=100)['reached']
    recovered=curve([100]*10+[120]*3)
    status=plateau_status(recovered,min_transitions=100)
    assert not status['reached'] and status['significant_best']==120


def test_plateau_stops_training_and_saves_actual_budget(tmp_path,monkeypatch):
    from research import transformer_td_experiment as experiment
    from rl2048.agents.neural import NeuralAgent
    monkeypatch.setattr(experiment,'evaluate_model',lambda *a,**kw:
        {'summary':{'mean_score':100.,'transitions':1},'episodes':[]})
    c=experiment.config('mlp_q','exponents',1)|dict(width=16,batch=32,warmup=128,
        monitor_initial=True,monitor_interval=128,save_best=True,
        plateau=dict(min_transitions=512,patience=2,window=1,min_gain=.02))
    result=experiment.train_one(c,0,2048,tmp_path/'run','cpu')
    assert result['stop_reason']=='policy_plateau'
    assert result['training_transitions']==512 and result['total_transitions']==512
    assert result['updates']==4 and not result['completed_budget']
    assert (tmp_path/'run/training.pt').exists()
    NeuralAgent.load(tmp_path/'run/best')
    NeuralAgent.load(tmp_path/'run/last')
