import numpy as np
from rl2048.rewards import learning_rewards, positional_potential


def test_corner_snake_orientation_and_strength():
    # Same tiles: putting the larger tile bottom-left improves both potentials.
    poor=np.zeros((1,16),np.uint8);poor[0,0]=10;poor[0,12]=1
    good=np.zeros((1,16),np.uint8);good[0,0]=1;good[0,12]=10
    assert positional_potential(good,'corner_snake')[0]>positional_potential(poor,'corner_snake')[0]
    raw=np.array([0.],np.float32);done=np.array([False])
    weak=learning_rewards(raw,poor,good,done,.99,'corner_snake',scale=2.)
    strong=learning_rewards(raw,poor,good,done,.99,'corner_snake',scale=10.)
    assert weak[0]>0
    np.testing.assert_allclose(strong,5*weak)


def test_shaping_telescopes_and_terminal_reward_is_retained():
    boards=np.array([[1]*16,[2]*16,[3]*16,[4]*16],np.uint8)
    raw=np.array([0.,8.,16.],np.float32);done=np.array([False,False,True])
    gamma=.99;scale=2.
    rewards=learning_rewards(raw,boards[:-1],boards[1:],done,gamma,'corner_snake',scale)
    differences=rewards-raw/128
    np.testing.assert_allclose(np.dot(gamma**np.arange(3),differences),
        -scale*positional_potential(boards[:1],'corner_snake')[0],rtol=1e-5)
    assert rewards[-1]==np.float32(raw[-1]/128-scale*positional_potential(boards[2:3],'corner_snake')[0])


def test_training_loop_passes_configured_shaping_strength(tmp_path,monkeypatch):
    from research import transformer_td_experiment as experiment
    received=[]
    original=experiment.learning_rewards
    def capture(*args,**kwargs):
        received.append(kwargs['scale'])
        return original(*args,**kwargs)
    monkeypatch.setattr(experiment,'learning_rewards',capture)
    monkeypatch.setattr(experiment,'evaluate_model',lambda *a,**kw:
        {'summary':{'mean_score':0.,'transitions':1},'episodes':[]})
    c=experiment.config('mlp_q','exponents',3,'corner_snake')|dict(width=16,shaping_scale=2.)
    result=experiment.train_one(c,0,512,tmp_path/'run','cpu')
    assert result['completed_budget'] and received==[2.]*4
