import torch
import pytest
from rl2048.agents.afterstate_mlp import AfterstateMLP, AfterstateLearner, AfterstateMLPAgent
from research.afterstate_symmetry_screen import SymmetryAverage
from research.neural_afterstate_experiment import train_one


def test_frozen_eight_view_average_is_d4_invariant_without_changing_weights():
    torch.set_num_threads(1);torch.manual_seed(5)
    model=AfterstateMLP(width=8,input_encoding='embedding')
    # Nonzero asymmetric readout makes invariance a substantive test.
    torch.nn.init.normal_(model.layers[-1].weight)
    saved={k:v.clone() for k,v in model.state_dict().items()}
    board=torch.tensor([[1,2,3,4],[0,0,5,6],[1,7,8,9],[0,4,3,1]])
    ensemble=SymmetryAverage(model)
    target=ensemble(board.reshape(1,16))
    for k in range(4):
        for reflected in (False,True):
            view=board.flip(1) if reflected else board
            view=torch.rot90(view,k,(0,1)).reshape(1,16)
            torch.testing.assert_close(ensemble(view),target,atol=1e-6,rtol=1e-6)
    for k,v in model.state_dict().items():torch.testing.assert_close(v,saved[k])


def test_trained_symmetry_model_gradient_invariance_and_reload(tmp_path):
    torch.manual_seed(5);torch.set_num_threads(1)
    learner=AfterstateLearner(width=8,input_encoding='embedding',architecture='afterstate_sym_mlp')
    model=learner.policy
    assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in AfterstateMLP(width=8,input_encoding='embedding').parameters())
    torch.nn.init.normal_(model.layers[-1].weight)
    boards=torch.randint(0,10,(4,16))
    prediction=model(boards)
    prediction.sum().backward()
    assert model.embedding.weight.grad.abs().sum()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    for k in range(4):
        rotated=torch.rot90(boards.reshape(-1,4,4),k,(1,2)).reshape(-1,16)
        torch.testing.assert_close(model(rotated),prediction,atol=1e-6,rtol=1e-6)
    agent=AfterstateMLPAgent(learner);agent.save(tmp_path,dict(width=8))
    loaded=AfterstateMLPAgent.load(tmp_path)
    assert loaded.learner.policy.architecture=='afterstate_sym_mlp'
    torch.testing.assert_close(loaded.learner.policy(boards),prediction)


def test_explicit_symmetry_conversion_preserves_training_history(tmp_path):
    config=dict(algorithm='neural_afterstate',architecture='afterstate_mlp',
        width=8,depth=2,input_encoding='embedding',embedding_dim=4,n=1,reward_mode='score',
        gamma=1.,lr=1e-4,target_tau=.005,weight_decay=.01,envs=8,batch=8,
        capacity=128,warmup=8,epsilon_end=.05,epsilon_steps=500000,
        monitor_interval=32,monitor_games=8,monitor_seed_start=8900000,final_seed_start=8910000)
    first=train_one(config,5,32,tmp_path/'plain','cpu')
    changed=config|dict(resume=str(tmp_path/'plain'),architecture='afterstate_sym_mlp')
    with pytest.raises(ValueError,match='architecture'):
        train_one(changed,5,32,tmp_path/'rejected','cpu')
    changed['symmetry_warm_start_from_plain']=True
    second=train_one(changed,5,32,tmp_path/'symmetric','cpu')
    assert first['complete'] and second['complete']
    assert second['prior_transitions']==32 and second['total_transitions']==64
    assert first['parameter_count']==second['parameter_count']
