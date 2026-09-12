import json
import pytest
import torch
from rl2048.agents.qr_dqn import QuantileTransformer,QRDQN,quantile_targets,quantile_huber_loss
from rl2048.agents.transformer_q import TransformerQNetwork
from rl2048.agents.neural import NeuralAgent
from rl2048.offline_train import training_state


def test_pairwise_quantile_loss_by_hand_and_target_stop_gradient():
    prediction=torch.tensor([[0.,2.]],requires_grad=True)
    target=torch.tensor([[1.,3.]],requires_grad=True)
    # Pair losses: [.125,.625; .125,.375], averaged = .3125.
    loss=quantile_huber_loss(prediction,target)
    torch.testing.assert_close(loss,torch.tensor(.3125))
    loss.backward()
    torch.testing.assert_close(prediction.grad,torch.tensor([[-.125,-.125]]))
    assert target.grad is None


def test_double_selection_uses_mean_legal_value_and_keeps_quantiles():
    online=torch.tensor([[[0.,100.],[9.,11.],[8.,8.],[0.,0.]]]*3)
    target=torch.tensor([[[999.,999.],[2.,6.],[10.,20.],[0.,0.]]]*3)
    mask=torch.tensor([[False,True,True,False],[False]*4,[False,True,True,False]])
    result=quantile_targets(torch.tensor([1.,7.,1.]),torch.tensor([False,True,False]),
        online,target,mask,torch.tensor([.5,.9,.125]))
    torch.testing.assert_close(result,torch.tensor([[2.,4.],[7.,7.],[1.25,1.75]]))
    ordinary=quantile_targets(torch.tensor([1.]*3),torch.zeros(3,dtype=torch.bool),
        online,target,mask,.5,double=False)
    torch.testing.assert_close(ordinary[0],torch.tensor([6.,11.]))


@pytest.mark.parametrize('encoding',['exponents','relative','embedding'])
def test_quantile_shapes_initial_control_match_entropy_and_roundtrip(tmp_path,encoding):
    torch.manual_seed(7);scalar=TransformerQNetwork(16,encoding,2,4)
    torch.manual_seed(7);model=QuantileTransformer(16,encoding,2,4,5)
    board=torch.randint(0,12,(3,16))
    assert model.quantile_values(board).shape==(3,4,5)
    torch.testing.assert_close(model(board),scalar(board))
    torch.testing.assert_close(model(board),model.forward_with_entropy(board)[0])
    agent=NeuralAgent(model,'qr_dqn',width=16);agent.save(tmp_path)
    loaded=NeuralAgent.load(tmp_path)
    assert loaded.algorithm=='qr_dqn' and loaded.policy.num_quantiles==5
    torch.testing.assert_close(loaded.policy.quantile_values(board),model.quantile_values(board))


def test_real_update_separates_quantiles_and_restores_optimizer():
    torch.manual_seed(1);learner=QRDQN(width=16,heads=4,num_quantiles=5)
    b=torch.randint(0,12,(8,16))
    batch=dict(states=b,next_states=b,actions=torch.arange(8)%4,rewards=torch.ones(8),
        terminated=torch.ones(8,dtype=torch.bool),next_masks=torch.zeros(8,4,dtype=torch.bool))
    before=learner.policy.quantile_values(b).detach().clone()
    metrics=learner.update(batch)
    assert all(torch.isfinite(v) for v in metrics.values())
    after=learner.policy.quantile_values(b).detach()
    assert not torch.equal(before,after)
    assert after.std(-1).max()>0
    assert learner.policy.positions.grad.abs().sum()>0
    assert all(p.grad is None for p in learner.target.parameters())
    restored=QRDQN(width=16,heads=4,num_quantiles=5)
    for key,state in training_state(learner).items():getattr(restored,key).load_state_dict(state)
    assert len(restored.optimizer.state)>0
    torch.testing.assert_close(restored.policy(b),learner.policy(b))


def test_qr_training_loop_resume_metadata_and_replay(tmp_path):
    from research.transformer_td_experiment import train_one,config
    from rl2048.view import save_replay
    c=config('transformer_q','exponents',3)|dict(algorithm='qr_dqn',width=16,heads=4,
        num_quantiles=5,batch=32,warmup=128,monitor_games=2,monitor_interval=512,
        monitor_initial=True,save_best=True,final_seed_start=8910000)
    r=train_one(c,0,512,tmp_path/'first','cpu')
    assert r['completed_budget'] and r['updates']==2
    a=NeuralAgent.load(tmp_path/'first/last');assert a.algorithm=='qr_dqn'
    continued=train_one(c|dict(resume=str(tmp_path/'first')),0,512,tmp_path/'continued','cpu')
    assert continued['total_transitions']==1024
    game=save_replay(a,tmp_path/'replay.html',seed=8930001)
    assert game['terminated'] and not game['truncated']
    assert 'QR_DQN' in (tmp_path/'replay.html').read_text()
