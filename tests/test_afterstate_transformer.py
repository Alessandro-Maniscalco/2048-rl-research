import json
import numpy as np
import torch

from rl2048.agents.afterstate_mlp import AfterstateLearner, AfterstateMLPAgent
from rl2048.agents.afterstate_transformer import SymmetricAfterstateTransformer
from research.afterstate_teacher import digest
from research.teacher_transformer_experiment import (prepare_data, train_one,
    action_distillation_loss, stage_pools, sample_group_ids)


def test_transformer_symmetry_scaling_gradients_and_reload(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(7)
    learner=AfterstateLearner(architecture='afterstate_sym_transformer',width=8,
        embedding_dim=8,heads=2,depth=1,value_scale=3.,value_offset=10.)
    model=learner.policy
    boards=torch.arange(32).reshape(2,4,4)%8
    torch.testing.assert_close(model(boards),torch.full((2,),10.))
    with torch.no_grad():model.readout.weight.normal_(0,.1)
    expected=model(boards)
    for k in range(4):
        torch.testing.assert_close(model(torch.rot90(boards,k,(1,2))),expected,rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(model(torch.rot90(boards.flip(2),k,(1,2))),expected,rtol=1e-5,atol=1e-5)
    model(boards).square().mean().backward()
    assert model.tokenizer.weight.grad.abs().sum()>0
    assert model.blocks[0].self_attn.in_proj_weight.grad.abs().sum()>0
    assert model.blocks[0].linear1.weight.grad.abs().sum()>0
    agent=AfterstateMLPAgent(learner);agent.save(tmp_path/'model',dict(width=8))
    restored=AfterstateMLPAgent.load(tmp_path/'model');model.eval()
    torch.testing.assert_close(restored.learner.policy(boards),model(boards))
    assert restored.learner.policy.heads==2
    assert restored.learner.policy.value_scale.item()==3.


def test_teacher_only_pipeline_normalizes_fit_data_and_saves_for_online_td(tmp_path):
    torch.set_num_threads(1)
    folder=tmp_path/'data';folder.mkdir()
    rng=np.random.default_rng(11)
    after=rng.integers(0,5,(8,4,16),dtype=np.uint8)
    legal=np.ones((8,4),bool);legal[:,3]=False
    values=np.tile(np.array([1.,2.,3.,9999.],np.float32),(8,1))
    values[-2:,:3]+=10 # Held-out labels must not affect the normalization.
    data=dict(states=after[:,0],afterstates=after,values=values,legal=legal,gains=np.zeros((8,4),np.float32),
              validation=np.array([False]*6+[True]*2))
    np.savez(folder/'data.npz',**data)
    (folder/'manifest.json').write_text(json.dumps(dict(complete=True,gamma=1.,
        label_units='future raw merge points / 128',data_sha256=digest(folder/'data.npz'),
        teacher_sha256='fixture')))
    _,_,_,_,mean,scale=prepare_data(folder)
    assert mean==2. and scale==1.
    config=dict(algorithm='neural_afterstate',architecture='afterstate_sym_transformer',
        width=8,embedding_dim=8,heads=2,depth=1,input_encoding='embedding',gamma=1.,
        reward_mode='score',lr=.001,batch=8,teacher_dataset=str(folder),
        validation_interval=2,policy_monitor_interval=2,monitor_games=2,
        monitor_seed_start=11000000,final_seed_start=11001000,max_training_seconds=60)
    result=train_one(config,0,2,tmp_path/'run','cpu')
    assert result['complete'] and result['completed_budget']
    assert result['supervised_updates']==2 and result['training_transitions']==0
    assert not result['online_training']
    state=torch.load(tmp_path/'run/training.pt',weights_only=True)
    assert not state['optimizer']['state']
    for key in state['policy']:torch.testing.assert_close(state['policy'][key],state['target'][key])
    continued=train_one(config|dict(resume_distillation=str(tmp_path/'run'),grouped_actions=True,
        action_loss_weight=.05,action_temperature=4.,stage_balanced=True),1,2,tmp_path/'continued','cpu')
    assert continued['complete'] and continued['total_supervised_updates']==4
    assert continued['supervised_examples_seen']==2*8*3
    assert continued['fitting_stage_counts']==[6,0,0,0] # Held-out boards never enter the sampler.


def test_stage_sampling_equalizes_coverage_and_preserves_uniform_default():
    exponents=np.array([0,8,9,11,12,13,14,17],dtype=np.uint8)
    pools=stage_pools(np.repeat(exponents[:,None],16,axis=1))
    assert [p.tolist() for p in pools]==[[0,1],[2,3],[4,5],[6,7]]
    # A highly unequal fixture: each nonempty stage should receive equal mass.
    pools=[np.arange(90),np.arange(90,99),np.array([99]),np.array([],dtype=int)]
    ids=sample_group_ids(np.random.default_rng(7),100,30000,pools)
    counts=np.array([(ids<90).sum(),((ids>=90)&(ids<99)).sum(),(ids==99).sum()])
    np.testing.assert_allclose(counts/len(ids),1/3,atol=.01)
    np.testing.assert_array_equal(sample_group_ids(np.random.default_rng(7),100,32),
        np.random.default_rng(7).integers(100,size=32))


def test_action_distillation_matches_manual_cross_entropy_and_masks():
    legal=torch.tensor([[True,True,False],[False,False,False]])
    prediction=torch.zeros((2,3),requires_grad=True)
    # Teacher probabilities [.25,.75], with the second logit contributed by
    # known immediate reward rather than by U. Illegal large values are ignored.
    target=torch.tensor([[0.,0.,9999.],[0.,0.,0.]])
    gains=torch.tensor([[0.,128*np.log(3),9999.],[0.,0.,0.]],dtype=torch.float32)
    # Cancel the known reward on the student side to make its probabilities .5/.5.
    student=prediction-gains/128
    loss=action_distillation_loss(student,target,gains,legal,1.)
    torch.testing.assert_close(loss,torch.tensor(np.log(2),dtype=torch.float32))
    loss.backward()
    torch.testing.assert_close(prediction.grad,torch.tensor([[.25,-.25,0.],[0.,0.,0.]]))
    shifted=action_distillation_loss(student+100,target+100,gains,legal,1.)
    torch.testing.assert_close(shifted,loss)
