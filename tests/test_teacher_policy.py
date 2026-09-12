import json
import numpy as np
import torch

from research.afterstate_teacher import digest
from research.teacher_policy_experiment import policy_value_loss,train_one
from rl2048.agents.afterstate_transformer import SymmetricAfterstateTransformer
from rl2048.agents.teacher_policy_transformer import TeacherPolicyTransformer,copy_value_encoder
from rl2048.agents.neural import NeuralAgent
from rl2048.symmetry import transform_batch


def test_policy_equivariance_value_invariance_and_reload(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(4)
    model=TeacherPolicyTransformer(8,'embedding',1,2,10.,3.)
    with torch.no_grad():model.readout.weight.normal_(0,.1)
    boards=np.arange(64,dtype=np.uint8).reshape(4,16)%10
    expected=model(torch.tensor(boards))
    for reflected in (False,True):
        for rotation in range(4):
            batch=transform_batch(dict(states=boards,actions=np.arange(4)),rotation,reflected)
            output=model(torch.tensor(batch['states']))
            torch.testing.assert_close(output[:,:4][:,batch['actions']],expected[:,:4],atol=1e-5,rtol=1e-5)
            torch.testing.assert_close(output[:,4],expected[:,4],atol=1e-5,rtol=1e-5)
    expected.square().mean().backward()
    assert model.tokenizer.weight.grad.abs().sum()>0
    assert model.blocks[0].self_attn.in_proj_weight.grad.abs().sum()>0
    assert model.blocks[0].linear1.weight.grad.abs().sum()>0
    NeuralAgent(model,'teacher_policy_value',width=8).save(tmp_path/'saved')
    restored=NeuralAgent.load(tmp_path/'saved');model.eval()
    torch.testing.assert_close(restored.policy(torch.tensor(boards)),model(torch.tensor(boards)))
    assert restored.policy.value_offset.item()==10.
    assert restored.policy.value_scale.item()==3.


def test_separate_policy_loss_and_value_head_gradient():
    output=torch.zeros((2,5),requires_grad=True)
    legal=torch.tensor([[True,True,False,False],[False]*4])
    q=torch.tensor([[0.,np.log(3),9999.,9999.],[0.,0.,0.,0.]],dtype=torch.float32)
    ce=policy_value_loss(output,q,legal,1.,1.,0.)
    torch.testing.assert_close(ce,torch.tensor(np.log(2),dtype=torch.float32))
    ce.backward()
    torch.testing.assert_close(output.grad,torch.tensor([[.25,-.25,0.,0.,0.],[0.,0.,0.,0.,0.]]))
    torch.testing.assert_close(policy_value_loss(output,q+100,legal,1.,1.,0.),ce)
    loss=policy_value_loss(output,q,legal,2.,1.,.5)
    expected=np.log(2)+.5*(np.log(3)/2)**2
    torch.testing.assert_close(loss,torch.tensor(expected,dtype=torch.float32))


def test_encoder_transfer_and_complete_teacher_policy_training(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(7)
    old=SymmetricAfterstateTransformer(8,'embedding',1,8,2)
    new=TeacherPolicyTransformer(8,'embedding',1,2)
    initial_head=new.readout.weight.clone();copy_value_encoder(new,old.state_dict())
    torch.testing.assert_close(new.positions,old.positions)
    torch.testing.assert_close(new.blocks[0].linear1.weight,old.blocks[0].linear1.weight)
    torch.testing.assert_close(new.readout.weight,initial_head)
    folder=tmp_path/'data';folder.mkdir()
    rng=np.random.default_rng(6)
    legal=np.array([[True,True,False,False]]*8)
    values=np.array([[1.,2.,999.,999.]]*8,dtype=np.float32)
    values[-2:,:2]+=100 # Excluded from normalization and fitting.
    data=dict(states=rng.integers(0,5,(8,16),dtype=np.uint8),
        afterstates=rng.integers(0,5,(8,4,16),dtype=np.uint8),values=values,legal=legal,
        gains=np.zeros((8,4),np.float32),validation=np.array([False]*6+[True]*2))
    np.savez(folder/'data.npz',**data)
    (folder/'manifest.json').write_text(json.dumps(dict(complete=True,gamma=1.,
        label_units='future raw merge points / 128',data_sha256=digest(folder/'data.npz'),teacher_sha256='fixture')))
    c=dict(algorithm='teacher_policy_value',width=8,input_encoding='embedding',depth=1,heads=2,
        teacher_dataset=str(folder),batch=8,lr=.001,weight_decay=.01,action_temperature=4.,
        value_loss_weight=.05,monitor_seed_start=11900000,final_seed_start=11901000,
        monitor_games=2,validation_interval=2,policy_monitor_interval=2,max_training_seconds=60)
    result=train_one(c,0,2,tmp_path/'run','cpu')
    assert result['complete'] and result['completed_budget']
    assert result['value_offset']==2. and result['value_scale']==1.
    assert result['training_transitions']==0 and result['supervised_updates']==2
    assert (tmp_path/'run/replay_best.html').exists()
    assert json.loads((tmp_path/'run/evaluation.json').read_text())['summary']['episodes']==100
    assert torch.load(tmp_path/'run/distillation_training.pt',weights_only=True)['optimizer']['state']
    # Continue the exact model/Adam state on fitting boards only. Constants
    # remain those of the full fitting partition, and no encoder reset occurs.
    continuation=train_one(c|dict(resume_distillation=str(tmp_path/'run'),diagnostic_only=True,
        fit_subset_size=4,fit_monitor_boards=4),1,2,tmp_path/'continued','cpu')
    assert continuation['complete'] and continuation['total_supervised_updates']==4
    assert continuation['fitting_boards']==4 and continuation['available_fitting_boards']==6
    assert continuation['value_offset']==2.
    fitted=json.loads((tmp_path/'continued/fit_curve.json').read_text())
    assert fitted and fitted[-1]['policy_kl']>=-1e-5
    assert json.loads((tmp_path/'continued/fitting_subset.json').read_text())['indices_in_full_fitting_array']
    stopped=train_one(c|dict(diagnostic_only=True,fit_subset_size=4,fit_kl_stop=1e6,
        fit_agreement_stop=0.),2,2,tmp_path/'fit_goal','cpu')
    assert stopped['completed_fit_goal'] and stopped['completed_budget']
    assert stopped['supervised_updates']==0 and stopped['stop_reason']=='diagnostic_fit_goal'
