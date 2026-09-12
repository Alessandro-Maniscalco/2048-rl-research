import json
import numpy as np
import torch

from research.teacher_ppo_experiment import reset_critic,train_one
from rl2048.agents.teacher_policy_transformer import TeacherPolicyTransformer
from rl2048.agents.neural import NeuralAgent


def test_critic_reset_preserves_actor_exactly():
    torch.set_num_threads(1);torch.manual_seed(12)
    model=TeacherPolicyTransformer(8,'embedding',1,2,value_offset=123.,value_scale=456.)
    with torch.no_grad():model.readout.weight.normal_()
    boards=torch.randint(0,10,(12,16));before=model(boards).detach()
    encoder={k:v.clone() for k,v in model.state_dict().items() if k.startswith(('blocks.','positions','tokenizer.','norm.'))}
    reset_critic(model);after=model(boards).detach()
    torch.testing.assert_close(after[:,:4],before[:,:4],rtol=0,atol=0)
    torch.testing.assert_close(after[:,4],torch.zeros(12))
    for k,v in encoder.items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)


def test_teacher_ppo_collects_own_games_and_saves_reloadable_policy(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(3)
    model=TeacherPolicyTransformer(8,'embedding',1,2,value_offset=4000.,value_scale=900.)
    with torch.no_grad():model.readout.weight.normal_(0,.05)
    NeuralAgent(model,'teacher_policy_value',width=8).save(tmp_path/'actor',{'supervised_updates':7})
    config=dict(algorithm='teacher_ppo',architecture='transformer_policy_value',width=8,depth=1,heads=2,
        input_encoding='embedding',initial_actor_checkpoint=str(tmp_path/'actor'),
        gamma=1.,lam=.95,reward_mode='score',envs=2,horizon=4,batch=4,epochs=2,lr=1e-4,
        clip=.2,target_kl=.03,value_weight=.5,entropy_weight=.005,max_grad_norm=.5,
        collection_seed=12000000,monitor_seed_start=12000100,final_seed_start=12000200,
        monitor_games=2,monitor_interval=8,max_training_seconds=60)
    result=train_one(config,0,16,tmp_path/'run','cpu')
    assert result['complete'] and result['completed_budget']
    assert result['training_transitions']==16 and result['rollouts']==2 and result['updates']>0
    assert result['online_training'] and not result['teacher_labels_used_online'] and not result['replay_buffer']
    assert result['value_offset']==0. and result['value_scale']==1.
    log=json.loads((tmp_path/'run/progress.json').read_text())
    assert all(np.isfinite(r['value_loss']) and np.isfinite(r['approx_kl']) for r in log)
    saved=torch.load(tmp_path/'run/training.pt',map_location='cpu',weights_only=True)
    assert saved['optimizer']['state'] and saved['transitions']==16
    loaded=NeuralAgent.load(tmp_path/'run/last');loaded.policy(torch.zeros((2,16),dtype=torch.long))
    assert (tmp_path/'run/replay_best.html').exists()
    assert json.loads((tmp_path/'run/evaluation.json').read_text())['summary']['episodes']==100
