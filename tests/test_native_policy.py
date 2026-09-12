import json
from pathlib import Path

import numpy as np
import pytest
import torch

from research.afterstate_teacher import digest, write
from research.native_policy_data import extract_game
from research.native_policy_experiment import action_loss, initialize, train_one, sample_indices
from rl2048.agents.neural import NeuralAgent
from rl2048.agents.native_policy import NativePolicy
from rl2048.agents.pretrained2048 import Pretrained2048
from rl2048.game import Game2048, ACTION_NAMES


def episode(path, seed):
    env=Game2048();board,info=env.reset(seed=seed);rng=np.random.default_rng(seed)
    frames=[dict(board=board.tolist(),score=0,action=None,reward=0)]
    done=False
    while not done:
        action=int(rng.choice(np.flatnonzero(info['action_mask'])))
        board,reward,done,truncated,info=env.step(action)
        frames.append(dict(board=board.tolist(),score=env.score,action=action,reward=reward))
    data=dict(actions=list(ACTION_NAMES),frames=frames,result=dict(seed=seed,score=env.score,
        length=len(frames)-1,max_tile=int(board.max()),terminated=True,truncated=False))
    write(path,data)
    return data


def test_old_features_preserved_new_ranks_can_learn_and_actor_is_fresh(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(14)
    source=Pretrained2048()
    # Also cover nonzero learned biases in transferred depthwise layers.
    with torch.no_grad():
        source._encoder._depthwise_full.bias.uniform_(-1,1)
    NeuralAgent(source,'original',width=1024).save(tmp_path/'source')
    cold,cold_actor=initialize(9,'scratch',tmp_path/'source')
    warm,warm_actor=initialize(9,'encoder',tmp_path/'source')
    assert cold_actor==warm_actor
    boards=torch.randint(16,(17,16))
    torch.testing.assert_close(warm._encoder(boards),source._encoder(boards),rtol=1e-5,atol=2e-6)
    assert all(torch.equal(v,warm._actor.state_dict()[k]) for k,v in cold._actor.state_dict().items())
    boards.fill_(0);boards[:,0]=16
    warm(boards).square().mean().backward()
    assert warm._encoder._depthwise_full.weight.grad[256:272].abs().sum()>0
    player=NeuralAgent(warm,'native_behavior_cloning',width=1024)
    player.save(tmp_path/'saved');loaded=NeuralAgent.load(tmp_path/'saved')
    torch.testing.assert_close(loaded.policy(boards),warm(boards))
    assert not any('critic' in k for k in warm.state_dict())


def test_action_loss_hand_calculation_and_illegal_gradient():
    logits=torch.tensor([[0.,100.,np.log(3.),-20.]],requires_grad=True)
    legal=torch.tensor([[True,False,True,False]])
    loss=action_loss(logits,torch.tensor([2]),legal)
    assert loss.item()==pytest.approx(-np.log(.75))
    loss.backward()
    assert logits.grad[0,1]==0 and logits.grad[0,3]==0
    assert logits.grad[0,0]==pytest.approx(.25)
    assert logits.grad[0,2]==pytest.approx(-.25)


def test_stage_sampling_and_explicit_exact_continuation(tmp_path):
    pools=[np.arange(0,10),np.arange(10,200),np.arange(200,1000)]
    ids=sample_indices(np.random.default_rng(2),1000,100000,pools,.5)
    observed=np.array([(ids<10).mean(),((ids>=10)&(ids<200)).mean(),(ids>=200).mean()])
    np.testing.assert_allclose(observed,.5*np.array([.01,.19,.8])+.5/3,atol=.005)
    assert ids.min()>=0 and ids.max()<1000
    torch.set_num_threads(1);model=NativePolicy()
    NeuralAgent(model,'native_behavior_cloning',width=1024).save(tmp_path/'source')
    loaded,_=initialize(99,'continuation','unused',tmp_path/'source')
    assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items())
    with pytest.raises(ValueError,match='explicit'):
        initialize(99,'scratch','unused',tmp_path/'source')


def test_replay_action_alignment_rules_and_reject_corruption(tmp_path):
    path=tmp_path/'game.json';data=episode(path,71)
    states,actions,legal,result=extract_game(path)
    np.testing.assert_array_equal(actions,[f['action'] for f in data['frames'][1:]])
    assert result['length']==len(states) and legal[np.arange(len(actions)),actions].all()
    assert (states[0]>0).sum()==2
    data['frames'][1]['reward']+=4;write(path,data)
    with pytest.raises(ValueError,match='Misaligned'):
        extract_game(path)


def test_complete_training_save_reload_and_frozen_source(tmp_path):
    torch.set_num_threads(1)
    blocks=[]
    for seed in (72,73):
        path=tmp_path/f'{seed}.json';episode(path,seed)
        blocks.append(extract_game(path)[:3])
    dataset=tmp_path/'data';dataset.mkdir()
    arrays={key:np.concatenate([b[i] for b in blocks]) for i,key in enumerate(['states','actions','legal'])}
    arrays['validation']=np.r_[np.zeros(len(blocks[0][0]),bool),np.ones(len(blocks[1][0]),bool)]
    np.savez_compressed(dataset/'data.npz',**arrays)
    write(dataset/'manifest.json',dict(complete=True,data_sha256=digest(dataset/'data.npz')))
    source=tmp_path/'source';NeuralAgent(Pretrained2048(),'original',width=1024).save(source)
    original=digest(source/'policy.pt')
    config=dict(expert_dataset=str(dataset),source_checkpoint=str(source),initialization='encoder',
        batch=8,lr=1e-4,max_grad_norm=.5,validation_boards=16,monitor_seed_start=8200100,
        final_seed_start=8200200,monitor_games=2,final_games=2,monitor_interval=2,
        max_training_seconds=60,stop_file=str(tmp_path/'STOP'))
    out=tmp_path/'run';result=train_one(config,1,2,out,'cpu')
    assert result['complete'] and result['completed_budget'] and result['updates']==2
    assert result['checkpoint_reload_verified'] and result['source_unchanged'] and result['dataset_unchanged']
    assert digest(source/'policy.pt')==original
    state=torch.load(out/'training.pt',map_location='cpu',weights_only=True)
    assert {int(v['step']) for v in state['optimizer']['state'].values()}=={2}
    assert not any('critic' in key for key in state['policy'])
