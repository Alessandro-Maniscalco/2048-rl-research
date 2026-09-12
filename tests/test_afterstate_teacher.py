import json
import numpy as np
import pytest
import torch

from research.afterstate_teacher import digest, warm_start, validation_metrics, grouped_distillation_loss
from rl2048.agents.afterstate_mlp import AfterstateLearner


def dataset(tmp_path):
    folder = tmp_path/'data'; folder.mkdir()
    rng = np.random.default_rng(0)
    after = rng.integers(0, 6, (8, 4, 16), dtype=np.uint8)
    legal = np.ones((8, 4), bool)
    legal[:, 3] = False
    # Illegal alternatives must neither be fitted nor win agreement metrics.
    labels = np.ones((8, 4), np.float32)
    labels[:, 3] = 999999
    validation = np.array([False]*6+[True]*2)
    data = dict(afterstates=after, legal=legal, values=labels,
                gains=np.zeros((8, 4), np.float32), validation=validation)
    np.savez(folder/'data.npz', **data)
    manifest = dict(complete=True, gamma=1., label_units='future raw merge points / 128',
                    data_sha256=digest(folder/'data.npz'), teacher_sha256='fixture', source_transitions=8)
    (folder/'manifest.json').write_text(json.dumps(manifest))
    return folder, data


def test_teacher_fit_improves_values_then_syncs_target_and_resets_optimizer(tmp_path):
    torch.set_num_threads(1); torch.manual_seed(0)
    folder, data = dataset(tmp_path)
    learner = AfterstateLearner(width=16, embedding_dim=4, architecture='afterstate_sym_mlp',
                               input_encoding='embedding', lr=.01)
    initial = validation_metrics(learner.policy, data, 'cpu')['value_mse']
    config = dict(gamma=1., teacher_pretrain=dict(dataset=str(folder), updates=80,
                  seconds=30, batch=16, validation_interval=40))
    result = warm_start(learner, config, tmp_path)
    assert result['updates'] == 80
    assert result['fitting_afterstates'] == 18
    assert result['final_validation']['value_mse'] < initial*.2
    assert result['final_validation']['teacher_action_agreement'] == 1.
    assert not learner.optimizer.state
    for online, target in zip(learner.policy.parameters(), learner.target.parameters()):
        torch.testing.assert_close(online, target)
    assert not result['teacher_data_in_online_replay']


def test_reject_changed_data_and_incompatible_gamma(tmp_path):
    folder, _ = dataset(tmp_path)
    learner = AfterstateLearner(width=8)
    config = dict(gamma=.99, teacher_pretrain=dict(dataset=str(folder)))
    with pytest.raises(ValueError, match='incompatible'):
        warm_start(learner, config, tmp_path)
    config['gamma'] = 1.
    with (folder/'data.npz').open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError, match='changed'):
        warm_start(learner, config, tmp_path)


def test_action_gap_loss_removes_common_offset_and_masks_invalid_moves():
    legal=torch.tensor([[True,True,False],[True,True,False]])
    target=torch.tensor([[0.,2.,-99999.],[1.,7.,-99999.]])
    prediction=torch.tensor([[1.,3.,99999.],[4.,10.,99999.]],requires_grad=True)
    # Errors [1,1] and [3,3] are common board offsets: gap penalty is zero.
    torch.testing.assert_close(grouped_distillation_loss(prediction,target,legal,10.),torch.tensor(5.))
    prediction2=torch.tensor([[1.,5.,99999.],[1.,7.,99999.]],requires_grad=True)
    # Errors [1,3] and [0,0]: ordinary MSE2.5, centered MSE.5.
    loss=grouped_distillation_loss(prediction2,target,legal,10.)
    torch.testing.assert_close(loss,torch.tensor(7.5))
    loss.backward()
    assert (prediction2.grad[:,2]==0).all()


def test_grouped_teacher_initialization_runs_with_real_gradients(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(0)
    folder,data=dataset(tmp_path)
    learner=AfterstateLearner(width=16,embedding_dim=4,architecture='afterstate_sym_mlp',
                             input_encoding='embedding',lr=.01)
    config=dict(gamma=1.,teacher_pretrain=dict(dataset=str(folder),updates=80,seconds=30,
        batch=4,validation_interval=40,grouped_actions=True,gap_weight=10.))
    result=warm_start(learner,config,tmp_path)
    assert result['grouped_actions'] and result['updates']==80
    assert result['final_validation']['value_mse']<.2
