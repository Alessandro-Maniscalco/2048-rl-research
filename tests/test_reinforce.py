import json
import numpy as np
import pytest
import torch

from rl2048.agents.neural import NeuralAgent
from rl2048.agents.reinforce import (PolicyTransformer, copy_actor, reward_to_go, policy_loss,
                                    leave_one_out_time_baseline, action_log_probs)
from rl2048.agents.teacher_policy_transformer import TeacherPolicyTransformer
from research.reinforce_experiment import collect_complete_games, train_one


def test_returns_and_masked_policy_gradient_by_hand():
    np.testing.assert_allclose(reward_to_go([4., 0., 8.]), [12., 8., 8.])
    np.testing.assert_allclose(reward_to_go([4., 0., 8.], .5), [6., 4., 8.])
    # Two legal equal-probability actions, chosen action 0, G=6: dL/dz=[-3,3,0,0].
    logits = torch.tensor([[0., 0., 99., -99.]], requires_grad=True)
    loss = policy_loss(logits, torch.tensor([[True, True, False, False]]),
                       torch.tensor([0]), torch.tensor([6.]), 1)
    assert loss.item() == pytest.approx(6*np.log(2))
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[-3., 3., 0., 0.]]))


def test_chunk_gradient_equals_one_complete_batch_gradient():
    torch.manual_seed(7)
    inputs = torch.randn(7, 3)
    masks = torch.ones(7, 4, dtype=torch.bool)
    actions = torch.tensor([0, 1, 2, 3, 0, 1, 2])
    # Two episodes with lengths 3 and 4. Normalize by two games, not seven moves.
    returns = torch.tensor([8., 4., 4., 12., 8., 8., 4.])
    full = torch.nn.Linear(3, 4)
    chunked = torch.nn.Linear(3, 4)
    chunked.load_state_dict(full.state_dict())
    policy_loss(full(inputs), masks, actions, returns, 2).backward()
    for start in (0, 3, 6):
        sl = slice(start, start+3)
        policy_loss(chunked(inputs[sl]), masks[sl], actions[sl], returns[sl], 2).backward()
    for expected, actual in zip(full.parameters(), chunked.parameters()):
        torch.testing.assert_close(expected.grad, actual.grad)


def test_actor_transfer_has_no_critic_and_preserves_logits(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(9)
    teacher = TeacherPolicyTransformer(8, 'embedding', 1, 2, 123., 456.)
    with torch.no_grad():
        teacher.readout.weight.normal_()
    actor = PolicyTransformer(8, 'embedding', 1, 2)
    copy_actor(actor, teacher)
    boards = torch.randint(0, 12, (8, 16))
    torch.testing.assert_close(actor(boards), teacher(boards)[:, :4], rtol=0, atol=0)
    assert actor.outputs == 4 and not hasattr(actor, 'value_scale')
    NeuralAgent(actor, 'reinforce', width=8).save(tmp_path / 'policy')
    restored = NeuralAgent.load(tmp_path / 'policy')
    torch.testing.assert_close(restored.policy(boards), actor(boards))


def test_collection_finishes_every_game_under_unchanged_policy():
    torch.set_num_threads(1)
    model = PolicyTransformer(8, 'embedding', 1, 2)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    data, episodes, count, reason = collect_complete_games(model, 3, 12900000,
        np.random.default_rng(7), 'cpu')
    assert reason is None and len(episodes) == 3
    assert count == sum(e['length'] for e in episodes) == len(data['actions'])
    assert all(e['terminated'] and not e['truncated'] for e in episodes)
    assert data['masks'][np.arange(count), data['actions']].all()
    offset = 0
    for e in episodes:
        assert data['returns'][offset] == e['score']/128
        offset += e['length']
    for k, v in before.items():
        torch.testing.assert_close(v, model.state_dict()[k], rtol=0, atol=0)
    repeat, same, same_count, _ = collect_complete_games(model, 3, 12900000,
        np.random.default_rng(7), 'cpu')
    assert episodes == same and count == same_count
    np.testing.assert_array_equal(data['states'], repeat['states'])
    assert collect_complete_games(model, 3, 1, np.random.default_rng(7), 'cpu',
                                  lambda: 'stop_file') == (None, [], 0, 'stop_file')


@pytest.mark.parametrize('baseline,temperature', [('none',1.), ('leave_one_out_time',1.), ('leave_one_out_time',.25)])
def test_complete_training_reload_and_replay(tmp_path, baseline, temperature):
    torch.set_num_threads(1)
    config = dict(algorithm='reinforce', architecture='transformer_policy', width=8,
        depth=1, heads=2, input_encoding='embedding', gamma=1., reward_mode='score',
        episode_batch=2, batch=32, lr=1e-4, max_grad_norm=.5,
        collection_seed=12910000, monitor_seed_start=12910100, final_seed_start=12910200,
        monitor_games=2, monitor_interval=10000, max_training_seconds=60)
    config['baseline_mode'] = baseline
    config['action_temperature'] = temperature
    if temperature != 1.:
        config.update(monitor_sampled=True, sampling_seed=12940)
    if baseline != 'none':
        config['algorithm'] = 'reinforce_loo'
    result = train_one(config, 0, 1, tmp_path / 'run', 'cpu')
    assert result['complete'] and result['completed_budget']
    assert result['completed_training_games'] == 2 and result['updates'] == 1
    assert result['training_transitions'] == result['trained_transitions'] > 1
    assert not result['critic'] and not result['bootstrap']
    assert result['baseline'] == (False if baseline == 'none' else baseline)
    assert result['action_temperature'] == temperature
    assert not result['replay_buffer'] and not result['teacher_labels_used_online']
    log = json.loads((tmp_path / 'run/progress.json').read_text())
    assert np.isfinite(log[0]['policy_loss']) and log[0]['max_behavior_logp_difference'] < 1e-5
    saved = torch.load(tmp_path / 'run/training.pt', map_location='cpu', weights_only=True)
    # Adam's internal step count is one for every parameter, despite many chunks.
    assert all(v['step'] == 1 for v in saved['optimizer']['state'].values())
    assert (tmp_path / 'run/replay_best.html').exists()
    assert json.loads((tmp_path / 'run/evaluation.json').read_text())['summary']['episodes'] == 100
    if temperature != 1.:
        sampled = json.loads((tmp_path / 'run/sampled_evaluation.json').read_text())
        assert sampled['summary']['episodes'] == 100 and sampled['summary']['temperature'] == temperature
        assert result['sampled_mean_score'] == sampled['summary']['mean_score']
        assert all('sampled_mean_score' in row for row in json.loads((tmp_path / 'run/curve.json').read_text()))


def test_leave_one_out_baseline_by_hand_and_independence():
    # Completed games of lengths 3, 2, 1, flattened in game order.
    returns = np.array([12., 8., 8., 2., 0., 6.])
    baseline = leave_one_out_time_baseline(returns, [3, 2, 1])
    np.testing.assert_allclose(baseline, [4., 0., 0., 9., 4., 7.])
    changed = returns.copy()
    changed[:3] += 1000
    # No reward/action from the first game may influence its own baseline.
    np.testing.assert_array_equal(leave_one_out_time_baseline(changed, [3, 2, 1])[:3], baseline[:3])
    with pytest.raises(ValueError):
        leave_one_out_time_baseline([1.], [1])


def test_temperature_probabilities_and_gradient_by_hand():
    logits = torch.tensor([[np.log(3.), 0., 100., -100.]], dtype=torch.float64, requires_grad=True)
    masks = torch.tensor([[True, True, False, False]])
    # At T=1/2, odds 3:1 become 9:1; illegal logits never contribute.
    logs = action_log_probs(logits, masks, .5)
    torch.testing.assert_close(logs.exp(), torch.tensor([[.9, .1, 0., 0.]], dtype=torch.float64))
    loss = policy_loss(logits, masks, torch.tensor([0]), torch.tensor([6.]), 1, .5)
    loss.backward()
    # dL/dz = G/T * (p-one_hot(action)); the 1/T factor must be differentiated.
    torch.testing.assert_close(logits.grad, torch.tensor([[-1.2, 1.2, 0., 0.]], dtype=torch.float64))
    for invalid in (0., -1., float('nan')):
        with pytest.raises(ValueError):
            action_log_probs(logits, masks, invalid)


@pytest.mark.parametrize('spawn_safety', [False, True])
def test_pretrained_cnn_complete_batch_budget_and_unused_critic(tmp_path, spawn_safety):
    from rl2048.agents.pretrained2048 import Pretrained2048
    from research.afterstate_teacher import digest
    torch.set_num_threads(1)
    model = Pretrained2048()
    source = tmp_path/'source'
    NeuralAgent(model, 'pretrained_ppo', width=1024).save(source)
    checksum = digest(source/'policy.pt')
    config = dict(algorithm='reinforce_loo',baseline_mode='leave_one_out_time',
        architecture='pretrained_ml2048',width=1024,depth=None,heads=None,input_encoding='one_hot',
        initial_actor_checkpoint=str(source),gamma=1.,reward_mode='score',episode_batch=2,
        batch=32,lr=1e-5,max_grad_norm=.5,collection_seed=12930000,monitor_seed_start=12930100,
        final_seed_start=12930200,monitor_games=2,monitor_interval=10000,
        max_training_seconds=60,complete_batches=2,spawn_safety=spawn_safety,evaluate_initial_policy=spawn_safety)
    result = train_one(config,0,1,tmp_path/'run','cpu')
    assert result['complete'] and result['completed_budget'] and result['updates']==2
    assert result['completed_training_games']==4
    assert result['requested_transitions'] is None and result['requested_complete_batches']==2
    assert not result['teacher_labels_used_online'] and not result['critic'] and not result['bootstrap']
    assert digest(source/'policy.pt') == checksum
    trained = torch.load(tmp_path/'run/training.pt',map_location='cpu',weights_only=True)
    for key,value in model.state_dict().items():
        if key.startswith('_critic.'):
            torch.testing.assert_close(trained['policy'][key],value,rtol=0,atol=0)
    assert any(not torch.equal(trained['policy'][k],v) for k,v in model.state_dict().items()
               if k.startswith('_encoder.'))
    assert all(v['step']==2 for v in trained['optimizer']['state'].values())
    assert len(result['first_complete_batch_sha256']) == 64
    progress=json.loads((tmp_path/'run/progress.json').read_text())
    assert len(progress)==2 and max(p['max_behavior_logp_difference'] for p in progress)<1e-4
    assert (tmp_path/'run/replay_last.html').exists()
    restored=NeuralAgent.load(tmp_path/'run/last')
    assert restored.spawn_safety==spawn_safety
    first=json.loads((tmp_path/'run/first_training_game.json').read_text())
    for transition in first['transitions']:
        assert transition['legal_mask'][transition['action']]
        assert transition['policy_mask'][transition['action']]
    if spawn_safety:
        from rl2048.agents.spawn_safety import minimum_risk_mask
        initial=json.loads((tmp_path/'run/initial_evaluation.json').read_text())
        assert initial['summary']['complete'] and len(initial['episodes'])==100
        assert NeuralAgent.load(tmp_path/'run/initial').spawn_safety
        for transition in first['transitions']:
            actual=minimum_risk_mask(np.array([transition['board_exponents']],dtype=np.uint8))
            np.testing.assert_array_equal(actual[0],transition['policy_mask'])
