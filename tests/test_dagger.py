import json
import numpy as np
import pytest
import torch

from research.afterstate_teacher import digest
from research.dagger_experiment import aggregate, teacher_targets, train_one, initialize_actor
from research.reinforce_experiment import collect_complete_games
from rl2048.afterstate_compare import moves
from rl2048.agents.imitation import imitation_loss, optimal_actions, teacher_gap_penalty, reference_policy_kl
from rl2048.agents.neural import NeuralAgent, policy_logits
from rl2048.agents.pretrained2048 import Pretrained2048
from rl2048.agents.ntuple import NTupleAgent, row_tables
from rl2048.agents.reinforce import PolicyTransformer
from rl2048.vector_game import VectorGame


def test_cnn_actor_only_initialization_preserves_logits_and_freezes_critic(tmp_path):
    torch.set_num_threads(1)
    model = Pretrained2048()
    NeuralAgent(model, 'pretrained_ppo', width=1024).save(tmp_path/'source')
    config = dict(architecture='pretrained_ml2048', width=1024,
                  input_encoding='one_hot', depth=None, heads=None)
    actor, _ = initialize_actor(config, tmp_path/'source', 'cpu')
    boards = torch.arange(32).reshape(2, 16) % 18
    torch.testing.assert_close(policy_logits(actor, boards), model(boards)[:, :4], rtol=0, atol=0)
    def unused_critic(*args):
        raise AssertionError('Actor-only training must not evaluate the critic')
    handle = actor._critic.register_forward_pre_hook(unused_critic)
    policy_logits(actor, boards).square().sum().backward()
    handle.remove()
    assert all(not p.requires_grad and p.grad is None for p in actor._critic.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor._encoder.parameters())
    with pytest.raises(ValueError, match='no configurable attention'):
        initialize_actor(config | dict(heads=4), tmp_path/'source', 'cpu')


def test_reference_policy_kl_value_gradient_mask_and_detachment():
    old = torch.tensor([[np.log(.75), np.log(.25), 999., 999.]], requires_grad=True)
    new = torch.tensor([[np.log(.5), np.log(.5), -999., -999.]], requires_grad=True)
    legal = torch.tensor([[True, True, False, False]])
    kl = reference_policy_kl(new, old, legal)
    assert kl.item() == pytest.approx(.75*np.log(1.5)+.25*np.log(.5))
    kl.backward()
    assert old.grad is None
    torch.testing.assert_close(new.grad, torch.tensor([[-.25, .25, 0., 0.]], dtype=new.dtype))
    assert reference_policy_kl(old, old, legal).item() == 0.
    saturated = torch.tensor([[1000., -1000., 123., 123.]], requires_grad=True)
    reverse = reference_policy_kl(saturated, old, legal)
    reverse.backward()
    assert torch.isfinite(reverse) and torch.isfinite(saturated.grad).all()


def test_tied_teacher_loss_and_gradient_by_hand():
    logits = torch.tensor([[np.log(.2), np.log(.3), np.log(.5), 999.]], dtype=torch.float32, requires_grad=True)
    legal = torch.tensor([[True, True, True, False]])
    best = torch.tensor([[True, True, False, False]])
    loss = imitation_loss(logits, legal, best)
    assert loss.item() == pytest.approx(-np.log(.5))
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[-.2, -.3, .5, 0.]]))
    # Unique target reduces to standard categorical cross entropy.
    unique = imitation_loss(logits, legal, torch.tensor([[False, False, True, False]]))
    torch.testing.assert_close(unique, torch.nn.functional.cross_entropy(logits[:, :3], torch.tensor([2])))
    with pytest.raises(ValueError):
        imitation_loss(logits, legal, torch.tensor([[False, False, False, True]]))
    with pytest.raises(ValueError):
        imitation_loss(logits, legal, torch.zeros_like(best))
    q = np.array([[10., 10.+1e-6, 1., 999.]])
    np.testing.assert_array_equal(optimal_actions(q, legal.numpy()), best.numpy())


def test_teacher_uses_complete_value_plus_merge_and_no_search():
    teacher = NTupleAgent('4x4', initial_value=128.)
    states = np.array([[1, 1, 0, 0]+[0]*12], dtype=np.uint8)
    data = teacher_targets(states, teacher)
    _, gains, legal = moves(states, *row_tables())
    # 4 tables x 8 orientations sum to exactly 128 future points.
    np.testing.assert_allclose(data['q'][legal], 1+gains[legal]/128)
    np.testing.assert_array_equal(data['best'], legal & (gains == 4))
    joined = aggregate(data, data)
    assert len(joined['states']) == 2
    for k in data:
        np.testing.assert_array_equal(joined[k][:1], data[k])
        np.testing.assert_array_equal(joined[k][1:], data[k])


def test_expected_teacher_gap_loss_and_gradient_by_hand():
    logits = torch.tensor([[np.log(.2), np.log(.3), np.log(.5), 999.]], requires_grad=True)
    legal = torch.tensor([[True, True, True, False]])
    best = torch.tensor([[True, False, False, False]])
    q = torch.tensor([[10., 8., 0., -torch.inf]], requires_grad=True)
    # Q is points/128. Scale 256 raw points gives costs [0, 1, 5].
    penalty = teacher_gap_penalty(logits, legal, best, q, 256.)
    assert penalty.item() == pytest.approx(2.8)
    penalty.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[-.56, -.54, 1.1, 0.]], dtype=logits.dtype))
    assert q.grad is None
    torch.testing.assert_close(teacher_gap_penalty(logits, legal, best, q+1000, 256.), penalty)
    tied = torch.tensor([[True, True, False, False]])
    assert teacher_gap_penalty(logits, legal, tied, q, 256.).item() == pytest.approx(2.5)
    with pytest.raises(ValueError, match='positive'):
        teacher_gap_penalty(logits, legal, best, q, 0.)
    with pytest.raises(ValueError, match='finite'):
        teacher_gap_penalty(logits, legal, best, torch.tensor([[torch.inf, 8., 0., 0.]]))


def test_cross_entropy_retains_gradient_when_wrong_policy_is_confident():
    legal = torch.tensor([[True, True, False, False]])
    best = torch.tensor([[True, False, False, False]])
    logits = torch.tensor([[-30., 30., 999., 999.]], requires_grad=True)
    q = torch.tensor([[10., 0., -torch.inf, -torch.inf]])
    loss = imitation_loss(logits, legal, best)+teacher_gap_penalty(logits, legal, best, q)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0].item() == pytest.approx(-1.)
    assert logits.grad[0, 1].item() == pytest.approx(1.)
    torch.testing.assert_close(logits.grad[0, 2:], torch.zeros(2))


def test_greedy_student_collects_own_legal_choices_without_weight_change():
    torch.set_num_threads(1)
    torch.manual_seed(71)
    model = PolicyTransformer(8, 'embedding', 1, 2)
    with torch.no_grad():
        model.readout.weight.normal_(0, .1)
    before = {k:v.clone() for k,v in model.state_dict().items()}
    data, episodes, used, reason = collect_complete_games(model, 2, 123456,
        np.random.default_rng(0), 'cpu', greedy=True)
    assert reason is None and used == sum(e['length'] for e in episodes)
    with torch.no_grad():
        logits = model(torch.tensor(data['states'])).numpy()
    legal_logits = np.where(data['masks'], logits, -np.inf)
    # A different inference batch can perturb exact symmetry ties by roundoff.
    np.testing.assert_allclose(legal_logits[np.arange(used), data['actions']],
                               legal_logits.max(1), atol=1e-6, rtol=1e-6)
    repeat, same, _, _ = collect_complete_games(model, 2, 123456,
        np.random.default_rng(55), 'cpu', greedy=True)
    np.testing.assert_array_equal(repeat['actions'], data['actions'])
    assert same == episodes  # Greedy collection does not use the action RNG.
    for k,v in before.items():
        torch.testing.assert_close(model.state_dict()[k], v, atol=0, rtol=0)


@pytest.mark.parametrize('source,architecture', [('fixed', 'transformer_policy'),
    ('student', 'transformer_policy'), ('student', 'pretrained_ml2048')])
def test_complete_paired_training_aggregation_reload_and_replay(tmp_path, source, architecture):
    torch.set_num_threads(1)
    teacher = NTupleAgent('4x4', initial_value=128.)
    teacher.save(tmp_path/'teacher')
    teacher_hash = digest(tmp_path/'teacher/weights.npy')
    folder = tmp_path/'data'
    folder.mkdir()
    states = VectorGame(8, 777).boards.copy()
    after, gains, legal = moves(states, *row_tables())
    np.savez(folder/'data.npz', states=states, afterstates=after, gains=gains,
        legal=legal, values=np.ones((8,4),np.float32), validation=np.array([False]*6+[True]*2))
    (folder/'manifest.json').write_text(json.dumps(dict(complete=True, gamma=1.,
        label_units='future raw merge points / 128', data_sha256=digest(folder/'data.npz'), teacher_sha256=teacher_hash)))
    cnn = architecture == 'pretrained_ml2048'
    model = Pretrained2048() if cnn else PolicyTransformer(8, 'embedding', 1, 2)
    width, depth, heads, encoding = (1024, None, None, 'one_hot') if cnn else (8, 1, 2, 'embedding')
    NeuralAgent(model, 'pretrained_ppo' if cnn else 'reinforce', width=width).save(tmp_path/'actor')
    config = dict(algorithm='dagger', data_source=source, architecture=architecture,
        initial_actor_checkpoint=str(tmp_path/'actor'), teacher_dataset=str(folder),
        teacher_checkpoint=str(tmp_path/'teacher'), width=width, depth=depth, heads=heads, input_encoding=encoding,
        batch=8, updates_per_round=2, collection_games=2, lr=1e-4, max_grad_norm=.5,
        collection_seed=12400000, monitor_games=2, monitor_seed_start=12410000,
        final_seed_start=12420000, monitor_every_rounds=1, max_training_seconds=60)
    if cnn:
        config['reference_kl_weight'] = 10.
    out = tmp_path/'run'
    result = train_one(config, 0, 4, out, 'cpu')
    assert result['complete'] and result['completed_budget']
    assert result['updates'] == 4 and result['rounds'] == 2
    assert result['reinforcement_learning'] is False
    assert result['original_fitting_boards'] == 6 and result['validation_boards_used'] == 2
    assert result['fitting_boards'] == 6+result['labelled_transitions']
    assert result['training_transitions'] == result['labelled_transitions']
    blocks = json.loads((out/'dataset_blocks.json').read_text())
    if source == 'student':
        assert result['completed_training_games'] == 4 and len(blocks) == 2
        assert result['training_transitions'] == sum(b['boards'] for b in blocks)
        for b in blocks:
            assert digest(out/b['file']) == b['sha256']
            with np.load(out/b['file']) as data:
                actual = teacher_targets(data['states'], teacher)
                np.testing.assert_array_equal(actual['best'], data['best'])
                np.testing.assert_array_equal(actual['q'], data['q'])
                assert data['episode_lengths'].sum() == len(data['states'])
    else:
        assert result['training_transitions'] == 0 and not blocks
    assert (out/'replay_last.html').exists() and (out/'replay_best.html').exists()
    assert json.loads((out/'evaluation.json').read_text())['summary']['episodes'] == 100
    saved = torch.load(out/'training.pt', map_location='cpu', weights_only=True)
    assert saved['updates'] == 4 and saved['optimizer']['state']
    if cnn:
        for key, value in model.state_dict().items():
            if key.startswith('_critic.'):
                torch.testing.assert_close(saved['policy'][key], value, rtol=0, atol=0)
        assert any(not torch.equal(saved['policy'][key], value)
                   for key, value in model.state_dict().items() if key.startswith('_encoder.'))
    reloaded = NeuralAgent.load(out/'last')
    for k,v in reloaded.policy.state_dict().items():
        torch.testing.assert_close(v, saved['policy'][k])
    assert digest(tmp_path/'teacher/weights.npy') == teacher_hash
    # Two complete rounds must be identical to a saved/reloaded split between
    # rounds: this checks dataset aggregation, new collection seeds, Adam and RNG.
    half = tmp_path/'half'
    train_one(config, 0, 2, half, 'cpu')
    continued = train_one(config | dict(resume_dagger=str(half)), 0, 2, tmp_path/'continued', 'cpu')
    restored = torch.load(tmp_path/'continued/training.pt', map_location='cpu', weights_only=True)
    assert continued['updates'] == 2 and continued['total_dagger_updates'] == 4
    assert continued['rounds'] == 1 and continued['total_rounds'] == 2
    assert continued['fitting_boards'] == result['fitting_boards']
    assert continued['total_training_transitions'] == result['training_transitions']
    for k,v in saved['policy'].items():
        torch.testing.assert_close(restored['policy'][k], v, rtol=0, atol=0)
    for index, values in saved['optimizer']['state'].items():
        for key,value in values.items():
            torch.testing.assert_close(restored['optimizer']['state'][index][key], value, rtol=0, atol=0)
    for block in blocks:
        with np.load(out/block['file']) as expected, np.load(tmp_path/'continued'/block['file']) as actual:
            for key in expected.files:
                np.testing.assert_array_equal(actual[key], expected[key])
    with pytest.raises(ValueError, match='preserve lr'):
        train_one(config | dict(resume_dagger=str(half), lr=.02), 0, 2, tmp_path/'changed_lr', 'cpu')
    if source == 'fixed':
        with pytest.raises(ValueError, match='positive weight'):
            train_one(config | dict(teacher_action_weight=0.), 0, 2, tmp_path/'no_teacher', 'cpu')
        with pytest.raises(ValueError, match='explicit transfer'):
            train_one(config | dict(resume_dagger=str(half), teacher_action_weight=0., teacher_gap_weight=1.),
                0, 2, tmp_path/'silent_action_loss_change', 'cpu')
        cost_only = train_one(config | dict(teacher_action_weight=0., teacher_gap_weight=1.),
            0, 2, tmp_path/'cost_only', 'cpu')
        assert cost_only['complete'] and cost_only['teacher_action_weight'] == 0.
        assert cost_only['teacher_gap_weight'] == 1. and cost_only['updates'] == 2
        assert np.isfinite(json.loads((tmp_path/'cost_only/progress.json').read_text())[-1]['mean_minibatch_loss'])
    if source == 'student':
        with pytest.raises(ValueError, match='explicit transfer'):
            train_one(config | dict(resume_dagger=str(half), teacher_gap_weight=1.),
                0, 2, tmp_path/'silent_loss_change', 'cpu')
        with pytest.raises(ValueError, match='actually change'):
            train_one(config | dict(resume_dagger=str(half), transfer_dagger_objective=True),
                0, 2, tmp_path/'noop_transfer', 'cpu')
        source_hash = digest(half/'training.pt')
        branch = tmp_path/'cost_branch'
        cost_result = train_one(config | dict(resume_dagger=str(half), teacher_gap_weight=1.,
            teacher_gap_scale_points=128., transfer_dagger_objective=True), 0, 2, branch, 'cpu')
        assert cost_result['complete'] and cost_result['source_objective']['teacher_gap_weight'] == 0.
        assert cost_result['total_dagger_updates'] == 4
        assert cost_result['reinforcement_learning'] is False
        assert cost_result['fitting_boards'] == continued['fitting_boards']
        assert digest(half/'training.pt') == source_hash
        # The first branch collection happens before either changed-loss update,
        # so equal initial weights and RNG must produce the identical new block.
        with np.load(branch/'round_002.npz') as a, np.load(tmp_path/'continued/round_002.npz') as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
        cost_state = torch.load(branch/'training.pt', map_location='cpu', weights_only=True)
        assert any(not torch.equal(cost_state['policy'][k], restored['policy'][k]) for k in restored['policy'])
        assert json.loads((branch/'progress.json').read_text())[-1]['mean_teacher_gap_penalty'] >= 0.
        with pytest.raises(ValueError, match='explicit transfer_dagger_sampling'):
            train_one(config | dict(resume_dagger=str(half), stage_sampling_fraction=.5),
                0, 2, tmp_path/'silent_sampling_change', 'cpu')
        with pytest.raises(ValueError, match='actually change'):
            train_one(config | dict(resume_dagger=str(half), transfer_dagger_sampling=True),
                0, 2, tmp_path/'noop_sampling_transfer', 'cpu')
        sampled = tmp_path/'sampling_branch'
        sampled_result = train_one(config | dict(resume_dagger=str(half), stage_sampling_fraction=.5,
            transfer_dagger_sampling=True), 0, 2, sampled, 'cpu')
        assert sampled_result['complete'] and sampled_result['source_stage_sampling_fraction'] == 0.
        assert sampled_result['fitting_boards'] == continued['fitting_boards']
        assert digest(half/'training.pt') == source_hash
        with np.load(sampled/'round_002.npz') as a, np.load(tmp_path/'continued/round_002.npz') as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
        assert sum(json.loads((sampled/'progress.json').read_text())[-1]['sampling_stage_counts']) == sampled_result['fitting_boards']
        with (half/'round_001.npz').open('ab') as stream:
            stream.write(b'changed')
        with pytest.raises(ValueError, match='hash changed'):
            train_one(config | dict(resume_dagger=str(half)), 0, 2, tmp_path/'changed_block', 'cpu')
