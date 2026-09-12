"""Bounded own-game policy-gradient study starting from the original CNN."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.afterstate_teacher import digest, write
from research.cnn_dagger_study import enqueue


BASE = Path('runs/research/scaled_transformer').resolve()
PROTOCOL = BASE/'cnn_reinforce_protocol.json'


def prepare():
    if PROTOCOL.exists():
        raise ValueError('Preserve the existing CNN REINFORCE study')
    source = Path('runs/research/pretrained_cnn/original').resolve()
    imitation = json.loads((BASE/'cnn_anchored_vs_original.json').read_text())
    question = ('Longer teacher imitation reduced the original CNN game score. Test actual '
        'own-game policy gradients instead: complete sampled games, undiscounted raw merge '
        'points/128, no teacher labels, replay, critic, TD targets or strategic shaping. '
        'Both full arms start the original frozen-source CNN and fresh Adam. Compare ordinary '
        'REINFORCE with a leave-one-out baseline from other games at the same move index. '
        'The baseline can reduce update noise without supplying a teacher or learned value. '
        'Pretraining is inherited PPO, so this is fine-tuning, not REINFORCE from scratch. '
        'Two diagnostic updates on16games each gate an8-update,64games/update pair. '
        'Equal complete-game and optimizer-update budgets, not equal transitions or time. '
        'Every sample is used exactly once at unchanged weights before one optimizer step. '
        'Monitor greedy and sampled policies separately; sampled-return improvement does '
        'not establish greedy improvement. One adaptation seed and reused selection games.')
    config = dict(algorithm='reinforce_loo', baseline_mode='leave_one_out_time',
        architecture='pretrained_ml2048', width=1024, depth=None, heads=None,
        input_encoding='one_hot', initial_actor_checkpoint=str(source),
        gamma=1., reward_mode='score', episode_batch=16, batch=512,
        lr=1e-5, max_grad_norm=.5, complete_batches=2,
        collection_seed=12600000, monitor_seed_start=8900000,
        final_seed_start=8910000, monitor_games=128, monitor_interval=1,
        monitor_sampled=True, sampling_seed=8958700, action_temperature=1.,
        max_training_seconds=300, monitor_report='research.report_cnn_reinforce',
        stop_file=str(BASE/'STOP'), diagnostic_only=True,
        experiment_group='pretrained_cnn_own_game_reinforce')
    smoke = dict(id='cnn_reinforce_loo_smoke_seed0', config=config,
        seed=0, steps=1, status='pending', question=question)
    protocol = dict(question=question, source=str(source), source_sha256=digest(source/'policy.pt'),
        preceding_imitation=imitation, smoke=smoke, full_jobs=[], stage='diagnostic_queued',
        validation='32 targeted REINFORCE, CNN and neural tests passed in46.20s before this MPS diagnostic.',
        budget_note='Complete-batch budget overrides the legacy steps argument. Requested transition budget is null. '
        'Training time excludes monitoring; final evaluations are recorded separately.',
        baseline_monitor=json.loads((BASE/'dagger_stage_diagnostic/cnn_monitor.json').read_text())['summary'],
        baseline_selection=json.loads((BASE/'dagger_stage_diagnostic/cnn_selection.json').read_text())['summary'])
    write(PROTOCOL, protocol)
    enqueue([smoke])


def verify_smoke(protocol_path=PROTOCOL):
    protocol = json.loads(Path(protocol_path).read_text())
    folder = BASE/protocol['smoke']['id']
    result = json.loads((folder/'result.json').read_text())
    assert result['complete'] and result['completed_budget'] and result['updates'] == 2
    assert result['completed_training_games'] == 32 and result['discarded_transitions'] == 0
    assert not any(result[k] for k in ('critic', 'bootstrap', 'replay_buffer', 'teacher_labels_used_online'))
    assert digest(Path(protocol['source'])/'policy.pt') == protocol['source_sha256']
    original = torch.load(Path(protocol['source'])/'policy.pt', map_location='cpu', weights_only=True)
    state = torch.load(folder/'training.pt', map_location='cpu', weights_only=True)
    assert all(torch.equal(v, state['policy'][k]) for k,v in original.items() if k.startswith('_critic.'))
    assert any(not torch.equal(v, state['policy'][k]) for k,v in original.items() if k.startswith('_encoder.'))
    assert all(v['step'] == 2 for v in state['optimizer']['state'].values())
    progress = json.loads((folder/'progress.json').read_text())
    assert len(progress) == 2
    assert max(p['max_behavior_logp_difference'] for p in progress) < .01
    assert all(np.isfinite(p['policy_loss']) and np.isfinite(p['gradient_norm_before_clip']) for p in progress)
    first = json.loads((folder/'first_training_game.json').read_text())
    assert sum(t['reward_points'] for t in first['transitions']) == first['episode']['score']
    assert first['transitions'][0]['actual_return_points'] == first['episode']['score']
    curves = json.loads((folder/'curve.json').read_text())
    assert len(curves) == 3 and all(c['complete'] for c in curves)
    for name in ('evaluation.json', 'sampled_evaluation.json'):
        evaluation = json.loads((folder/name).read_text())
        assert evaluation['summary']['complete'] and len(evaluation['episodes']) == 100
        assert all(e['complete'] and not e['truncated'] for e in evaluation['episodes'])
    assert (folder/'replay_last.html').exists()
    stable = all(curves[-1][k] >= .8*curves[0][k] for k in ('mean_score', 'sampled_mean_score'))
    verified = dict(implementation_verified=True, source_and_unused_critic_unchanged=True,
        full_complete_batches=True, adam_steps=2,
        max_behavior_logp_difference=max(p['max_behavior_logp_difference'] for p in progress),
        initial_greedy=curves[0]['mean_score'], final_greedy=curves[-1]['mean_score'],
        initial_sampled=curves[0]['sampled_mean_score'], final_sampled=curves[-1]['sampled_mean_score'],
        no_large_monitor_regression=stable,
        note='The80% screen prevents obvious collapse, not a statistical safety guarantee or evidence of improvement.')
    write(folder/'verified.json', verified)
    return protocol, verified


def full_pair():
    protocol, verified = verify_smoke()
    if protocol['full_jobs']:
        raise ValueError('Full pair already queued')
    if not verified['no_large_monitor_regression']:
        raise ValueError('Diagnostic regressed substantially; inspect before extending this recipe')
    config = protocol['smoke']['config'] | dict(episode_batch=64, complete_batches=8,
        max_training_seconds=1200, diagnostic_only=False)
    jobs = [dict(id=f'cnn_reinforce_{label}_seed0', seed=0, steps=1, status='pending',
        question=protocol['question'], config=config | dict(algorithm=algorithm, baseline_mode=baseline))
        for label,algorithm,baseline in [('plain', 'reinforce', 'none'),
                                         ('loo', 'reinforce_loo', 'leave_one_out_time')]]
    protocol.update(full_jobs=jobs, diagnostic_verification=verified, stage='full_pair_queued')
    write(PROTOCOL, protocol)
    enqueue(jobs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'verify_smoke', 'full_pair'])
    globals()[parser.parse_args().mode]()
