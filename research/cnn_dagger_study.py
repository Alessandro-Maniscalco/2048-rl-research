"""Prespecified CNN adaptation pair; a verified diagnostic gates full jobs."""
import argparse
import json
from pathlib import Path

import torch

from research.afterstate_teacher import digest, write


BASE = Path('runs/research/scaled_transformer').resolve()


def enqueue(jobs):
    manifest = json.loads((BASE/'manifest.json').read_text())
    previous = {j['id'] for j in manifest['jobs']}
    if any(j['id'] in previous or (BASE/j['id']).exists() for j in jobs):
        raise ValueError('Existing attempt preserved; refusing duplicate job')
    manifest['jobs'].extend(jobs)
    write(BASE/'manifest.json', manifest)


def prepare():
    path = BASE/'cnn_dagger_protocol.json'
    if path.exists():
        raise ValueError('CNN study already prepared')
    diagnostic = BASE/'dagger_stage_diagnostic'
    assert json.loads((diagnostic/'verified.json').read_text())['weights_and_data_unchanged']
    source = Path('runs/research/pretrained_cnn/original').resolve()
    question = ('Can a stronger externally pretrained CNN improve by receiving frozen teacher '
        'corrections on its own complete-game states? Compare original fixed fitting data with '
        'original data plus every student-collected round. Start both from the identical external '
        'actor and a fresh Adam optimizer, with cost-aware action teaching, batch128, LR1e-5, '
        '8192 updates and 1200 training seconds per arm. The lower learning rate is intended to '
        'reduce destruction of the already useful policy. Uniform board sampling, no new symmetry '
        'ensemble, no search and no critic training. CNN was rescored at81622.96 on the same100 '
        'selection seeds and85014.47 on128monitor seeds before this study. This tests adaptation '
        'within one pretrained CNN, not CNN versus Transformer architecture at equal pretraining. '
        'The512-update diagnostic is excluded and never used to initialize either full arm. '
        'Judge final complete-game scores and inspect mistakes; lower imitation loss alone is insufficient.')
    c = dict(algorithm='dagger', architecture='pretrained_ml2048', width=1024,
        input_encoding='one_hot', depth=None, heads=None,
        initial_actor_checkpoint=str(source), teacher_dataset=str(BASE/'transformer_teacher_data'),
        teacher_checkpoint=str(BASE/'transformer_teacher_data/teacher'),
        teacher_only=False, diagnostic_only=False, gamma=1., reward_mode='score',
        batch=128, lr=1e-5, max_grad_norm=.5, updates_per_round=512, collection_games=64,
        collection_seed=12500000, monitor_seed_start=8900000, final_seed_start=8910000,
        monitor_games=128, validation_boards=2048, monitor_every_rounds=2,
        max_training_seconds=1200, monitor_report='research.report_cnn_dagger',
        stop_file=str(BASE/'STOP'), teacher_gap_weight=1., teacher_gap_scale_points=16384.,
        stage_sampling_fraction=0., experiment_group='pretrained_cnn_dagger')
    jobs = [dict(id=f'cnn_dagger_{mode}_seed0', seed=0, steps=8192, status='pending',
                 question=question, config=c | dict(data_source=mode)) for mode in ('fixed', 'student')]
    smoke = dict(id='cnn_dagger_smoke_seed0', seed=0, steps=512, status='pending',
        question='Diagnostic only: verify actor-only CNN collection, teacher labels, MPS updates, save/reload, source and critic preservation before the full matched pair.',
        config=c | dict(data_source='student', diagnostic_only=True, max_training_seconds=300))
    write(path, dict(question=question, source_sha256=digest(source/'policy.pt'),
        baseline_selection=json.loads((diagnostic/'cnn_selection.json').read_text())['summary'],
        baseline_monitor=json.loads((diagnostic/'cnn_monitor.json').read_text())['summary'],
        pretrained_input_limit='16 tile categories, ranks above15 clipped to15; inherited, no engineered features',
        stage='diagnostic_queued', smoke=smoke, jobs=jobs))
    enqueue([smoke])


def promote():
    path = BASE/'cnn_dagger_protocol.json'
    protocol = json.loads(path.read_text())
    if protocol['stage'] != 'diagnostic_queued':
        raise ValueError('Study has already advanced')
    out = BASE/protocol['smoke']['id']
    result = json.loads((out/'result.json').read_text())
    assert result['complete'] and result['completed_budget'] and result['updates'] == 512
    assert result['training_transitions'] == result['labelled_transitions'] > 0
    assert result['reinforcement_learning'] is False and result['critic'] is False
    source = Path(result['initial_actor_checkpoint'])
    assert digest(source/'policy.pt') == protocol['source_sha256'] == result['initial_actor_sha256']
    original = torch.load(source/'policy.pt', map_location='cpu', weights_only=True)
    trained = torch.load(out/'last/policy.pt', map_location='cpu', weights_only=True)
    for key in original:
        if key.startswith('_critic.'):
            assert torch.equal(original[key], trained[key])
    assert any(not torch.equal(original[k], trained[k]) for k in original if k.startswith('_encoder.'))
    first = json.loads((out/'curve.json').read_text())[0]
    assert first['mean_score'] == protocol['baseline_monitor']['mean_score']
    last = json.loads((out/'curve.json').read_text())[-1]
    if last['mean_score'] < .8*protocol['baseline_monitor']['mean_score']:
        raise ValueError('Severe diagnostic policy regression; do not promote this recipe')
    protocol['diagnostic_verification'] = dict(source_unchanged=True, unused_critic_unchanged=True,
        encoder_updated=True, initial_monitor_exactly_matches_frozen_baseline=True,
        completed_updates=512, full_arms_start_original_not_diagnostic=True)
    enqueue(protocol['jobs'])
    protocol['stage'] = 'matched_pair_queued'
    write(path, protocol)


def stability():
    """A matched small diagnostic asks whether reference KL prevents collapse."""
    path = BASE/'cnn_dagger_protocol.json'
    protocol = json.loads(path.read_text())
    if protocol['stage'] != 'diagnostic_queued':
        raise ValueError('Unexpected study stage or duplicate stability jobs')
    smoke_path = BASE/protocol['smoke']['id']
    result = json.loads((smoke_path/'result.json').read_text())
    assert result['complete'] and result['completed_budget']
    curve = json.loads((smoke_path/'curve.json').read_text())
    assert curve[-1]['mean_score'] < .8*curve[0]['mean_score']
    protocol['stage'] = 'original_full_pair_cancelled_after_diagnostic_collapse'
    protocol['original_recipe_rejection'] = dict(initial_monitor=curve[0]['mean_score'],
        final_monitor=curve[-1]['mean_score'], final_selection=result['mean_score'],
        initial_teacher_loss=curve[0]['hard_action_loss'], final_teacher_loss=curve[-1]['hard_action_loss'],
        reason='Teacher loss improved while complete-game performance collapsed. Do not spend full budgets on this recipe.')
    for job in protocol['jobs']:
        job['status'] = 'not_run_diagnostic_regression'
    question = ('The pretrained CNN collapsed in512updates atLR1e-5 despite lower imitation loss. '
        'Hypothesis: teacher adaptation is changing an already confident useful policy too much. '
        'At LR1e-6 compare no policy anchor with100times KL(frozen original policy || student). '
        'Both start from identical original weights, fresh Adam, same first64complete games, '
        'same original fitting data plus own boards, teacher CE and mistake cost, batch128 and512updates. '
        'The reference is the starting CNN, not the n-tuple teacher. This is regularization, not '
        'a hard trust-region guarantee or PPO. Measure full-game score, teacher loss and KL. '
        'No full training promotion before inspecting both diagnostics and policy transitions.')
    jobs = []
    for label,weight in [('small_step', 0.), ('kl_anchor', 100.)]:
        config = protocol['smoke']['config'] | dict(lr=1e-6, reference_kl_weight=weight,
            experiment_group='pretrained_cnn_stability', monitor_every_rounds=1)
        jobs.append(dict(id=f'cnn_dagger_{label}_seed0', config=config, question=question,
                         steps=512, seed=0, status='pending'))
    protocol['stability_question'] = question
    protocol['stability_jobs'] = jobs
    write(path, protocol)
    enqueue(jobs)


def anchored_pair():
    """Full adaptation test starts original weights, after the stability audit."""
    path = BASE/'cnn_dagger_protocol.json'
    protocol = json.loads(path.read_text())
    if protocol.get('anchored_jobs'):
        raise ValueError('Anchored comparison already queued')
    comparisons = json.loads((BASE/'cnn_stability_comparison.json').read_text())
    assert comparisons['identical_first_collection']
    audit = json.loads((BASE/'reinforce_diagnostics/cnn_kl_anchor_last/audit.json').read_text())['summary']
    assert audit['checked_decisions'] == audit['cpu_action_matches'] > 0
    source = Path(protocol['smoke']['config']['initial_actor_checkpoint'])
    assert digest(source/'policy.pt') == protocol['source_sha256']
    original = torch.load(source/'policy.pt', map_location='cpu', weights_only=True)
    for label in ('small_step', 'kl_anchor'):
        folder = BASE/f'cnn_dagger_{label}_seed0'
        result = json.loads((folder/'result.json').read_text())
        assert result['complete'] and result['completed_budget'] and result['updates'] == 512
        saved = torch.load(folder/'last/policy.pt', map_location='cpu', weights_only=True)
        assert all(torch.equal(saved[k], original[k]) for k in original if k.startswith('_critic.'))
    config = protocol['stability_jobs'][1]['config'] | dict(diagnostic_only=False,
        max_training_seconds=1200, monitor_every_rounds=2, experiment_group='pretrained_cnn_anchored')
    question = ('KL-anchored512-update diagnostic preserved the128-game monitor and scored90371.92 '
        'on100selection games, compared with68407.24without KL and81622.96original. Gain over original '
        'is uncertain. Test sustained adaptation with the stable recipe: original CNN weights, '
        'fresh Adam, LR1e-6, KL(original||student)weight100, teacher-action CE plus teacher cost1, '
        'batch128,8192updates/1200trainingseconds per arm. Compare original fixed teacher data with '
        'original plus every64complete student-game boards per512updates. Both start the original '
        'checkpoint, never diagnostic weights. Only data collection differs. No search or safety '
        'filter is used. Judge final whole-game scores, training curves and transition audits. '
        'One adaptation seed and reused selection games; no automatic extension after this pair.')
    jobs = [dict(id=f'cnn_dagger_anchored_{mode}_seed0', seed=0, steps=8192,
                 status='pending', question=question, config=config | dict(data_source=mode))
            for mode in ('student', 'fixed')]
    protocol['anchored_question'] = question
    protocol['anchored_jobs'] = jobs
    protocol['stage'] = 'anchored_pair_queued'
    write(path, protocol)
    enqueue(jobs)


def cost_only():
    """Test whether forcing every teacher argmax is the damaging part of teaching."""
    path = BASE/'cnn_dagger_protocol.json'
    protocol = json.loads(path.read_text())
    if protocol.get('cost_only_jobs'):
        raise ValueError('Cost-only diagnostic already queued')
    for job in protocol['anchored_jobs']:
        result = json.loads((BASE/job['id']/'result.json').read_text())
        assert result['complete'] and result['completed_budget'] and result['updates'] == 8192
    config = protocol['stability_jobs'][1]['config'] | dict(teacher_action_weight=0.,
        experiment_group='pretrained_cnn_cost_only', diagnostic_only=True)
    question = ('Longer KL-anchored teaching finished70251with own boards versus20327fixed, '
        'both below original81623. Hypothesis: hard action cross-entropy forces changes even '
        'when the teacher advantage is tiny; the policy-loss reduction is not aligned with game score. '
        'Remove only the hard-action term. Keep expected teacher mistake cost1, frozen-original '
        'KL100, LR1e-6, original weights/freshAdam, same64complete games and original+student '
        'fitting data, batch128 and512updates. Compare with the already completed matched KL '
        'diagnostic, verifying identical first collection. The cost gradient can vanish for a '
        'confident policy, so inspect actual policy change as well as score. Diagnostic only, '
        'no automatic long continuation or claim of improvement from one mean.')
    job = dict(id='cnn_dagger_cost_only_seed0', config=config, question=question,
               steps=512, seed=0, status='pending')
    protocol['cost_only_question'] = question
    protocol['cost_only_jobs'] = [job]
    protocol['stage'] = 'cost_only_diagnostic_queued'
    write(path, protocol)
    enqueue([job])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'promote', 'stability', 'anchored_pair', 'cost_only'])
    args = p.parse_args()
    {'prepare':prepare, 'promote':promote, 'stability':stability,
     'anchored_pair':anchored_pair, 'cost_only':cost_only}[args.action]()
