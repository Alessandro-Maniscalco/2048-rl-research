"""One-factor encoder-transfer comparison on stronger native expert actions."""
import argparse
import json
from pathlib import Path

from research.afterstate_teacher import write, digest
from research.cnn_dagger_study import enqueue

BASE=Path('runs/research/scaled_transformer').resolve()
PROTOCOL=BASE/'native_policy_protocol.json'


def prepare():
    if PROTOCOL.exists():raise ValueError('Study already prepared')
    protocol=dict(stage='dataset_preparation',dataset=str(BASE/'native_expert_data'),
        source_checkpoint=str(Path('runs/research/pretrained_cnn/original').resolve()),
        native_root=str(Path('runs/research/tablebase_search').resolve()),
        stop_file=str(BASE/'STOP'),expert_seeds=list(range(8950000,8950080)),
        split='First80record500games in seed order, every fifth whole game held out.64fit/16holdout. No score selection.',
        question='Does the stronger native search teacher provide useful action supervision, and does a pretrained encoder help? '
            'Same18-category CNN, identical fresh actor heads, same minibatches/Adam/budgets; only encoder initialization differs. '
            'Full-game greedy raw score is the primary policy metric. Held-out action accuracy is a diagnostic. '
            'Native games are reused as training data, so their scores are not independent student validation. '
            'No critic, reward shaping, Q targets, online correction, symmetry augmentation or inference search. '
            'Rank16/17receive separate trainable channels so large expert tiles are not clipped to32768. '
            'Encoder transfer copies pretrained features while preserving features on original0..15boards; '
            'the actor is reset in both arms to avoid forcing the previous CNN policy onto a different expert strategy.',
        planned_full_updates=4096,full_training_seconds=900,smoke_jobs=[],full_jobs=[])
    protocol['source_sha256']=digest(Path(protocol['source_checkpoint'])/'policy.pt')
    write(PROTOCOL,protocol)


def smoke():
    p=json.loads(PROTOCOL.read_text())
    if p['smoke_jobs']:raise ValueError('Diagnostic already queued')
    data=json.loads((Path(p['dataset'])/'manifest.json').read_text())
    assert data['complete'] and data['all_transitions_rechecked']
    config=dict(algorithm='native_behavior_cloning',architecture='cnn_policy18',width=1024,
        input_encoding='one_hot',depth=None,heads=None,teacher_only=True,diagnostic_only=True,
        expert_dataset=p['dataset'],source_checkpoint=p['source_checkpoint'],initialization='encoder',
        batch=512,lr=1e-4,max_grad_norm=.5,validation_boards=8192,
        monitor_seed_start=8900000,final_seed_start=8910000,monitor_games=128,final_games=100,
        monitor_interval=128,max_training_seconds=300,stop_file=p['stop_file'],
        monitor_report='research.report_native_policy',experiment_group='native_expert_imitation',
        reward_mode='none_action_labels',gamma=None)
    job=dict(id='native_policy_encoder_smoke_seed0',seed=0,steps=128,status='pending',
             question=p['question'],config=config)
    p.update(stage='diagnostic_queued',smoke_jobs=[job],data_sha256=data['data_sha256'])
    write(PROTOCOL,p);enqueue([job])


def full():
    p=json.loads(PROTOCOL.read_text())
    if p['full_jobs']:raise ValueError('Full pair already queued')
    smoke=p['smoke_jobs'][0];folder=BASE/smoke['id']
    result=json.loads((folder/'result.json').read_text())
    assert result['complete'] and result['completed_budget'] and result['updates']==128
    assert result['checkpoint_reload_verified'] and result['source_unchanged'] and result['dataset_unchanged']
    curve=json.loads((folder/'curve.json').read_text())
    assert curve[-1]['action_cross_entropy']<curve[0]['action_cross_entropy']
    verified=json.loads((folder/'verified.json').read_text())
    assert verified['all_replay_actions_legal'] and verified['cpu_actions_match']
    config=smoke['config']|dict(diagnostic_only=False,monitor_interval=1024,
        max_training_seconds=p['full_training_seconds'])
    jobs=[dict(id=f'native_policy_{initialization}_seed0',seed=0,steps=p['planned_full_updates'],
        status='pending',question=p['question'],config=config|dict(initialization=initialization))
        for initialization in ('encoder','scratch')]
    p.update(stage='matched_pair_queued',full_jobs=jobs,diagnostic_verification=verified)
    write(PROTOCOL,p);enqueue(jobs)


def verify():
    import numpy as np
    import torch
    from research.native_policy_data import extract_game
    from rl2048.agents.neural import NeuralAgent, tensor_boards
    p=json.loads(PROTOCOL.read_text());torch.set_num_threads(1)
    for job in p['smoke_jobs']+p['full_jobs']+p.get('stage_jobs',[]):
        folder=BASE/job['id']
        if not (folder/'result.json').exists():continue
        result=json.loads((folder/'result.json').read_text())
        assert result['complete'] and result['completed_budget']
        states,actions,legal,replay=extract_game(folder/'replay_last.json')
        agent=NeuralAgent.load(folder/'last','cpu')
        ids=np.unique(np.r_[np.linspace(0,len(actions)-1,16,dtype=int),
                              np.arange(max(0,len(actions)-8),len(actions))])
        with torch.no_grad():
            logits=agent.policy(tensor_boards(states[ids],'cpu')).numpy()
        choices=np.where(legal[ids],logits,-np.inf).argmax(1)
        checkpoint=torch.load(folder/'training.pt',map_location='cpu',weights_only=True)
        assert {int(v['step']) for v in checkpoint['optimizer']['state'].values()}=={job['steps']}
        assert not any('critic' in key for key in checkpoint['policy'])
        verified=dict(all_replay_actions_legal=True,all_replay_transitions_rechecked=True,
            checked_decisions=len(ids),cpu_actions_match=bool(np.array_equal(choices,actions[ids])),
            source_unchanged=digest(Path(p['source_checkpoint'])/'policy.pt')==p['source_sha256'],
            dataset_unchanged=digest(Path(p['dataset'])/'data.npz')==p['data_sha256'],
            replay_score=replay['score'],replay_moves=len(actions),adam_updates=job['steps'],
            critic_absent=True)
        write(folder/'verified.json',verified);print(job['id'],json.dumps(verified))


def stages():
    p=json.loads(PROTOCOL.read_text())
    if p.get('stage_jobs'):raise ValueError('Stage comparison already queued')
    for job in p['full_jobs']:
        r=json.loads((BASE/job['id']/'result.json').read_text())
        assert r['complete'] and r['completed_budget']
    diagnosis=json.loads((BASE/'native_policy_stage_diagnostic.json').read_text())
    source=BASE/'native_policy_encoder_seed0/last'
    question=('Uniform expert-action teaching improved held-out agreement but both students still play poorly. '
        'Only1.16%of fitting boards have max tile<=256 and82.64%have>=8192. Hypothesis: too little early-game '
        'practice prevents the student reaching the stages dominating expert trajectories. Continue BOTH arms '
        'from the exact same4096-update encoder student, fresh Adam, LR1e-4, batch512,16384additional updates '
        'or900trainingseconds. Control samples uniformly; treatment draws half uniformly and half from an '
        'equal mixture of max-tile stages<=256,512..4096,>=8192. This lifts early-stage sampling to17.25%. '
        'Only sampling differs; no augmentation, new data, critic, policy filter or search. '
        'The control also tests whether the earlier1.5passes simply undertrained. Longer training costs are '
        'explicit; compare both final100game endpoints and curves, not only held-out action agreement. '
        'Data imbalance is a measured hypothesis, not established cause of poor play.')
    config=p['full_jobs'][0]['config']|dict(initialization='continuation',continuation_checkpoint=str(source),
        monitor_interval=4096,experiment_group='native_expert_stage_sampling')
    jobs=[dict(id=f'native_policy_continue_{label}_seed0',seed=0,steps=16384,status='pending',
        question=question,config=config|dict(stage_fraction=fraction))
        for label,fraction in [('uniform',0.),('stages',.5)]]
    p.update(stage='stage_sampling_pair_queued',stage_jobs=jobs,stage_question=question,
        stage_diagnostic=diagnosis,continuation_source_sha256=digest(source/'policy.pt'))
    write(PROTOCOL,p);enqueue(jobs)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','smoke','verify','full','stages'])
    globals()[parser.parse_args().action]()
