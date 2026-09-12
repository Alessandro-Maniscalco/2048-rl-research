"""Does actual-return training improve the already useful frozen safety policy?"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.afterstate_teacher import write, digest
from research.cnn_dagger_study import enqueue
from rl2048.agents.neural import NeuralAgent
from rl2048.agents.ntuple import encode
from rl2048.agents.spawn_safety import minimum_risk_mask


BASE=Path('runs/research/scaled_transformer').resolve()
PROTOCOL=BASE/'cnn_filtered_reinforce_protocol.json'


def prepare():
    if PROTOCOL.exists():
        raise ValueError('Keep the existing filtered-policy study')
    original=json.loads((BASE/'cnn_reinforce_protocol.json').read_text())
    comparison=json.loads((BASE/'cnn_reinforce_comparison.json').read_text())
    assert comparison['complete'] and comparison['identical_first_complete_batch']
    replicated=json.loads((BASE/'cnn_spawn_safety_validation200/comparison.json').read_text())
    assert replicated['complete']
    base_config=next(j['config'] for j in original['full_jobs'] if j['config']['algorithm']=='reinforce_loo')
    config=base_config|dict(spawn_safety=True, evaluate_initial_policy=True, complete_batches=2, episode_batch=16,
        max_training_seconds=300, diagnostic_only=True,
        experiment_group='pretrained_cnn_filtered_reinforce',monitor_report='research.report_cnn_filtered_reinforce')
    question=('The unchanged original CNN improved on two independent spawn-safety evaluations. '
        'Test whether actual-return REINFORCE with a leave-one-out baseline can improve that '
        'filtered policy, while training on the states it actually visits. At each move restrict '
        'softmax support to legal actions with minimum exact next-spawn death probability. '
        'Store both actual legal masks and restricted policy masks. Use the same restricted '
        'distribution for collection, gradients, evaluation and restored checkpoint playback. '
        'Rewards remain actual points/128, gamma1. No teacher labels, value critic or TD targets. '
        'The initial filtered checkpoint is the within-run frozen control; improvement over '
        'the unfiltered player alone cannot count as learning. Compare the full arm with the '
        'completed unfiltered baseline run, noting that only the action restriction intentionally '
        'changes and first trajectories are expected to differ. Same original weights, fresh Adam, '
        'LR1e-5,64complete games/update,512-board gradient chunks and8updates. '
        'The restriction can exclude an action with better long-term return; it is a tested '
        'heuristic rather than a theorem of optimality. No automatic longer continuation.')
    smoke=dict(id='cnn_reinforce_filtered_smoke_seed0',seed=0,steps=1,status='pending',config=config,question=question)
    protocol=dict(question=question,source=original['source'],source_sha256=original['source_sha256'],
        smoke=smoke,full_jobs=[],stage='diagnostic_queued',reference_job='cnn_reinforce_loo_seed0',
        frozen_replication=replicated,
        budget_note='One Adam update per complete-game batch. Legacy steps field ignored when complete_batches is set. '
        'All monitor and final evaluation games begin from standard two-tile boards. '
        'Initial/best metadata and checkpoint action-mask flags remain explicit.')
    assert digest(Path(protocol['source'])/'policy.pt')==protocol['source_sha256']
    write(PROTOCOL,protocol)
    enqueue([smoke])


def full():
    protocol=json.loads(PROTOCOL.read_text())
    if protocol['full_jobs']:
        raise ValueError('Full arm already queued')
    folder=BASE/protocol['smoke']['id']
    verified=json.loads((folder/'verified.json').read_text())
    assert verified['implementation_verified'] and verified['no_large_monitor_regression']
    config=protocol['smoke']['config']|dict(episode_batch=64,complete_batches=8,
        max_training_seconds=1200,diagnostic_only=False)
    job=dict(id='cnn_reinforce_filtered_seed0',seed=0,steps=1,status='pending',config=config,question=protocol['question'])
    protocol.update(full_jobs=[job],stage='full_arm_queued',diagnostic_verification=verified)
    write(PROTOCOL,protocol)
    enqueue([job])


def verify():
    from research.cnn_reinforce_study import verify_smoke
    protocol,verified=verify_smoke(PROTOCOL)
    folder=BASE/protocol['smoke']['id']
    player=NeuralAgent.load(folder/'last','cpu')
    initial=NeuralAgent.load(folder/'initial','cpu')
    assert player.spawn_safety and initial.spawn_safety
    source=torch.load(Path(protocol['source'])/'policy.pt',map_location='cpu',weights_only=True)
    assert all(torch.equal(v,initial.policy.state_dict()[k]) for k,v in source.items())
    first=json.loads((folder/'first_training_game.json').read_text())
    transitions=first['transitions']
    boards=np.array([t['board_exponents'] for t in transitions],dtype=np.uint8)
    legal=np.array([t['legal_mask'] for t in transitions],dtype=bool)
    masks=np.array([t['policy_mask'] for t in transitions],dtype=bool)
    assert np.array_equal(masks,minimum_risk_mask(boards,legal))
    actions=np.array([t['action'] for t in transitions])
    assert masks[np.arange(len(actions)),actions].all()
    replay=json.loads((folder/'replay_last.json').read_text())
    boards=np.array([encode(np.array(f['board'])) for f in replay['frames'][:-1]],dtype=np.uint8)
    masks=minimum_risk_mask(boards)
    actions=np.array([f['action'] for f in replay['frames'][1:]])
    assert masks[np.arange(len(actions)),actions].all()
    assert replay['result']['terminated'] and not replay['result']['truncated']
    verified.update(policy_filter_restored=True,initial_control_exact_source_weights=True,
        first_training_game_policy_masks_verified=True,every_replay_action_minimum_risk=True,
        replay_score=replay['result']['score'],replay_moves=len(actions))
    write(folder/'verified.json',verified)
    print(json.dumps(verified))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','verify','full'])
    globals()[parser.parse_args().mode]()
