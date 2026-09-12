"""Summarize the prospectively matched uniform/stage-balanced sampling pair."""
import json
from pathlib import Path
import numpy as np

from research.afterstate_teacher import digest, write


def compare(base=Path('runs/research/scaled_transformer')):
    folders = {arm:base/f'transformer_dagger_stage_{arm}_seed0'
               for arm in ('uniform', 'balanced')}
    if not all((p/'result.json').exists() for p in folders.values()):
        return None
    results = {arm:json.loads((p/'result.json').read_text()) for arm,p in folders.items()}
    if not all(r.get('complete') and r.get('completed_budget') for r in results.values()):
        return None  # Do not silently treat different update counts as matched.
    a, b = results['uniform'], results['balanced']
    shared = ('resume_dagger', 'resumed_policy_sha256', 'inherited_fitting_boards',
        'initial_actor_sha256', 'data_sha256', 'teacher_sha256', 'batch', 'lr',
        'max_grad_norm', 'collection_games', 'updates_per_round', 'collection_seed',
        'teacher_gap_weight', 'teacher_gap_scale_points', 'seed', 'width', 'depth',
        'heads', 'input_encoding', 'final_seed_start', 'updates', 'max_training_seconds')
    assert all(a[k]==b[k] for k in shared)
    assert a['updates']==8192 and a['stage_sampling_fraction']==0 and b['stage_sampling_fraction']==.5
    assert not any(r.get('diagnostic_only') for r in results.values())
    assert digest(folders['uniform']/'round_129.npz')==digest(folders['balanced']/'round_129.npz')
    scores = {}
    for arm,p in folders.items():
        episodes = json.loads((p/'evaluation.json').read_text())['episodes']
        assert len(episodes)==100 and all(r['complete'] and not r['truncated'] for r in episodes)
        scores[arm] = {r['seed']:r['score'] for r in episodes}
        assert len(scores[arm])==100 and np.mean(list(scores[arm].values()))==results[arm]['mean_score']
    assert scores['uniform'].keys()==scores['balanced'].keys()
    difference = np.array([scores['balanced'][s]-scores['uniform'][s] for s in sorted(scores['uniform'])])
    boot = difference[np.random.default_rng(7132).integers(100,size=(20000,100))].mean(1)
    payload = dict(uniform_mean=a['mean_score'], balanced_mean=b['mean_score'],
        mean_paired_difference=float(difference.mean()),
        paired_game_bootstrap_95_interval=np.quantile(boot,[.025,.975]).tolist(),
        wins=int((difference>0).sum()), ties=int((difference==0).sum()),
        matched_updates=a['updates'], shared_source=a['resume_dagger'],recipe_checks=list(shared),
        training_seconds={arm:r['training_seconds'] for arm,r in results.items()},
        training_transitions={arm:r['training_transitions'] for arm,r in results.items()},
        note='One fine-tuning seed, same pretrained actor and reused selection games. The paired interval reflects game variation, not training-seed variation. Final endpoints only; diagnostic and selected-best snapshots excluded. Stage mixture deliberately changes fitting distribution without changing inputs, labels, loss or teacher. Collection trajectories may diverge after the first fitting round.')
    write(base/'dagger_stage_comparison.json',payload)
    return payload


if __name__=='__main__':
    print(json.dumps(compare(),indent=2))
