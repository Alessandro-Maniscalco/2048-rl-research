"""Reproduce the two prespecified endpoints of the same cost-loss contrast."""
import json
from pathlib import Path
import numpy as np

from research.afterstate_teacher import write


def compare(base=Path('runs/research/scaled_transformer')):
    shared = ('initial_actor_sha256', 'teacher_sha256', 'data_sha256', 'batch',
        'lr', 'max_grad_norm', 'collection_games', 'updates_per_round', 'seed',
        'width', 'depth', 'heads', 'input_encoding', 'teacher_gap_scale_points',
        'collection_seed', 'final_seed_start')
    first = None
    for suffix, updates in [('', 8192), ('_cont', 16384)]:
        folders = {arm:base/f'transformer_dagger_cost_{arm}{suffix}_seed0'
                   for arm in ('control', 'hybrid')}
        if not all((p/'result.json').exists() for p in folders.values()):
            continue
        results = {a:json.loads((p/'result.json').read_text()) for a,p in folders.items()}
        a, b = results['control'], results['hybrid']
        assert all(r['complete'] and r['completed_budget'] and r['updates']==8192 for r in (a,b))
        assert not any(r.get('diagnostic_only') for r in (a,b))
        assert all(a[k]==b[k] for k in shared)
        assert a['teacher_gap_weight']==0 and b['teacher_gap_weight']==1
        if not suffix:
            assert a['resume_dagger']==b['resume_dagger']
            assert a['resumed_policy_sha256']==b['resumed_policy_sha256']
            assert a['inherited_fitting_boards']==b['inherited_fitting_boards']
            first = results
        else:
            assert first is not None
            for arm,r in results.items():
                assert Path(r['resume_dagger']).resolve()==(base/f'transformer_dagger_cost_{arm}_seed0').resolve()
                assert r['total_dagger_updates']==first[arm]['total_dagger_updates']+8192
                assert not r.get('transfer_dagger_objective')
        games = {arm:json.loads((p/'evaluation.json').read_text())['episodes']
                 for arm,p in folders.items()}
        for arm, rows in games.items():
            assert len(rows)==100 and len({r['seed'] for r in rows})==100
            assert all(r['complete'] and not r['truncated'] for r in rows)
            assert np.mean([r['score'] for r in rows])==results[arm]['mean_score']
        scores = {arm:{r['seed']:r['score'] for r in rows} for arm,rows in games.items()}
        assert scores['control'].keys()==scores['hybrid'].keys()
        differences = np.array([scores['hybrid'][s]-scores['control'][s] for s in sorted(scores['control'])])
        rng = np.random.default_rng(5831)
        boot = differences[rng.integers(100,size=(20000,100))].mean(1)
        payload = dict(control_mean=a['mean_score'], hybrid_mean=b['mean_score'],
            mean_difference=float(differences.mean()),
            paired_game_bootstrap_95_interval=np.quantile(boot,[.025,.975]).tolist(),
            wins=int((differences>0).sum()),ties=int((differences==0).sum()),
            matched_updates_since_common_source=updates,
            common_source=first['control']['resume_dagger'],recipe_checks=list(shared),
            training_seconds_since_common_source={arm:r['training_seconds']+
                (first[arm]['training_seconds'] if suffix else 0) for arm,r in results.items()},
            training_transitions_since_common_source={arm:r['training_transitions']+
                (first[arm]['training_transitions'] if suffix else 0) for arm,r in results.items()},
            notes='One fine-tuning seed and reused 100 selection games. Paired bootstrap covers game variation only. The extension continues both existing arms; it is not an independent replication. Compare final endpoints, with diagnostics excluded. Later collection trajectories differ as a consequence of the learning rule.')
        output='dagger_cost_extension_comparison.json' if suffix else 'dagger_cost_comparison.json'
        write(base/output,payload)
        print(output, json.dumps(payload),flush=True)


if __name__=='__main__':
    compare()
