"""Matched QR-DQN TD-horizon study using the existing single GPU queue."""
import argparse
from copy import deepcopy
from html import escape
import json
from pathlib import Path

import numpy as np


def setup(base):
    from research.scaled_transformer import write
    manifest = json.loads((base / 'manifest.json').read_text())
    jobs = {j['id']: j for j in manifest['jobs']}
    template = jobs['qr_compare_qr_dqn_score_seed0']
    study = []
    for seed in (0, 1, 2):
        for n in (3, 1, 5):
            job_id = (f'qr_compare_qr_dqn_score_seed{seed}' if n == 3
                      else f'qr_td_n{n}_score_seed{seed}')
            expected = deepcopy(template['config']) | {'n': n}
            if job_id in jobs:
                assert jobs[job_id]['config'] == expected
                assert jobs[job_id]['steps'] == template['steps']
            else:
                jobs[job_id] = dict(id=job_id, seed=seed, steps=template['steps'],
                    config=expected, status='pending', question=(
                        f'Test {n}-step TD against 1/3/5 with QR-DQN, training seed {seed}. '
                        'Only TD horizon changes within a seed. All start from scratch; '
                        '4,194,304 transitions, batch512, ordinary points/128 reward, '
                        'same two-block width128 Transformer with51 quantiles/action. '
                        'No planning or teacher. Report final100-game scores and128-game '
                        'learning curves; repeat across three training seeds.'))
            study.append(dict(id=job_id, seed=seed, n=n))
    # Preserve completed work and the active worker; prioritize the requested study next.
    done = {r['id'] for r in json.loads((base / 'results.json').read_text())}
    active = json.loads((base / 'status.json').read_text()).get('job')
    preserved = [j for j in manifest['jobs'] if j['id'] in done or j['id'] == active]
    used = {j['id'] for j in preserved}
    priority = [jobs[r['id']] for r in study if r['id'] not in used]
    used.update(j['id'] for j in priority)
    manifest['jobs'] = preserved + priority + [j for j in manifest['jobs'] if j['id'] not in used]
    write(base / 'manifest.json', manifest)
    write(base / 'td_horizon_study.json', dict(runs=study, transitions=template['steps'],
        selection='Final checkpoint at matched experience; best128 monitoring is secondary.',
        note='Reuse matching n3 runs; no duplicate training. Evaluation suites are selection data.'))
    print('Scheduled 1/3/5 TD study across seeds0/1/2; reused existing n3 runs.')


def render(base):
    import matplotlib.pyplot as plt
    spec_file = base / 'td_horizon_study.json'
    if not spec_file.exists():
        return
    spec = json.loads(spec_file.read_text())
    status = json.loads((base / 'status.json').read_text())
    results = {r['id']: r for r in json.loads((base / 'results.json').read_text())}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    rows, finals = [], {n: [] for n in (1, 3, 5)}
    for run in spec['runs']:
        path = base / run['id']
        curve = json.loads((path / 'curve.json').read_text()) if (path / 'curve.json').exists() else []
        result = results.get(run['id'], {})
        phase = result.get('status', 'running' if status.get('job') == run['id'] else 'pending')
        score = result.get('result', {}).get('mean_score') if phase == 'complete' else None
        if score is not None:
            finals[run['n']].append(score)
        if curve:
            axes[run['seed']].plot([r['transitions'] for r in curve],
                [r['mean_score'] for r in curve], label=f"{run['n']}-step TD")
        cells = [f"{run['n']}-step TD", run['seed'], phase,
            f"{curve[-1]['transitions']:,}" if curve else '—',
            f"{score:,.2f}" if score is not None else '—',
            f"{max(r['mean_score'] for r in curve):,.0f}" if curve else '—']
        rows.append('<tr>' + ''.join(f'<td>{escape(str(c))}</td>' for c in cells) + '</tr>')
    for seed, ax in enumerate(axes):
        ax.set(title=f'Training seed {seed}', xlabel='New training transitions', ylabel='128-game mean raw score')
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend()
    fig.savefig(base / 'td_horizon_curves.png', dpi=140)
    plt.close(fig)
    aggregate = ' · '.join(f'{n}-step: ' +
        (f'{np.mean(v):,.0f} ± {np.std(v, ddof=1):,.0f} points (training-seed SD)'
         if len(v) == 3 else f'{len(v)}/3 runs complete; aggregate pending') for n, v in finals.items())
    conclusion = ''
    if all(len(v) == 3 for v in finals.values()):
        conclusion = ('<p><b>All nine runs complete.</b> Three-step TD has the highest observed mean; '
            'five-step TD has the smallest spread in these three training seeds. Rankings change by seed, '
            'so this is not a decisive statistical result. Keep three-step as the main baseline and '
            'five-step as an alternative for longer training. '
            '<a href="td_horizon_comparison.json">Per-seed comparison and interpretation</a></p>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="60">
<title>2048 · One, three, or five-step TD?</title><style>body{font:16px/1.6 system-ui;max-width:1400px;margin:35px auto;padding:0 24px;background:#faf8ef;color:#544c44}a{color:#197970}img{width:100%}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:10px;border-bottom:1px solid #d9d2c5}.scroll{overflow:auto}</style>
<h1>Does one-, three-, or five-step TD learn better?</h1><a href="index.html">All Transformer experiments</a>
<p>Only the number of experienced rewards in the TD target changes within each training seed. Every run starts from scratch and gets 4,194,304 transitions. Matching three-step runs are reused. There is no search during play.</p>
<p>Target quantile: y = r₀ + γr₁ + … + γⁿ⁻¹rₙ₋₁ + γⁿz_target(sₙ, a*). The online network chooses a* by highest legal mean Q. Terminal transitions have no continuation; episodes ending early use the actual shorter horizon.</p>
<p>Same two-block Transformer, width128, four attention heads, 51 quantiles/action, exponent inputs, ordinary points/128 reward, batch512, learning rate0.0001, γ0.99 and replay settings. Collected trajectories can diverge as the policies learn; the replay update remains off-policy and uses no multi-step importance correction.</p>
<p>Primary comparison: final-checkpoint raw score over the same100 selection games, averaged across three training seeds. Curves and best monitoring scores use a different fixed128 games. Best monitoring scores are checkpoint-selected and should not be compared to final100 scores. Reserved final-test games stay unused.</p>
<p>''' + escape(aggregate) + '''</p><div class="scroll"><table><tr><th>TD horizon</th><th>Training seed</th><th>Status</th><th>New transitions</th><th>Final evaluation · 100 games</th><th>Best monitoring · 128 games</th></tr>''' + ''.join(rows) + '''</table></div><img src="td_horizon_curves.png" alt="One, three, and five-step TD learning curves, separated by training seed"></html>'''
    (base / 'td_horizon.html').write_text(page.replace('</html>', conclusion + '</html>'))


if __name__ == '__main__':
    from research.scaled_transformer import BASE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['setup', 'render'])
    args = parser.parse_args()
    if args.mode == 'setup':
        setup(BASE)
    render(BASE)
