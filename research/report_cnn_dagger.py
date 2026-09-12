"""Complete-game evidence for adaptation of the frozen pretrained CNN."""
import json
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from research.afterstate_teacher import write


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def render():
    base = Path('runs/research/scaled_transformer')
    protocol = read(base/'cnn_dagger_protocol.json')
    if not protocol:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    rows, endpoints = [], {}
    for job in ([protocol['smoke']] + protocol['jobs'] + protocol.get('stability_jobs', [])
                + protocol.get('anchored_jobs', []) + protocol.get('cost_only_jobs', [])):
        folder = base/job['id']
        diagnostic = job['config']['diagnostic_only']
        curve = read(folder/'curve.json', [])
        result = read(folder/'result.json', {})
        label = ('Diagnostic only' if diagnostic else
                 'Original fixed data' if job['config']['data_source'] == 'fixed' else
                 'Original + student boards')
        if 'small_step' in job['id']:
            label = 'Diagnostic: 10× smaller learning rate'
        elif 'kl_anchor' in job['id']:
            label = 'Diagnostic: smaller rate + frozen-policy KL'
        if 'anchored_' in job['id']:
            label = 'KL preserved CNN · '+('student boards' if job['config']['data_source']=='student' else 'fixed data')
        if 'cost_only' in job['id']:
            label = 'Diagnostic: estimated mistake cost, no forced action matching'
        if curve and not diagnostic:
            axes[0].plot([r['updates'] for r in curve], [r['mean_score'] for r in curve], 'o-', label=label)
            axes[1].plot([r['training_seconds']/60 for r in curve], [r['mean_score'] for r in curve], 'o-', label=label)
        phase = ('Complete' if result.get('completed_budget') else result.get('stop_reason') or
                 ('Running' if folder.exists() else 'Not started'))
        if job.get('status') == 'not_run_diagnostic_regression':
            phase = 'Not run: initial diagnostic collapsed'
        value = lambda x: '—' if x is None else f'{x:,.0f}'
        score = value(result.get('mean_score'))
        links = []
        for name,title in [('config.json', 'Config'), ('replay_last.html', 'Final replay'),
                           ('dataset_blocks.json', 'Teacher corrections'), ('progress.json', 'Training')]:
            if (folder/name).exists():
                links.append(f'<a href="{job["id"]}/{name}">{title}</a>')
        rows.append(f'<tr><td>{label}</td><td>{escape(phase)}</td>'
            f'<td>{value(curve[-1]["updates"] if curve else None)}</td>'
            f'<td>{value(curve[-1]["mean_score"] if curve else None)}</td>'
            f'<td>{score}</td><td>{" · ".join(links)}</td></tr>')
        if not diagnostic and result.get('complete') and result.get('completed_budget'):
            endpoints[job['config']['data_source']] = (job, result, read(folder/'evaluation.json'))
    baseline = protocol['baseline_monitor']['mean_score']
    for ax in axes:
        ax.axhline(baseline, color='gray', linestyle='--', label='Frozen CNN baseline')
        ax.set_ylabel('Mean raw score · same 128 monitor games')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_xlabel('New optimizer updates')
    axes[1].set_xlabel('Training minutes · evaluation excluded')
    fig.tight_layout()
    fig.savefig(base/'cnn_dagger_curves.png', dpi=150)
    plt.close(fig)
    finding = 'Full matched comparison pending; the diagnostic is excluded from model rankings.'
    if protocol.get('original_recipe_rejection'):
        rejection = protocol['original_recipe_rejection']
        finding = (f'The first diagnostic collapsed from {rejection["initial_monitor"]:,.0f} to '
            f'{rejection["final_monitor"]:,.0f} on 128 monitor games, despite lower teacher loss. '
            'The original full pair was cancelled before launch. Smaller-step and reference-KL '
            'diagnostics test whether useful behavior can be preserved during adaptation.')
    stability = read(base/'cnn_stability_comparison.json')
    if stability:
        a = stability['comparisons']['original']
        low, high = a['ci95']
        finding += (f' The KL diagnostic scored {a["anchored_mean"]:,.0f} versus original {a["control_mean"]:,.0f}; '
            f'paired gain95% interval [{low:,.0f}, {high:,.0f}], so improvement over the original remains uncertain. '
            'The longer anchored fixed/student-data pair starts from original weights.')
    if set(endpoints) == {'fixed', 'student'}:
        f, a = endpoints['fixed'], endpoints['student']
        for key in ('initial_actor_sha256', 'data_sha256', 'teacher_sha256', 'updates', 'batch', 'lr',
                    'teacher_gap_weight', 'teacher_gap_scale_points', 'reference_kl_weight', 'architecture', 'seed'):
            assert a[1][key] == f[1][key]
        assert a[1]['updates'] == 8192
        games = {mode:{e['seed']:e for e in entry[2]['episodes']} for mode,entry in endpoints.items()}
        assert games['fixed'].keys() == games['student'].keys() and len(games['fixed']) == 100
        delta = np.array([games['student'][s]['score']-games['fixed'][s]['score'] for s in sorted(games['fixed'])])
        rng = np.random.default_rng(8958100)
        interval = np.quantile(delta[rng.integers(100, size=(20000, 100))].mean(1), [.025, .975])
        result = dict(complete=True, fixed_mean=f[1]['mean_score'], student_mean=a[1]['mean_score'],
            mean_paired_difference=float(delta.mean()), wins=int((delta > 0).sum()),
            paired_game_bootstrap_95_interval=interval.tolist(),
            note='One adaptation seed with shared external pretraining; reused selection games, not independent final test.')
        write(base/'cnn_dagger_comparison.json', result)
        finding = (f'Final 100-game means: fixed {result["fixed_mean"]:,.0f}; student boards '
            f'{result["student_mean"]:,.0f}. Paired gain {delta.mean():,.0f}, game-bootstrap 95% interval '
            f'[{interval[0]:,.0f}, {interval[1]:,.0f}]. One adaptation seed and reused selection games. '
            'This does not measure training-seed uncertainty.')
        finding += (f' Both finish below the original CNN at {protocol["baseline_selection"]["mean_score"]:,.0f}. '
            'Fresh boards reduce imitation damage but do not establish improvement over initialization. '
            'Both128-game monitor curves were highest before training; this recipe is not extended.')
    cost_only = read(base/'cnn_cost_only_comparison.json')
    if cost_only:
        control = cost_only['comparisons']['original']
        finding += (f' Removing hard-action cross-entropy scored {control["candidate_mean"]:,.0f}; '
            f'paired change versus original {control["difference"]:,.0f},95% interval '
            f'[{control["ci95"][0]:,.0f}, {control["ci95"][1]:,.0f}]. '
            'The91k monitor value is a separate reused selection set, not evidence of a91k final mean.')
    body = f'''<!doctype html><html><meta charset="utf-8"><title>Pretrained CNN: learning from its own boards</title>
    <style>body{{font:17px system-ui;max-width:1150px;margin:40px auto;padding:0 20px;line-height:1.5;color:#19212d}}
    table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #ddd}}
    img{{width:100%}}a{{color:#245ab5}}small{{color:#555}}</style>
    <a href="index.html">All experiments</a><h1>Pretrained CNN: learning from its own boards</h1>
    <p>Frozen starting model: <b>{protocol['baseline_selection']['mean_score']:,.0f}</b> mean on the same 100 selection games,
    and <b>{baseline:,.0f}</b> on 128 monitor games. Different pretraining from the Transformer; this is a within-CNN adaptation test.</p>
    <p>{escape(protocol.get('cost_only_question', protocol.get('anchored_question', protocol.get('stability_question', protocol['question']))))}</p><p><b>{finding}</b></p>
    <img src="cnn_dagger_curves.png" alt="Mean complete-game score against updates and training time">
    <table><tr><th>Experiment</th><th>Status</th><th>Updates</th><th>Latest 128 monitor mean</th><th>Final 100 selection mean</th><th>Inspect</th></tr>{''.join(rows)}</table>
    <h2>What is learned</h2><p>16 cells → tile one-hot categories → learned whole-board, row and column convolutions → learned dense layers → four move logits.
    Mask illegal moves; choose the largest legal logit. The old critic is frozen and skipped. No search runs during student play.</p>
    <p>For a training board, slide each legal action and add its merge points to the teacher's complete afterstate table value.
    Train the probability of teacher-best actions upward, with an additional expected teacher-cost penalty.
    This is interactive imitation, not Q-learning or a calibrated neural Q output.</p>
    <p>Loss = w<sub>action</sub>[−ln Σ<sub>a in teacher-best set</sub> π(a|s)] + Σ<sub>a legal</sub> π(a|s)[max Q<sub>T</sub>(s,·) − Q<sub>T</sub>(s,a)] / 16384,
    with Q<sub>T</sub> expressed in raw points. The teacher is an approximation, not ground truth future score.</p>
    <p>w<sub>action</sub>=1 in ordinary teaching and0 in the cost-only diagnostic. The latter retains teacher mistake cost and the reference-KL term.</p>
    <p>The stability diagnostic additionally tests β KL(π<sub>original</sub> || π<sub>student</sub>), with β = 0 or 100.
    This penalizes changes to the starting CNN on sampled boards. It is not a hard guarantee about all states or game score.</p>
    <p>Limitation: the inherited CNN has 16 tile categories and clips ranks above15 to15. It has not learned a distinct 65,536 embedding.
    Original weights, failed attempts, diagnostic runs and complete-game histories are preserved.</p>
    <p><a href="cnn_dagger_protocol.json">Protocol</a> · <a href="dagger_stage_diagnostic/cnn_selection.json">Frozen baseline games</a></p></html>'''
    (base/'cnn_dagger.html').write_text(body)


if __name__ == '__main__':
    render()
