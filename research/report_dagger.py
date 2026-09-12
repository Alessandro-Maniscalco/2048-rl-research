"""Live comparison of fixed fitting data and teacher-labelled student boards."""
import json
from html import escape
from pathlib import Path
import numpy as np

from research.afterstate_teacher import write

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def replications(base):
    """Summarize independent fine-tuning pairs, not 300 independent runs.

    The actor pretraining and evaluation seeds are shared. Each completed
    pair contributes ONE difference to the across-training-seed summary.
    Continuations and diagnostic checkpoints never enter this comparison.
    """
    rows, differences, adaptive, fixed = [], [], [], []
    shared = ('initial_actor_sha256', 'data_sha256', 'teacher_sha256', 'algorithm',
        'width', 'depth', 'heads', 'input_encoding', 'batch', 'lr', 'max_grad_norm',
        'updates_per_round', 'seed')
    for seed in (0, 1, 2):
        results = {source:read(base/f'transformer_dagger_{source}_seed{seed}'/'result.json', {})
                   for source in ('student', 'fixed')}
        a, f = results['student'], results['fixed']
        row = dict(seed=seed, adaptive_mean=a.get('mean_score'), fixed_mean=f.get('mean_score'),
            paired_difference=None, complete=False)
        if all(x.get('complete') and x.get('completed_budget') for x in (a, f)):
            if any(a[key] != f[key] for key in shared) or any(x['updates'] != 8192 for x in (a, f)):
                raise ValueError('Replication arms have mismatched recipes or update budgets')
            if any(x.get('resume_dagger') or x.get('diagnostic_only') for x in (a, f)):
                raise ValueError('Continuation or diagnostic cannot enter initial-budget replication')
            games = {source:read(base/f'transformer_dagger_{source}_seed{seed}'/'evaluation.json')['episodes']
                     for source in ('student', 'fixed')}
            if any(len(x) != 100 for x in games.values()) or ({x['seed'] for x in games['student']} != {x['seed'] for x in games['fixed']}):
                raise ValueError('Replication endpoints require the same 100 game seeds')
            delta = a['mean_score']-f['mean_score']
            row.update(paired_difference=delta, complete=True,
                adaptive_training_seconds=a['training_seconds'], fixed_training_seconds=f['training_seconds'])
            differences.append(delta)
            adaptive.append(a['mean_score'])
            fixed.append(f['mean_score'])
        rows.append(row)
    result = dict(rows=rows, complete_pairs=len(differences), requested_pairs=3,
        complete=len(differences)==3,
        mean_paired_difference=float(np.mean(differences)) if differences else None,
        paired_difference_sample_sd=float(np.std(differences, ddof=1)) if len(differences)>1 else None,
        adaptive_mean=float(np.mean(adaptive)) if adaptive else None,
        fixed_mean=float(np.mean(fixed)) if fixed else None,
        note='Each pair is one independent fine-tuning seed with the same original pretrained actor. SD measures variation of paired mean-score differences across these seeds; it is not a confidence interval. Same100selectiongame seeds reused for comparability; reserved final test unused. Only complete matched8192-update pairs are aggregated.')
    write(base/'dagger_replications.json', result)
    return result


def render(base=Path('runs/research/scaled_transformer')):
    protocol = read(base/'dagger_protocol.json')
    if not protocol:
        return
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for job in protocol['jobs']:
        folder = base/job['id']
        c = job['config']
        diagnostic = c.get('diagnostic_only', False)
        label = 'Student boards + original data' if c['data_source']=='student' else 'Original fixed data only'
        if 'dagger_cost_' in job['id']:
            label = 'Teacher actions + mistake cost' if c.get('teacher_gap_weight', 0) else 'Matched teacher-action control'
        if 'dagger_stage_' in job['id']:
            label = 'Half stage-balanced sampling' if c.get('stage_sampling_fraction', 0) else 'Matched uniform sampling'
        label += f' · seed {job["seed"]}'
        if c.get('resume_dagger'):
            label += f' · continuation {job.get("continuation_stage",1)}'
        if diagnostic:
            label += ' (diagnostic)'
        curve = read(folder/'curve.json', [])
        result = read(folder/'result.json', {})
        meta = read(folder/'last/metadata.json', {}).get('experiment', {})
        chosen = read(folder/'best_selection.json', {}).get('summary', {})
        phase = ('complete' if result.get('completed_budget') else result.get('stop_reason','running')) if folder.exists() else 'queued'
        if curve and not diagnostic:
            axes[0].plot([r.get('total_dagger_updates',r['updates']) for r in curve], [r['mean_score'] for r in curve], 'o-', label=label)
            axes[1].plot([r.get('total_training_seconds',r['training_seconds'])/60 for r in curve], [r['mean_score'] for r in curve], 'o-', label=label)
            axes[2].plot([r.get('total_dagger_updates',r['updates']) for r in curve], [r['teacher_action_agreement']*100 for r in curve], 'o-', label=label)
        def fmt(value):
            return f'{value:,.0f}' if value is not None else '—'
        links = [f'<a href="{job["id"]}/config.json">config</a>'] if folder.exists() else []
        for file, title in [('replay_last.html','final replay'), ('replay_best.html','selected replay'),
                            ('dataset_blocks.json','teacher corrections'), ('progress.json','rounds')]:
            if (folder/file).exists():
                links.append(f'<a href="{job["id"]}/{file}">{title}</a>')
        rows.append(f'<tr><td>{escape(label)}<br><small>{escape(job["id"])}</small></td><td>{escape(phase)}</td>'
            f'<td>{fmt(meta.get("updates"))}</td><td>{fmt(meta.get("training_transitions"))}</td>'
            f'<td>{fmt(meta.get("fitting_boards"))}</td><td>{fmt(curve[-1]["mean_score"] if curve else None)}</td>'
            f'<td>{fmt(result.get("mean_score"))}</td><td>{fmt(chosen.get("mean_score"))}</td><td>{" · ".join(links)}</td></tr>')
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].set(xlabel='Imitation updates since original actor', ylabel='Mean raw score · 128 monitoring games')
    axes[1].set(xlabel='Training minutes since original actor', ylabel='Mean raw game score')
    axes[2].set(xlabel='Imitation updates since original actor', ylabel='Teacher agreement on 2,048 held-out boards (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5,.01), ncol=2, fontsize=8)
    fig.tight_layout(rect=(0,.20,1,1))
    fig.savefig(base/'dagger_curves.png', dpi=150)
    plt.close(fig)
    findings = ''
    comparison = read(base/'dagger_comparison.json')
    if comparison:
        treatment = comparison['runs']['transformer_dagger_student_seed0']['summary']['mean_score']
        control = comparison['runs']['transformer_dagger_fixed_seed0']['summary']['mean_score']
        delta = next(x for x in comparison['comparisons'] if x['control']=='transformer_dagger_fixed_seed0')
        low,high = delta['paired_game_bootstrap_95_interval']
        findings = f'<p><b>Completed matched seed-0 pair:</b> student-board corrections {treatment:,.0f} versus fixed data {control:,.0f}, with 8,192 updates each. Paired difference {delta["mean_difference"]:,.0f} points, bootstrap 95% interval [{low:,.0f}, {high:,.0f}]; {delta["wins"]}/100 game wins. This interval covers game-seed variation, not training-seed variation. The adaptive arm used 443 training seconds versus 312 for the control, plus shared historical pretraining. <a href="dagger_comparison.json">Full comparison</a> · <a href="reinforce_diagnostics/dagger_student_last/index.html">Inspect its remaining transition mistakes</a>.</p>'
    for job in reversed(protocol['jobs']):
        c = job['config']
        if not c.get('resume_dagger') or c.get('diagnostic_only'):
            continue
        result = read(base/job['id']/'result.json', {})
        if not result.get('complete'):
            continue
        findings += f'<p><b>Latest completed continuation:</b> {result["mean_score"]:,.0f} mean on 100 selection games after {result["total_dagger_updates"]:,} cumulative imitation updates. This includes {result["updates"]:,} new updates and {result["training_transitions"]:,} new training moves; it has more training than the initial-budget pairs. <a href="{job["id"]}/replay_last.html">Watch the final policy</a>.</p>'
        break
    repeated = replications(base)
    findings += '<h2>Independent fine-tuning seeds · 8,192 updates per arm</h2><table><tr><th>Seed</th><th>Student-board corrections</th><th>Fixed data</th><th>Paired difference</th></tr>'
    for row in repeated['rows']:
        cells = ''.join('<td>'+('—' if row[k] is None else f'{row[k]:,.0f}')+'</td>'
                        for k in ('adaptive_mean', 'fixed_mean', 'paired_difference'))
        findings += f'<tr><td>{row["seed"]}</td>{cells}</tr>'
    findings += '</table><p>'+f'{repeated["complete_pairs"]}/3 complete matched pairs. '
    if repeated['complete']:
        findings += f'Mean paired gain {repeated["mean_paired_difference"]:,.0f}; sample SD across the three paired gains {repeated["paired_difference_sample_sd"]:,.0f}. '
    findings += 'These vary fine-tuning randomness and share the original actor pretraining. Continuations and diagnostic runs are excluded. <a href="dagger_replications.json">Exact results and aggregation definition</a>.</p>'
    cost_comparison = read(base/'dagger_cost_comparison.json')
    if cost_comparison:
        low, high = cost_comparison['paired_game_bootstrap_95_interval']
        findings += (f'<p><b>Cost-aware teaching, first matched 8,192-update pair:</b> '
            f'hard-action control {cost_comparison["control_mean"]:,.0f}; '
            f'actions plus mistake cost {cost_comparison["hybrid_mean"]:,.0f}. '
            f'Paired gain {cost_comparison["mean_difference"]:,.0f}, game-bootstrap 95% interval '
            f'[{low:,.0f}, {high:,.0f}], {cost_comparison["wins"]}/100 wins. '
            'The interval includes zero; this is one fine-tuning seed and reused selection games. '
            'The prespecified extension is reported separately below. '
            '<a href="dagger_cost_comparison.json">Matched results and checks</a>.</p>')
    extended = read(base/'dagger_cost_extension_comparison.json')
    if extended:
        low, high = extended['paired_game_bootstrap_95_interval']
        findings += (f'<p><b>Completed equal-budget extension, 16,384 updates per arm since the common source:</b> '
            f'hard-action control {extended["control_mean"]:,.0f}; '
            f'actions plus mistake cost {extended["hybrid_mean"]:,.0f}. '
            f'Paired gain {extended["mean_difference"]:,.0f}, game-bootstrap 95% interval '
            f'[{low:,.0f}, {high:,.0f}], {extended["wins"]}/100 wins. '
            'These are the same two training trajectories continued once, not independent replications. '
            'Further work tests a different question: whether late-game examples are sampled often enough. '
            '<a href="dagger_cost_extension_comparison.json">Endpoint comparison</a>.</p>')
    from research.compare_dagger_sampling import compare as compare_sampling
    sampled = compare_sampling(base)
    if sampled:
        low, high = sampled['paired_game_bootstrap_95_interval']
        findings += (f'<p><b>Matched game-stage sampling test, 8,192 new updates each:</b> '
            f'uniform {sampled["uniform_mean"]:,.0f}; half stage-balanced {sampled["balanced_mean"]:,.0f}. '
            f'Paired difference {sampled["mean_paired_difference"]:,.0f}, game-bootstrap 95% interval '
            f'[{low:,.0f}, {high:,.0f}], {sampled["wins"]}/100 wins. '
            'Same source actor, Adam, RNG, fitting boards, teacher and cost-aware loss. '
            'One fine-tuning seed and reused selection games; diagnostics excluded. '
            '<a href="dagger_stage_comparison.json">Results and checks</a>.</p>')
    gap_path = 'reinforce_diagnostics/dagger_cont_last/collection_gap_distribution.json'
    gaps = read(base/gap_path)
    if gaps:
        costly = next(row for row in gaps['bins'] if row['lower_points'] == 10000)
        findings += (f'<p><b>How costly are the recorded disagreements?</b> In the last four collection '
            f'rounds before the 19,151 checkpoint, {gaps["disagreements"]:,} of '
            f'{gaps["decisions"]:,} recorded decisions disagreed with the teacher. '
            f'The {costly["fraction_of_disagreements"]:.1%} of disagreements with at least '
            f'10,000 teacher-predicted points of gap account for '
            f'{costly["fraction_of_total_estimated_gap"]:.1%} of the summed gap. '
            'These are approximate teacher values and successive collecting policies, not measured '
            'losses in eventual score or a held-out test of the final actor. A matched comparison now '
            'tests hard teacher-action learning against the same loss plus expected teacher mistake cost. '
            'Both start from the completed 32,038 actor with the same data and optimizer; '
            'the 512-update diagnostic is excluded from conclusions. '
            f'<a href="{gap_path}">Data and provenance</a>.</p>')
    page = '''<!doctype html><html><meta charset="utf-8"><meta http-equiv="refresh" content="60">
<title>Teacher corrections on student boards</title><style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 20px;color:#202830;background:#f7f9fc}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;border-bottom:1px solid #dde3eb;text-align:left}img{width:100%}small{color:#667}pre{white-space:pre-wrap;background:white;padding:20px}a{color:#245bc0}</style>
<a href="index.html">All research</a> · <a href="reinforce.html">REINFORCE results</a>
<h1>Can teacher corrections on the student's own boards help?</h1>
<p>Both arms begin at the original supervised actor (6,514 mean on 100 games), with the same 805,892-parameter Transformer: 16 learned tile embeddings, four attention/MLP blocks, four heads and eight shared board orientations. It outputs four move logits. The teacher is never an input and never chooses a played action.</p>
<p><b>Changed factor:</b> both fit hard teacher-best action labels on the original fitting data. The adaptive arm also plays fresh complete games with its greedy policy, labels every visited board using the frozen n-tuple teacher, and keeps those boards in the growing dataset. The control uses only the original fitting data. Optimizer updates, minibatch size and learning rate match; collection and teacher queries are additional cost.</p>
<p>This is <b>DAgger imitation learning</b>, not another REINFORCE variant. The scalar critic and former soft-target/value loss are absent in both arms. The matched fixed-data control separates that objective change from adding student-visited data. See <a href="https://proceedings.mlr.press/v15/ross11a.html">Ross, Gordon &amp; Bagnell (2011)</a>.</p>
<pre>Teacher Q(s,a) = [immediate merge points + V_table(afterstate(s,a))] / 128
A*(s) = all legal actions tied for maximum teacher Q
Loss = - mean_s log[ sum_(a in A*(s)) πθ(a | s) ]
D_next = D_previous ∪ teacher-labelled boards visited by the student
Play: a = argmax_legal πθ(a | s)</pre>
<p>The table value includes all 8 patterns × 8 orientations = 64 lookups. No future tree search is used for labels or play. Ties are retained to avoid teaching inconsistent arbitrary directions to a rotation-equivariant network. Historical teacher and actor pretraining are extra cost. Training seeds, held-out label games, 128 monitoring games and 100 selection games remain separate; the reserved final test is unused.</p>
'''+findings+'''<div style="overflow:auto"><table><tr><th>Arm</th><th>Status</th><th>New updates</th><th>New game moves</th><th>Fitting boards</th><th>Latest 128 mean</th><th>Final 100 mean</th><th>Selected 100 mean</th><th>Evidence</th></tr>'''+''.join(rows)+'''</table></div>
<img src="dagger_curves.png" alt="Policy score versus optimizer updates and training time, and held-out teacher agreement">
<p>Final and monitor-selected checkpoints are distinct. Diagnostic runs are excluded from plots and competitive rankings. Continuation curves include earlier imitation updates and training time; the table reports new work in each run. Monitoring time and original actor pretraining are excluded from the training-time axis. One training seed is exploratory; lower imitation loss alone does not establish better gameplay.</p>
<h2>Predeclared questions and budgets</h2><pre>'''+escape('\n\n'.join(j['id']+'\n'+j['question'] for j in protocol['jobs']))+'</pre></html>'
    (base/'dagger.html').write_text(page)


if __name__ == '__main__':
    render()
