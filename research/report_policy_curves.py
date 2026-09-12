"""Plot policy quality separately from scores of unfinished training games."""
import argparse
import csv
from html import escape
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'mlp_q_exponents_n1_score':'MLP · one-step',
    'transformer_q_exponents_n1_score':'Transformer · one-step',
    'transformer_q_exponents_n3_score':'Transformer · three-step'}
COLORS=['#197970','#c97931','#7065aa']


def render(out):
    study=json.loads((out/'results.json').read_text())
    if not study['runs']:return
    groups={n:[r for r in study['runs'] if r['name']==n] for n in dict.fromkeys(r['name'] for r in study['runs'])}
    baseline=json.loads((out/'random_baseline.json').read_text())['summary']['mean_score']
    loaded={}
    for name,runs in groups.items():
        curves=[];progress=[]
        for r in runs:
            path=out/name/f"seed{r['seed']}"
            curves.append(json.loads((path/'curve.json').read_text()))
            with (path/'progress.csv').open() as f:progress.append(list(csv.DictReader(f)))
        loaded[name]=(curves,progress)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    summaries=[]
    for color,(name,runs) in zip(COLORS,groups.items()):
        curves,progress=loaded[name]
        scores=np.array([[p['mean_score'] for p in c] for c in curves])
        steps=np.array([p['transitions'] for p in curves[0]])
        times=np.array([[p['training_seconds'] for p in c] for c in curves]).mean(0)
        m=scores.mean(0);sd=scores.std(0,ddof=1) if len(scores)>1 else np.zeros_like(m)
        for ax,x in zip(axes,(steps,times)):
            ax.plot(x,m,'o-',color=color,label=LABELS[name])
            ax.fill_between(x,m-sd,m+sd,color=color,alpha=.15)
        summaries.append(dict(name=name,label=LABELS[name],training_seeds=len(runs),
            initial_monitor_score=float(m[0]),final_monitor_score=float(m[-1]),
            best_monitor_score=float(m.max()),best_monitor_transitions=int(steps[m.argmax()]),
            checkpoint_decreases=int((np.diff(m)<0).sum()),
            final_unseen_100_score=float(np.mean([r['mean_score'] for r in runs])),
            training_seconds=float(np.mean([r['training_seconds'] for r in runs])),
            monitoring_seconds=float(np.mean([r['monitoring_seconds'] for r in runs])),
            cpu_game_step_seconds=float(np.mean([r['cpu_game_step_seconds'] for r in runs])),
            update_seconds=float(np.mean([r['update_seconds'] for r in runs]))))
    for ax in axes:
        ax.axhline(baseline,color='#777',ls='--',label=f'Random legal moves: {baseline:.0f}')
        ax.set(ylabel='Mean final raw score on 128 fixed games',ylim=(0,None))
        ax.grid(alpha=.2);ax.legend(fontsize=9)
    axes[0].set(xlabel='Fresh training transitions',title='Does the frozen policy improve with experience?')
    axes[1].set(xlabel='Training seconds (monitoring excluded)',title='How much improvement per second?')
    fig.savefig(out/'policy_quality.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    fields=['mean_live_score_128','mean_completed_score_128','q_loss','epsilon']
    titles=['Mean score of 128 active games (resets included)',
        'Mean final score of last 128 completed training games',
        'TD training loss (not a policy score)','Exploration probability']
    for color,(name,runs) in zip(COLORS,groups.items()):
        curves,progress=loaded[name]
        x=[int(p['updates']) for p in progress[0]]
        for ax,field in zip(axes.flat,fields):
            values=np.array([[float(p[field]) if p[field] else np.nan for p in rows] for rows in progress])
            count=np.isfinite(values).sum(0)
            m=np.divide(np.nansum(values,axis=0),count,out=np.full(len(x),np.nan),where=count>0)
            ax.plot(x,m,color=color,lw=1,label=LABELS[name])
    for ax,title in zip(axes.flat,titles):
        ax.set(title=title,xlabel='Gradient updates');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out/'training_diagnostics.png',dpi=160);plt.close(fig)
    (out/'summary.json').write_text(json.dumps(summaries,indent=2))
    rows=[]
    for s in summaries:
        vals=[s['label'],s['training_seeds'],f"{s['initial_monitor_score']:,.0f}",
            f"{s['final_monitor_score']:,.0f}",f"{s['best_monitor_score']:,.0f} at {s['best_monitor_transitions']:,}",
            s['checkpoint_decreases'],f"{s['final_unseen_100_score']:,.0f}"]
        rows.append('<tr>'+''.join(f'<td>{escape(str(v))}</td>' for v in vals)+'</tr>')
    timing=[]
    for s in summaries:
        timing.append(f"<li>{s['label']}: {s['training_seconds']:.1f}s training, "
            f"{s['monitoring_seconds']:.1f}s monitoring, {s['cpu_game_step_seconds']:.3f}s inside CPU game stepping, "
            f"{s['update_seconds']:.1f}s for replay sampling/transfers/gradient updates. Means per training seed; "
            'game stepping excludes other CPU work such as replay assembly.</li>')
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>2048 · How policies improve</title><style>body{font:17px/1.6 system-ui;color:#544c44;background:#faf8ef;max-width:1250px;margin:40px auto;padding:0 24px}img{width:100%;height:auto}a{color:#197970}table{width:100%;border-collapse:collapse;font-size:15px}td,th{text-align:left;padding:12px;border-bottom:1px solid #d5cfc0}.scroll{overflow:auto}.callout{background:#e9eee6;padding:20px;border-radius:12px}</style>
<h1>How policies improve during training</h1>
<p class="callout"><b>The main measurement is the final score of complete games played by a frozen policy.</b> At every checkpoint, the current network plays the same 128 game seeds without exploration, learning, or search. Each curve averages three independent training seeds; shading is their standard deviation, not a confidence interval. Scores can go down as well as up.</p>
<p>All models use simple tile-exponent inputs and proportional merge rewards. The two-block, four-head Transformer is compared with one-step and three-step TD; a one-step MLP is the reference. All begin with random weights. Three-step returns use exploratory trajectories without off-policy correction.</p>
<p>For checkpoint k: mean score = (final score on seed1 + ... + final score on seed128) / 128. Repeating this for successive checkpoints gives the learning curve. This is different from averaging the scores of 128 unfinished games.</p>
<img src="policy_quality.png" alt="Frozen policy mean final scores versus transitions and training time">
<p>Each run collects '''+f"{study['steps']:,}"+''' transitions. Evaluation occurs at initialization and every '''+f"{study['interval']:,}"+''' transitions. The 128 monitoring seeds are disjoint from training seeds and the separate 100 final-evaluation seeds. Once monitoring guides model selection, it is validation data. Parameter counts differ between architectures; learning budgets are matched.</p>
<div class="scroll"><table><tr><th>Policy</th><th>Training seeds</th><th>Initial score</th><th>Final monitor score</th><th>Best monitor score</th><th>Downward intervals</th><th>Separate final 100-game score</th></tr>'''+''.join(rows)+'''</table></div>
<h2>What happens after every weight update?</h2>
<p>The curves below log every update and average across training seeds. The active-board mean is useful for observing collection, but games have different ages and scores reset to zero when they end. The last-128-completed-games mean also mixes policies from different moments and includes exploratory moves. Neither is the same as the frozen-policy evaluation above. TD loss is a learning diagnostic, not game performance.</p>
<img src="training_diagnostics.png" alt="Every-update active board score, completed game score, TD loss and exploration">
<h2>CPU and GPU execution</h2>
<p>The CPU executes game rules and replay assembly. Transformers run on the GPU through MPS; the small MLP runs on CPU because the preceding benchmark favored it. The current loop is sequential: inference → CPU game step → replay assembly → learning update. It does not yet overlap CPU games with GPU learning.</p>
<p>Overlap is possible by simulating already-selected moves on a CPU worker while the GPU learns from existing replay. Action selection still needs a network result, and sampling from older replay adds a scheduling choice. The measurements below show game simulation is only one part of CPU work; they do not establish the speedup an asynchronous implementation would achieve.</p><ul>'''+''.join(timing)+'''</ul>
<p><a href="results.json">All run results</a> · <a href="summary.json">Curve summary</a> · <a href="journal.md">Experiment journal</a></p></html>'''
    (out/'index.html').write_text(html)
    notes=['# Frozen policy learning curves','',f"Status: {study['status']}",
        f"{study['steps']:,} transitions per run; {study['monitor_games']} complete monitoring games per checkpoint; {len(study['training_seeds'])} training seeds.",
        'No 80-feature input, positional shaping, teacher copying, or inference planning.','']
    for s in summaries:
        notes += [f"## {s['label']}",
            'Testing: fixed policy evaluations throughout learning, with active-game scores logged separately at every update.',
            f"Result: initial monitor score {s['initial_monitor_score']:.2f}, final {s['final_monitor_score']:.2f}, best {s['best_monitor_score']:.2f} at {s['best_monitor_transitions']} transitions.",
            f"Learning: the mean curve decreased across {s['checkpoint_decreases']} checkpoint intervals; individual updates are not guaranteed to improve policy performance.",
            f"Separate final evaluation: {s['final_unseen_100_score']:.2f} mean raw score.",'']
    (out/'journal.md').write_text('\n'.join(notes))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('runs/research/policy_curves'))
    render(p.parse_args().out)
