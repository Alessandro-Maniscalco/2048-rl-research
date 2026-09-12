"""Separate the value of filtering actions from the value of training."""
import json
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from research.afterstate_teacher import write


def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}


def render():
    base=Path('runs/research/scaled_transformer')
    protocol=read(base/'cnn_filtered_reinforce_protocol.json')
    if not protocol:return
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    rows=[]
    findings=[]
    jobs=[dict(id=protocol['reference_job'],config=dict(spawn_safety=False,diagnostic_only=False))]+[protocol['smoke']]+protocol['full_jobs']
    for job in jobs:
        folder=base/job['id']
        curves=read(folder/'curve.json') or []
        result=read(folder/'result.json')
        initial=read(folder/'initial_evaluation.json')
        diagnostic=job['config'].get('diagnostic_only',False)
        label='Unfiltered baseline learning' if not job['config']['spawn_safety'] else 'Filtered diagnostic · 16 games/update' if diagnostic else 'Filtered learning · 64 games/update'
        for ax,key in zip(axes,['mean_score','sampled_mean_score']):
            if curves:
                ax.plot([c['updates'] for c in curves],[c.get(key) for c in curves],marker='o',linestyle=':' if diagnostic else '-',label=label)
        value=lambda x:'—' if x is None else f'{x:,.0f}'
        links=' · '.join(f'<a href="{job["id"]}/{file}">{name}</a>' for file,name in
            [('replay_last.html','Final replay'),('config.json','Config'),('first_training_game.json','Training trace'),('progress.json','Updates')]
            if (folder/file).exists())
        rows.append(f'<tr><td>{label}</td><td>{value(curves[-1]["updates"] if curves else None)}</td>'
            f'<td>{value(initial.get("summary",{}).get("mean_score"))}</td>'
            f'<td>{value(result.get("mean_score"))}</td><td>{value(result.get("sampled_mean_score"))}</td><td>{links}</td></tr>')
        if result.get('complete') and result.get('completed_budget') and initial:
            differences={}
            for mode,initial_file,final_file in [('greedy','initial_evaluation.json','evaluation.json'),
                                               ('sampled','initial_sampled_evaluation.json','sampled_evaluation.json')]:
                before=read(folder/initial_file);after=read(folder/final_file)
                if not before or not after:continue
                assert before['summary']['complete'] and after['summary']['complete']
                old={e['seed']:e for e in before['episodes']};new={e['seed']:e for e in after['episodes']}
                assert old.keys()==new.keys() and len(old)==100
                assert all(e['complete'] and not e['truncated'] for arm in [old,new] for e in arm.values())
                delta=np.array([new[s]['score']-old[s]['score'] for s in sorted(old)])
                boot=delta[np.random.default_rng(8958770).integers(100,size=(20000,100))].mean(1)
                differences[mode]=dict(initial_mean=before['summary']['mean_score'],final_mean=after['summary']['mean_score'],
                    gain=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),wins=int((delta>0).sum()))
            write(folder/'learning_comparison.json',dict(comparisons=differences,diagnostic_only=diagnostic,
                note='Matched100selection seeds with the same risk filter before/after training. One adaptation seed and reused selection games.'))
            if not diagnostic and 'greedy' in differences:
                g=differences['greedy']
                findings.append(f'<p><b>Actual training change with the filter held fixed: {g["initial_mean"]:,.0f} → {g["final_mean"]:,.0f}.</b> '
                    f'Gain {g["gain"]:,.0f}, paired95% interval [{g["ci95"][0]:,.0f}, {g["ci95"][1]:,.0f}]. '
                    'This interval describes game-seed variation, not training-seed uncertainty.</p>')
    for ax,title in zip(axes,['Greedy policy','Sampled policy']):
        ax.set_title(title);ax.set_xlabel('Complete-game batch updates');ax.set_ylabel('Mean raw score · same 128 monitor seeds')
        ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(base/'cnn_filtered_reinforce_curves.png',dpi=150);plt.close(fig)
    page=f'''<!doctype html><html><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>Learning with the safety policy</title>
    <style>body{{font:17px system-ui;max-width:1150px;margin:40px auto;padding:0 20px;line-height:1.5;color:#19212d}}img{{width:100%}}
    table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}a{{color:#245ab5}}</style>
    <a href="index.html">All experiments</a><h1>Learning with the safety policy</h1><p>{escape(protocol['question'])}</p>
    <p>The score at update zero is the frozen starting policy. A higher starting score caused by the risk filter is not a training gain.
    Compare changes from that starting point and final matched game outcomes. Diagnostic runs are excluded from rankings.</p>
    <img src="cnn_filtered_reinforce_curves.png" alt="Filtered and unfiltered policies across complete-game training updates">
    <table><tr><th>Experiment</th><th>Updates</th><th>Initial 100 greedy mean</th><th>Final 100 greedy mean</th><th>Final 100 sampled mean</th><th>Inspect</th></tr>{''.join(rows)}</table>{''.join(findings)}
    <h2>What changes</h2><p>For each legal move, calculate R(s,a), the probability that its next random spawn ends the game.
    Allow A(s)={{a legal : R(s,a)=min R(s,·)}}. During training, π(a|s)=softmax(logits over A(s)).
    The original game still treats all legal moves as legal; this policy chooses from a smaller set.</p>
    <p>Return G is the actual remaining merge score divided by 128. Subtract the mean return of other completed games at the same move index,
    then minimize −Σ log π(a|s)(G−b)/B. Exactly one Adam step follows the entire batch. The probabilities used for training match those used to collect the games.</p>
    <p>The filter is stored with the checkpoint and restored for playback. There are no teacher targets or learned critic.
    The same original pretrained CNN and raw reward are used. See <a href="cnn_reinforce.html">the unfiltered comparison and equations</a>,
    <a href="cnn_spawn_safety/index.html">the independent frozen-policy evidence</a>, and <a href="cnn_filtered_reinforce_protocol.json">the prespecified protocol</a>.</p></html>'''
    (base/'cnn_filtered_reinforce.html').write_text(page)


if __name__=='__main__':render()
