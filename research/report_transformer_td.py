"""Render saved measurements; never infer success from a training loss."""
import csv
from html import escape
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(out=Path('runs/research/transformer_td')):
    data=json.loads((out/'results.json').read_text());runs=data['runs']
    groups={name:[r for r in runs if r['name']==name] for name in dict.fromkeys(r['name'] for r in runs)}
    labels=list(groups)
    if not labels:return
    means=[np.mean([r['mean_score'] for r in groups[n]]) for n in labels]
    deviations=[np.std([r['mean_score'] for r in groups[n]],ddof=1) if len(groups[n])>1 else 0 for n in labels]
    colors=['#197970' if n.startswith('mlp') else '#c67b38' for n in labels]
    fig,ax=plt.subplots(figsize=(12,6),layout='constrained')
    ax.barh(labels,means,xerr=deviations,color=colors,capsize=4)
    baseline=json.loads((out/'random_baseline.json').read_text())['summary'] if (out/'random_baseline.json').exists() else None
    if baseline:
        ax.axvline(baseline['mean_score'],color='#555',ls='--',label=f"Random legal moves: {baseline['mean_score']:.0f}")
        ax.legend()
    ax.invert_yaxis();ax.set(xlabel='Mean raw score on 100 held-out games',title='131,072 fresh transitions per training seed')
    ax.grid(axis='x',alpha=.2)
    fig.savefig(out/'scores.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,(arch,encoding) in zip(axes.flat,[(a,e) for a in ('mlp_q','transformer_q') for e in ('exponents','embedding')]):
        for name,group in groups.items():
            if group[0]['architecture']!=arch or group[0]['input_encoding']!=encoding:continue
            curves=[json.loads((out/name/f"seed{r['seed']}"/'curve.json').read_text()) for r in group]
            y=np.array([[p['mean_score'] for p in c] for c in curves])
            x=[p['transitions'] for p in curves[0]];m=y.mean(0);s=y.std(0,ddof=1) if len(y)>1 else np.zeros_like(m)
            label=f"n={group[0]['n']}, {group[0]['reward_mode']}"
            ax.plot(x,m,'o-',label=label);ax.fill_between(x,m-s,m+s,alpha=.15)
        ax.set(title=f'{arch} · {encoding}',xlabel='Training transitions',ylabel='Mean raw score (20 monitoring games)')
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(out/'curves.png',dpi=150);plt.close(fig)
    rows=[]
    for name,g in groups.items():
        row=g[0]
        rows.append('<tr>'+''.join(f'<td>{escape(str(x))}</td>' for x in [name,len(g),
            f"{np.mean([r['mean_score'] for r in g]):,.1f} ± {np.std([r['mean_score'] for r in g],ddof=1) if len(g)>1 else 0:,.1f}",
            f"{np.mean([r['reaching_2048'] for r in g]):.1%}",f"{row['parameter_count']:,}",
            f"{np.mean([r['training_seconds'] for r in g]):.1f}",row['device']])+'</tr>')
    journal=escape((out/'journal.md').read_text())
    baseline_html=f"<p>Random legal-action baseline on the same 100 seeds: <b>{baseline['mean_score']:,.2f}</b> raw points (policy RNG seed42).</p>" if baseline else ''
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>2048 · Learned inputs and n-step TD</title><style>body{font:16px/1.6 system-ui;background:#faf8ef;color:#544c44;max-width:1200px;margin:40px auto;padding:0 24px}a{color:#197970}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;border-bottom:1px solid #d9d2c5;padding:10px}.scroll{overflow:auto}pre{white-space:pre-wrap;font:14px/1.6 system-ui;background:#f0ece2;padding:24px;border-radius:12px}</style>
<h1>Learned inputs and n-step TD</h1>
<p><b>Standard Transformers and MLPs learning online from their own games.</b> No engineered 80-feature inputs, teacher labels, lookahead search, or positional shaping in these experiments.</p>
<p>The Transformer has two standard encoder blocks, each with four-head self-attention, a feed-forward MLP, residual connections and layer normalization. Each of 16 cells becomes a 64-dimensional token, with a learned position vector. Inputs are either projected tile exponents or learned categorical embeddings. The final readout produces four Q-values.</p>
<p>All new models start from scratch. Each receives 131,072 transitions, batch size 512, learning rate 0.0001, discount 0.99, and the same update count. There are three training seeds per configuration. These short runs test early learning and cannot establish the best long-run player; the earlier 80-feature model had vastly more experience.</p>
<p>The core eight configurations vary architecture, input representation and one-step versus three-step TD. Two extra configurations change only reward from points / 128 to log2(1 + points) / log2(129). Logarithms change the objective by reducing the relative importance of big merges. All reported game scores remain raw points.</p>
<p>One-step and three-step updates use Double DQN and a replay buffer. Three-step returns are sampled from exploratory play without off-policy correction. Episode boundaries never mix two games, and time limits retain bootstrapping from the final board.</p>
<p>Each checkpoint is reloaded and evaluated on the same 100 held-out game seeds. Error bars show variation across three training seeds, not confidence intervals. CPU/GPU choice follows a local benchmark at the actual batch sizes. Parameter counts differ and are shown below.</p>
<p><a href="replay.html">Watch a fresh game from the leading configuration</a> · <a href="../deeper_q/moves_500_502.html">Inspect the earlier student's moves 500–502</a> · <a href="results.json">Raw results</a> · <a href="journal.md">Experiment journal</a></p>
<div class="scroll"><table><tr><th>Configuration</th><th>Training seeds</th><th>Score mean ± SD</th><th>2048 rate</th><th>Parameters</th><th>Training seconds / seed</th><th>Device</th></tr>'''+''.join(rows)+'''</table></div>'''+baseline_html+'''
<img src="scores.png" alt="Mean score by configuration with variation across training seeds">
<img src="curves.png" alt="Monitoring scores versus training transitions, grouped by architecture and input encoding">
<h2>Measurements and journal</h2><pre>'''+journal+'</pre></html>'
    (out/'index.html').write_text(html)


if __name__=='__main__':render()
