"""Render the human review and compare saved continuation lineages."""
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
b=root/'runs/research/mlp_overnight'
rows={r['name']:r for r in json.loads((b/'results.json').read_text()) if 'score' in r}
fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained')
groups={
    'Original rate 0.0003': ['tune_reward_mode_corner_snake']+sorted(n for n in rows if n.startswith('continuation_')),
    'Rate 0.00003 from 73M': ['continuation_06','refine_lr_00003']+sorted(n for n in rows if n.startswith('refined_continuation_')),
    'Rate 0.0001 from 73M': ['continuation_06','refine_lr_0001'],
    'Batch 1024 from 73M': ['continuation_06','refine_batch1024'],
}
plan=b/'adaptive_plan.json'
if plan.exists():
    anchor=json.loads(plan.read_text())['parent']
    for name,label in [('adaptive_control','Late control'),('adaptive_gamma_0999','Late gamma .999'),
                       ('adaptive_gamma_1','Late gamma 1'),('adaptive_tau_0001','Late target tau .001'),
                       ('adaptive_batch1024_double_steps','Late batch 1024, twice the steps')]:
        if name in rows:groups[label]=[anchor,name]
    selection=b/'adaptive_selection.json'
    if selection.exists():
        groups['Selected late branch']=[json.loads(selection.read_text())['chosen_continuation']]+sorted(n for n in rows if n.startswith('adaptive_continuation_'))
plan=b/'exploration_plan.json'
if plan.exists():
    anchor=json.loads(plan.read_text())['parent']
    for name,label in [('explore_epsilon001','Epsilon .01'),('explore_epsilon010','Epsilon .1'),
                       ('explore_epsilon020','Epsilon .2'),('explore_capacity200k','Replay 200k')]:
        if name in rows:groups[label]=[anchor,name]
    selection=b/'exploration_selection.json'
    if selection.exists():
        groups['Selected exploration branch']=[json.loads(selection.read_text())['chosen_continuation']]+sorted(n for n in rows if n.startswith('exploration_continuation_'))
selection=b/'robustness_selection.json'
if selection.exists():
    original='robust_epsilon0.05_seed1'
    groups['5% exploration, new seed']=[original]+sorted(n for n in rows if n.startswith('robust_continuation_'))
    if any(n.startswith('consensus_continuation_') for n in rows):
        groups['1% exploration, selected by seed mean']=[json.loads(selection.read_text())['chosen_continuation']]+sorted(n for n in rows if n.startswith('consensus_continuation_'))
for label,names in groups.items():
    data=[rows[n] for n in names if n in rows]
    x=[r['total_steps']/1e6 for r in data]
    axes[0].plot(x,[r['score'] for r in data],'-o',label=label,markersize=4)
    axes[1].plot(x,[100*r['summary']['tile_reaching_rates']['2048'] for r in data],'-o',markersize=4)
axes[0].set(ylabel='Mean raw game score',title='Final checkpoints: 100 reused validation seeds, one training lineage')
axes[0].legend(fontsize=9)
axes[1].set(xlabel='Cumulative training transitions (millions)',ylabel='Games reaching 2048 (%)')
for ax in axes:ax.grid(alpha=.2)
fig.savefig(b/'continuation_comparison.png',dpi=130);plt.close(fig)
text=(root/'research/overnight_review.md').read_text()
page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><meta http-equiv="refresh" content="120"><title>MLP experiment interpretation</title><style>body{font:17px/1.6 system-ui;max-width:1000px;margin:30px auto;padding:0 20px;background:#f5f2ed}pre{white-space:pre-wrap;font:inherit}img{max-width:100%}</style><a href="index.html">Live experiment journal</a> · <a href="latest_validation_replay.html">Selected validation replay</a><p>Validation comparisons; checkpoints from the same lineage are correlated. No fresh test scores appear in this chart.</p><img src="continuation_comparison.png" alt="Scores and 2048-reaching rates by cumulative training transitions"><pre>'''+html.escape(text)+'</pre>'
if (b/'exploration_validation_replay.html').exists():
    page=page.replace('<pre>','<p><a href="exploration_validation_replay.html">Watch the selected lower-exploration validation game</a> (best of 100 validation games).</p><pre>',1)
(b/'review.html').write_text(page)
