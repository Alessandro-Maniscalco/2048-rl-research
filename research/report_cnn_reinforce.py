"""CNN own-game REINFORCE report, separate from earlier Transformer experiments."""
import json
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from research.afterstate_teacher import write


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def render():
    base = Path('runs/research/scaled_transformer')
    protocol = read(base/'cnn_reinforce_protocol.json')
    if not protocol:
        return
    status = read(base/'status.json')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    rows = []
    endpoints = {}
    for job in [protocol['smoke']]+protocol['full_jobs']:
        folder = base/job['id']
        result, curve = read(folder/'result.json'), read(folder/'curve.json') or []
        diagnostic = job['config']['diagnostic_only']
        label = ('Diagnostic · 16 games/update' if diagnostic else
                 'Plain REINFORCE' if job['config']['algorithm'] == 'reinforce' else 'Leave-one-out baseline')
        phase = 'complete' if result.get('complete') else 'running' if status.get('job') == job['id'] else 'pending'
        if curve:
            for ax,key in zip(axes, ('mean_score', 'sampled_mean_score')):
                ax.plot([c['updates'] for c in curve], [c.get(key) for c in curve],
                    marker='o', linestyle=':' if diagnostic else '-', label=label)
        show = lambda x: '—' if x is None else f'{x:,.0f}'
        links = ' · '.join(f'<a href="{job["id"]}/{file}">{title}</a>' for file,title in
            [('config.json','Config'),('progress.json','Updates'),('first_training_game.json','One complete training game'),
             ('replay_last.html','Final greedy replay'),('evaluation.json','100 greedy games'),
             ('sampled_evaluation.json','100 sampled games')]
            if (folder/file).exists())
        rows.append(f'<tr><td>{label}</td><td>{phase}</td><td>{show(result.get("updates",curve[-1]["updates"] if curve else None))}</td>'
            f'<td>{show(result.get("training_transitions"))}</td><td>{show(result.get("mean_score"))}</td>'
            f'<td>{show(result.get("sampled_mean_score"))}</td><td>{links}</td></tr>')
        if not diagnostic and result.get('complete') and result.get('completed_budget'):
            endpoints[job['config']['algorithm']] = (folder,result)
    finding = 'The diagnostic is a code and stability check. Await both complete full-budget endpoints before comparing the two learning rules.'
    if set(endpoints) == {'reinforce', 'reinforce_loo'}:
        plain, loo = endpoints['reinforce'], endpoints['reinforce_loo']
        for key in ['first_complete_batch_sha256','initial_actor_sha256','episode_batch','updates',
                    'completed_training_games','lr','batch','seed','gamma','action_temperature']:
            assert plain[1][key] == loo[1][key]
        assert plain[1]['updates'] == 8 and plain[1]['completed_training_games'] == 512
        comparisons = {}
        for label,file in [('greedy','evaluation.json'),('sampled','sampled_evaluation.json')]:
            arms = {name:{e['seed']:e for e in read(folder/file)['episodes']}
                    for name,(folder,_) in [('plain',plain),('baseline',loo)]}
            assert arms['plain'].keys() == arms['baseline'].keys() and len(arms['plain']) == 100
            assert all(e['complete'] and not e['truncated'] for arm in arms.values() for e in arm.values())
            delta = np.array([arms['baseline'][s]['score']-arms['plain'][s]['score'] for s in sorted(arms['plain'])])
            bootstrap = delta[np.random.default_rng(8958750).integers(100,size=(20000,100))].mean(1)
            comparisons[label] = dict(plain_mean=float(np.mean([e['score'] for e in arms['plain'].values()])),
                baseline_mean=float(np.mean([e['score'] for e in arms['baseline'].values()])),
                difference=float(delta.mean()),ci95=np.quantile(bootstrap,[.025,.975]).tolist(),wins=int((delta>0).sum()))
        report = dict(complete=True,identical_first_complete_batch=True,comparisons=comparisons,
            note='Same source weights and complete-game/update budgets, one adaptation seed. Reused selection seeds; these intervals do not measure training-seed uncertainty.')
        write(base/'cnn_reinforce_comparison.json',report)
        g,s = comparisons['greedy'],comparisons['sampled']
        finding = (f'Final100-game means: greedy plain {g["plain_mean"]:,.0f}, baseline {g["baseline_mean"]:,.0f}; '
            f'sampled plain {s["plain_mean"]:,.0f}, baseline {s["baseline_mean"]:,.0f}. '
            f'Greedy baseline-minus-plain95% paired interval [{g["ci95"][0]:,.0f}, {g["ci95"][1]:,.0f}]. '
            'First collected batch hashes match. One adaptation seed and reused selection games; no automatic continuation.')
    for ax,title in zip(axes, ('Greedy policy: highest legal logit', 'Sampled policy: temperature = 1')):
        ax.set_title(title)
        ax.set_xlabel('Complete-game batch updates')
        ax.set_ylabel('Mean raw score · same 128 monitor seeds')
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(base/'cnn_reinforce_curves.png', dpi=150)
    plt.close(fig)
    body = f'''<!doctype html><html><meta charset="utf-8"><meta http-equiv="refresh" content="60">
    <title>CNN learning from actual game returns</title><style>body{{font:17px system-ui;max-width:1150px;margin:40px auto;padding:0 20px;line-height:1.5;color:#19212d}}
    table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #ddd}}img{{width:100%}}a{{color:#245ab5}}</style>
    <a href="index.html">All experiments</a><h1>CNN learning from actual game returns</h1>
    <p>{escape(protocol['question'])}</p>
    <p>Original CNN: <b>{protocol['baseline_selection']['mean_score']:,.0f}</b> mean on 100 selection games,
    <b>{protocol['baseline_monitor']['mean_score']:,.0f}</b> on 128 monitor games. Diagnostic runs are excluded from rankings.</p>
    <p><b>{finding}</b></p>
    <img src="cnn_reinforce_curves.png" alt="Greedy and sampled game score after each complete-game update">
    <table><tr><th>Experiment</th><th>Status</th><th>Updates</th><th>Training moves</th><th>Final 100 greedy</th><th>Final 100 sampled</th><th>Inspect</th></tr>{''.join(rows)}</table>
    <h2>Inputs, process, outputs</h2><p>16 cells → one-hot tile categories → learned whole-board, row and column convolutions → dense actor → 4 action logits.
    Illegal actions are masked. Training samples from softmax(logits); the main playing policy takes the largest legal logit.
    There are no 80 engineered features. The inherited input clips ranks above 15; the old critic is frozen and its forward pass skipped.</p>
    <p>Collect B complete games with unchanged weights. For move t in game i, compute the actual reward-to-go:
    G<sub>i,t</sub> = Σ<sub>u=t to terminal</sub> merge_points<sub>i,u</sub>/128. There is no discount, logarithm, corner reward, or teacher estimate.</p>
    <p>Plain loss: L = −(1/B) Σ<sub>i,t</sub> log π(a<sub>i,t</sub>|s<sub>i,t</sub>) G<sub>i,t</sub>.</p>
    <p>Baseline arm: b<sub>i,t</sub> = (1/(B−1)) Σ<sub>j≠i</sub> G<sub>j,t</sub>, with zero after another game ends.
    Replace G by G−b. A game's own actions and returns do not enter its baseline. This is a Monte Carlo baseline, not a learned critic or TD target.</p>
    <p>Accumulate gradients over 512-board microbatches, clipping the final gradient norm to 0.5, then take exactly one Adam step at learning rate 0.00001.
    Microbatches do not each take an optimizer step. Discard interrupted incomplete batches. Both full arms use 8 updates of 64 complete games, starting from original weights and a fresh optimizer.</p>
    <p>Final 100 games and repeated 128-game monitors are selection data. Reserved final tests remain unused. Curves can fall; lower policy loss is not a measure of strength.
    Different policy trajectories have different numbers of moves and runtime even with identical game/update budgets.</p>
    <p><a href="cnn_reinforce_protocol.json">Prespecified protocol</a> · <a href="cnn_dagger.html">Why teacher imitation was insufficient</a> · <a href="cnn_spawn_safety/index.html">Independent frozen-policy safety test</a></p></html>'''
    (base/'cnn_reinforce.html').write_text(body)


if __name__ == '__main__':
    render()
