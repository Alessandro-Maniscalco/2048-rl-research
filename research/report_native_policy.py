"""Keep action-label fitting separate from complete-game strength."""
import json
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def render():
    base=Path('runs/research/scaled_transformer')
    protocol=read(base/'native_policy_protocol.json')
    if not protocol:return
    data=read(Path(protocol['dataset'])/'manifest.json',{})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));rows=[]
    for job in protocol['smoke_jobs']+protocol['full_jobs']+protocol.get('stage_jobs',[]):
        folder=base/job['id'];curve=read(folder/'curve.json',[]);result=read(folder/'result.json',{})
        diagnostic=job['config']['diagnostic_only']
        label=('Pretrained encoder' if job['config']['initialization']=='encoder' else 'Random encoder')+(' · diagnostic' if diagnostic else '')
        if job['config']['initialization']=='continuation':
            label='Continuation · '+('more early-game sampling' if job['config'].get('stage_fraction') else 'uniform control')
        if curve:
            axes[0].plot([r['updates'] for r in curve],[r['mean_score'] for r in curve],marker='o',linestyle=':' if diagnostic else '-',label=label)
            axes[1].plot([r['updates'] for r in curve],[r['action_agreement'] for r in curve],marker='o',linestyle=':' if diagnostic else '-',label=label)
        score='—' if result.get('mean_score') is None else f'{result["mean_score"]:,.0f}'
        links=' · '.join(f'<a href="{job["id"]}/{file}">{name}</a>' for file,name in
            [('replay_last.html','Final replay'),('config.json','Config'),('curve.json','Curve'),('evaluation.json','100games')]
            if (folder/file).exists())
        rows.append(f'<tr><td>{label}</td><td>{curve[-1]["updates"] if curve else 0}</td><td>{score}</td><td>{links}</td></tr>')
    for ax,y in zip(axes,['Mean raw game score · same128monitor seeds','Expert-action agreement · held-out boards']):
        ax.set_xlabel('Adam updates');ax.set_ylabel(y);ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(base/'native_policy_curves.png',dpi=150);plt.close(fig)
    followup=''
    comparison=read(base/'native_policy_comparison.json')
    if comparison:
        lo,hi=comparison['ci95']
        followup=(f'<p><b>Initial comparison complete:</b> pretrained encoder {comparison["encoder_mean"]:,.0f} '
            f'versus random encoder {comparison["scratch_mean"]:,.0f} on100complete selection games. '
            f'Paired difference {comparison["encoder_minus_scratch"]:,.0f}, 95% interval [{lo:,.0f}, {hi:,.0f}]. '
            'Both remain weak despite improved imitation accuracy. The actor heads and first minibatches were identical; '
            'source weights and expert data were unchanged. Both full replays passed all transition checks and23selected CPU/MPS matches. '
            '<a href="native_policy_comparison.json">Saved comparison</a>.</p>')
    if protocol.get('stage_jobs'):
        followup+=f'<h2>Testing the data imbalance</h2><p>{escape(protocol["stage_question"])}</p>'
    stages=read(base/'native_policy_stage_comparison.json')
    if stages:
        lo,hi=stages['ci95']
        followup+=(f'<p><b>Longer sampling comparison complete:</b> uniform {stages["uniform_mean"]:,.0f}; '
            f'more early-game sampling {stages["stage_mean"]:,.0f}. Paired change {stages["stage_minus_uniform"]:,.0f}, '
            f'95% interval [{lo:,.0f}, {hi:,.0f}]. Both started the exact same policy and both remain weak. '
            'Early-state weighting alone has not solved imitation. No more automatic continuation. '
            '<a href="native_policy_stage_comparison.json">Results</a> · '
            '<a href="../endgame_tablebase/index.html">New stronger compact engine investigation</a>.</p>')
    page=f'''<!doctype html><html><meta charset="utf-8"><meta http-equiv="refresh" content="60">
    <title>Learning the search player's moves</title><style>body{{font:17px system-ui;line-height:1.5;max-width:1100px;margin:40px auto;padding:0 20px;color:#202b38}}img{{width:100%}}table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}a{{color:#245ab5}}</style>
    <a href="index.html">All experiments</a><h1>Learning the search player's moves</h1>
    <p>{escape(protocol['question'])}</p><p><b>State: {escape(protocol['stage'])}.</b>
    Recorded boards: {data.get('labelled_states','building')}; fitting: {data.get('fitting_states','—')}; held out: {data.get('validation_states','—')}.
    Each of80complete expert games belongs entirely to fitting or validation. No games were selected by score.</p>
    <img src="native_policy_curves.png" alt="Whole-game scores and separately held-out expert action agreement">
    <table><tr><th>Initialization</th><th>Updates</th><th>Final100-game mean</th><th>Inspect</th></tr>{''.join(rows)}</table>
    {followup}
    <h2>Input → prediction → learning</h2><p>Sixteen tile ranks →18one-hot channels → learned full-board, row and column convolutions →1024features →256 →64 →4action logits.
    Illegal moves receive zero probability. The target is the search player's recorded action on that board.
    Loss = −mean log π(expert action | board). Adam updates encoder and actor; there is no value head or critic.</p>
    <p>Batch512, learning rate0.0001, gradient norm cap0.5. The diagnostic gets128updates; each full comparison gets4096updates or900training seconds.
    Both full arms start fresh from their prescribed initialization, never the diagnostic. Whole fitting arrays are held on MPS to reduce repeated transfer overhead.
    Validation uses a fixed8192-board sample from16separate expert games, never optimized. Monitoring plays128standard new games; final evaluation uses100selection seeds.
    These game sets are reused across experiments and are not a final untouched test.</p>
    <p>Good imitation on expert states does not establish good play: mistakes can put the student in unfamiliar states.
    Expert precomputation, search and recorded game generation are prior costs. A student does not run those searches during a game.
    Ranks16and17have distinct trainable channels; ranks above17would still clip. Encoder transfer does not copy the previous policy head.</p>
    <a href="native_policy_protocol.json">Prespecified protocol</a> · <a href="native_expert_data/manifest.json">Dataset provenance</a> ·
    <a href="../tablebase_search/index.html">The stronger expert</a></html>'''
    (base/'native_policy.html').write_text(page)


if __name__=='__main__':render()
