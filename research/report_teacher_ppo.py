"""Report own-game PPO separately from fixed-label teacher supervision."""
from pathlib import Path
import json
from html import escape
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parents[1]/'runs/research/scaled_transformer'


def read(path,default=None):return json.loads(path.read_text()) if path.exists() else default


def render(base=BASE):
    protocol=read(base/'transformer_online_protocol.json')
    if not protocol:return
    status=read(base/'status.json',{});parts=[]
    for job in protocol['jobs']:
        path=base/job['id'];result=read(path/'result.json',{});curve=read(path/'curve.json',[])
        progress=read(path/'progress.json',[]);config=read(path/'config.json',job['config'])
        state='complete' if result.get('complete') else 'running' if status.get('job')==job['id'] else job.get('status','queued')
        if config.get('diagnostic_only'):state+=' (short verification, not a competitive training budget)'
        parts.append(f'<h2>{escape(job["id"])}</h2><p>{escape(state)}</p><p>{escape(job["question"])}</p>')
        if result.get('complete'):
            chosen=read(path/'best_selection.json',{}).get('summary',{})
            parts.append(f'<p>Final mean over100games: {result["mean_score"]:,.2f}; '
                f'monitor-selected mean: {chosen.get("mean_score",0):,.2f}. '
                f'New environment transitions: {result["training_transitions"]:,}; '
                f'gradient updates: {result["updates"]:,}; training seconds: {result["training_seconds"]:,.1f}.</p>')
        if curve:
            fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
            axes[0].plot([r['transitions'] for r in curve],[r['mean_score'] for r in curve],'o-')
            axes[0].set_ylabel('Mean raw score:128 greedy monitoring games')
            if progress:
                axes[1].plot([r['transitions'] for r in progress],[r['approx_kl'] for r in progress]);axes[1].set_ylabel('Approximate policy KL')
                axes[2].plot([r['transitions'] for r in progress],[r['value_loss'] for r in progress]);axes[2].set_ylabel('Own-policy value MSE')
            for ax in axes:ax.set_xlabel('New environment transitions');ax.grid(alpha=.2)
            name=job['id']+'_online.png';fig.savefig(base/name,dpi=130);plt.close(fig)
            parts.append(f'<img src="{name}" alt="Greedy game scores, policy change and critic loss versus new transitions">')
        if (path/'replay_best.html').exists():parts.append(f'<p><a href="{job["id"]}/replay_best.html">Watch the selected policy</a></p>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="60"><title>Transformer: own-game PPO</title>
<style>body{font:16px/1.6 system-ui;max-width:1250px;margin:30px auto;padding:20px;background:#faf8ef;color:#544c44}a{color:#197970}img{width:100%}pre{white-space:pre-wrap;background:#eee9df;padding:20px}</style>
<h1>Can the teacher-initialized Transformer improve through its own games?</h1>
<p><a href="transformer_teacher.html">Teacher-supervision experiments</a> · <a href="index.html">All research</a> · <a href="transformer_online_protocol.json">Full protocol</a></p>
<p>This is a new learning objective. The actor begins either with copied teacher-policy features or fresh weights. The scalar head is reset to zero: the teacher’s expected future score is not the weak student’s expected future score. Resetting the scalar head leaves every actor logit unchanged.</p>
<p>The CPU simulates128games. The GPU predicts move probabilities, samples are drawn with a separate recorded NumPy RNG, and64steps per game form each rollout. PPO updates use only this newly collected rollout; the next rollout uses the updated policy. There is no fixed teacher dataset, teacher loss or replay buffer in this phase. CPU simulation and GPU prediction/update phases alternate; this is not an asynchronous actor/learner system.</p>
<pre>Learning reward r = actual merge points / 128
TD residual: δ_t = r_t + γ V(s_next) - V(s_t); terminal next values are zero
GAE: A_t = δ_t + γ λ A_next; the recursion stops at episode boundaries
Value target: G_t = A_t + old V(s_t)
Probability ratio: ρ_t = π_new(a_t|s_t) / π_old(a_t|s_t)
Loss = -mean[min(ρ A, clip(ρ, 0.8, 1.2) A)]
       + 0.5 × mean[(V - G)²] - 0.005 × policy entropy</pre>
<p>Gamma=1 and lambda=0.95. Advantages are normalized within each rollout. Two shuffled passes are allowed, with minibatches of128; an approximate KL threshold0.03 can end the remaining passes early. Natural termination never bootstraps; a time limit bootstraps its final board but does not link advantages across a reset. All decisions mask illegal moves. Evaluation uses greedy actions on fixed128monitoring games and100selection games. Reserved final-test seeds remain untouched.</p>
<p>The short GPU verification run is excluded from competitive rankings. Larger runs require checking its actual runtime, numerical stability and checkpoint/replay output. Algorithm reference: <a href="https://arxiv.org/abs/1707.06347">Schulman et al., PPO</a>.</p>
'''
    (base/'transformer_online.html').write_text(page+''.join(parts)+'</html>')


if __name__=='__main__':render()
