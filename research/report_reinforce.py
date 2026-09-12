"""Show plain REINFORCE with its actual complete-game batch semantics."""
import json
from pathlib import Path
from html import escape
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1] / 'runs/research/scaled_transformer'


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def render(base=BASE):
    protocol = read(base / 'reinforce_protocol.json')
    if not protocol:
        return
    status = read(base / 'status.json', {})
    parts = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    plotted = False
    for job in protocol['jobs']:
        path = base / job['id']
        result = read(path / 'result.json', {})
        curve = read(path / 'curve.json', [])
        progress = read(path / 'progress.json', [])
        phase = 'complete' if result.get('complete') else 'running' if status.get('job') == job['id'] else job.get('status', 'queued')
        parts.append(f'<h2>{escape(job["id"])}</h2><p>{escape(phase)}. {escape(job["question"])}</p>')
        if result.get('complete'):
            selected = read(path / 'best_selection.json')['summary']['mean_score']
            parts.append(f'<p><b>Final mean on 100 games: {result["mean_score"]:,.2f}</b>; '
                f'monitor-selected mean: {selected:,.2f}. Actual new moves: {result["training_transitions"]:,}; '
                f'complete training games: {result["completed_training_games"]:,}; '
                f'optimizer steps: {result["updates"]:,}; training time: {result["training_seconds"]:,.1f} s.</p>')
            parts.append(f'<p><a href="{job["id"]}/replay_last.html">Watch final trained policy</a> · '
                         f'<a href="{job["id"]}/replay_best.html">Watch monitor-selected policy (may be the initialization)</a> · '
                         f'<a href="{job["id"]}/config.json">Exact configuration</a></p>')
            if result.get('sampled_mean_score') is not None:
                parts.append(f'<p>Final sampled-policy mean on 100 games at training temperature '
                             f'{job["config"].get("action_temperature",1):g}: {result["sampled_mean_score"]:,.2f}.</p>')
        if progress:
            p = progress[-1]
            parts.append(f'<p>Latest update: {p["updates"]}; moves: {p["transitions"]:,}; '
                f'mean sampled training-game score: {p["mean_training_score"]:,.1f}; '
                f'action entropy: {p["entropy"]:.3f}; gradient norm before clipping: {p["gradient_norm_before_clip"]:,.1f}. '
                f'Collection/recomputed log-probability difference: {p["max_behavior_logp_difference"]:.2g}.</p>')
            if p.get('baseline_mode') == 'leave_one_out_time':
                parts.append(f'<p>Baseline experiment: raw return standard deviation {p["return_std"]:.2f}; '
                             f'centered learning-weight standard deviation {p["learning_weight_std"]:.2f}. '
                             'These are return statistics, not an estimate of gradient variance.</p>')
        if curve and not job['config'].get('diagnostic_only'):
            family = 'With baseline' if job['config'].get('algorithm') == 'reinforce_loo' else 'Plain'
            label = family+f' · B={job["config"]["episode_batch"]} · T={job["config"].get("action_temperature",1):g} · '+('Teacher initialization' if job['config'].get('initial_actor_checkpoint') else 'Scratch')
            for ax, key in zip(axes, ('transitions', 'training_seconds')):
                ax.plot([p[key] for p in curve], [p['mean_score'] for p in curve], 'o-', label=label)
            plotted = True
        if curve and all('sampled_mean_score' in r for r in curve):
            detail, ax = plt.subplots(figsize=(9,3.5), constrained_layout=True)
            x = [r['transitions'] for r in curve]
            ax.plot(x,[r['mean_score'] for r in curve],'o-',label='Greedy argmax')
            ax.plot(x,[r['sampled_mean_score'] for r in curve],'o--',label=f'Sampled at training T={job["config"].get("action_temperature",1):g}')
            ax.set(xlabel='New training moves',ylabel='Mean score on 128 fixed monitoring games')
            ax.grid(alpha=.2);ax.legend()
            name = job['id']+'_sampling_curve.png'
            detail.savefig(base/name,dpi=130);plt.close(detail)
            parts.append(f'<img src="{name}" alt="Greedy and sampled policy scores during training">')
    for ax, xlabel in zip(axes, ('New training moves', 'Training seconds, evaluation excluded')):
        ax.set(xlabel=xlabel, ylabel='Mean raw score on 128 fixed greedy games')
        ax.grid(alpha=.2)
        if plotted:
            ax.legend()
    fig.savefig(base / 'reinforce_curves.png', dpi=140)
    plt.close(fig)
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="60"><title>Plain REINFORCE on 2048</title>
<style>body{font:16px/1.65 system-ui;max-width:1200px;margin:30px auto;padding:20px;background:#faf8ef;color:#544c44}a{color:#197970}img{width:100%}pre{white-space:pre-wrap;background:#eee9df;padding:20px}td,th{padding:8px;text-align:left}</style>
<h1>Plain REINFORCE: learn from complete games</h1>
<p><a href="index.html">All research</a> · <a href="transformer_online.html">PPO own-game comparison</a> · <a href="reinforce_protocol.json">Full experiment protocol</a></p>
<p>Input: 16 tile categories → learned tile and position embeddings → 4 Transformer blocks (attention plus MLP) → four move logits. Eight rotated/reflected views share weights; move logits are mapped back before averaging. Mask illegal moves, then softmax to probabilities. Sample during training; choose the largest probability during evaluation. There is no scalar value head or Q prediction.</p>
<p>Freeze the policy and play 128 complete games. Finished games wait for the others. Compute each move’s actual remaining reward, backpropagate through chunks of 128 boards, accumulate every chunk’s gradient, then take exactly ONE Adam step. Throw this data away and play a fresh batch using the new policy. The 128-game learning batch is different from the 128-board memory microbatch.</p>
<pre>r_t = actual merge points / 128
G_t = r_t + r_(t+1) + ... + final reward       (gamma = 1)
L = -(1 / B) Σ_games Σ_moves G_t log πθ(a_t | s_t), B = 128 complete games
θ ← Adam(θ, gradient of L), learning rate 0.0001
Global gradient norm capped at 0.5; no baseline, return centering,
adaptive normalization, critic, TD bootstrap, replay, PPO clipping, or entropy bonus.</pre>
<p>Dividing rewards by a fixed 128 changes scale, not the desired raw-score objective. Do not divide each episode by its length: the objective is expected total game score. Each whole batch may slightly exceed the requested move budget; actual counts are reported. If a batch is interrupted, incomplete Monte Carlo targets are discarded. CPU games and GPU inference/learning alternate; only one GPU learner runs.</p>
<p>The teacher-initialized arm copies the same supervised actor used by PPO and drops the scalar head. No teacher labels are queried during REINFORCE. The scratch arm uses the same architecture with fresh weights. This separates an algorithm change from the benefit of pretraining; teacher pretraining is extra compute. The first short GPU check is diagnostic only. A falling loss alone is not proof of better play; compare fixed-game score curves and replay decisions. The separate 100-game suite is selection data, and reserved final-test seeds remain unused.</p>
<p>Published large-lab examples differ from this plain baseline: OpenAI’s <a href="https://openai.com/index/instruction-following/">InstructGPT used PPO</a>; Cohere’s <a href="https://aclanthology.org/2024.acl-long.662/">RLOO work uses other sampled responses as a baseline</a>; DeepSeek-R1 uses <a href="https://arxiv.org/html/2501.12948v1#S2.SS2.SSS1">GRPO with group-relative rewards and clipped updates</a>. These are published examples, not claims about undisclosed current recipes. Their language-model results do not establish the best method for 2048.</p>
<h2>Separate next experiment: REINFORCE with a baseline</h2>
<p>The plain runs remain unchanged. This variant subtracts the mean return at the same move index in the OTHER 127 completed games. An already-ended other game contributes zero future reward. No return from a game enters its own baseline. This is a time-based control for noisy returns, not a learned critic or teacher target. It is inspired by leave-one-out baselines; it is not an exact reproduction of language-model RLOO, which groups answers to the same prompt.</p>
<pre>b_(i,t) = (1 / (B-1)) Σ_(j ≠ i) G_(j,t), with G_(j,t)=0 after game j ends
A_(i,t) = G_(i,t) - b_(i,t)
L_baseline = -(1/B) Σ_i Σ_t A_(i,t) log πθ(a_(i,t) | s_(i,t))</pre>
<p>The baseline is independent of the current game’s sampled actions. Before optimizer transformations, subtracting it leaves the expected policy gradient unchanged and may reduce variance. All other settings stay the same: initialization, reward, batch sizes, LR, gradient-norm cap and new-experience budget. Actual variance reduction and game improvement must be measured; neither is assumed.</p>
<h2>Update-frequency comparison</h2>
<p>A separate follow-up reduces the completed-game batch from 128 to 16, keeping GPU memory chunks at 128 boards and retaining the same baseline formula, actor initialization, reward, learning rate and approximate new-move budget. This permits more frequent policy updates but makes each update noisier and may cost more collection time. The 128-game baseline run took only 26 optimizer steps in roughly one million moves. More updates are a hypothesis to test, not an assumed advantage.</p>
<h2>Exploration comparison: lower sampling temperature</h2>
<p>The next distinct test keeps 128 complete games per update and changes action temperature from 1 to 0.25. The original supervised actor scores substantially better with greedy actions than with ordinary sampling. Concentrating probability on preferred actions may collect stronger, longer training games, but may also prevent useful exploration. The same temperature is applied during collection AND when differentiating the log probability. Greedy argmax evaluation is unchanged. New runs monitor both the sampled policy actually trained and greedy play.</p>
<pre>π_T(a|s) = softmax(legal logits / T), with T = 0.25 in this experiment
L = -(1/B) Σ_i Σ_t (G_(i,t) - b_(i,t)) log π_T(a_(i,t)|s_(i,t))</pre>
'''
    comparison = read(base / 'reinforce_comparison.json')
    if comparison:
        listed = {r['id'] for r in comparison['runs']}
        for job in protocol['jobs']:
            if job['id'] in listed or job['config'].get('diagnostic_only'):
                continue
            result = read(base / job['id'] / 'result.json', {})
            chosen = read(base / job['id'] / 'best_selection.json', {})
            if result.get('complete') and chosen:
                family = 'REINFORCE with baseline' if job['config']['algorithm'] == 'reinforce_loo' else 'Plain REINFORCE'
                label = family+f' · B={job["config"]["episode_batch"]} · T={job["config"].get("action_temperature",1):g} · '+('supervised actor' if job['config'].get('initial_actor_checkpoint') else 'scratch')
                comparison['runs'].append(dict(id=job['id'], label=label,
                    selected_score=chosen['summary']['mean_score'],
                    **{k:result[k] for k in ('mean_score','transitions','updates','training_seconds')}))
        page += '<h2>Completed runs: final versus selected policy</h2><table><tr><th>Method and initialization</th><th>Final 100-game mean</th><th>Selected 100-game mean</th><th>New moves</th><th>Updates</th><th>Training seconds</th></tr>'
        for r in comparison['runs']:
            page += f'<tr><td>{escape(r["label"])}</td><td>{r["mean_score"]:,.2f}</td><td>{r["selected_score"]:,.2f}</td><td>{r["transitions"]:,}</td><td>{r["updates"]:,}</td><td>{r["training_seconds"]:.1f}</td></tr>'
        page += '</table><p>One training seed per method. Selected checkpoints may be the unchanged initialization; initial teacher pretraining is extra compute. REINFORCE completes full game batches, so new-move counts differ slightly. These are selection games, not the reserved final test.</p>'
    if (base / 'reinforce_diagnostics/actor_last/index.html').exists():
        page += '<p><a href="reinforce_diagnostics/actor_last/index.html">Inspect the plain REINFORCE actor’s late-game decisions and exact death probabilities</a></p>'
    if (base / 'reinforce_diagnostics/baseline_last/index.html').exists():
        page += '<p><a href="reinforce_diagnostics/baseline_last/index.html">Inspect the baseline actor’s actual replay decisions</a></p>'
    if (base / 'reinforce_diagnostics/b16_last/index.html').exists():
        page += '<p><a href="reinforce_diagnostics/b16_last/index.html">Inspect the smaller-batch actor’s actual replay decisions</a></p>'
    temperature_control = read(base / 'reinforce_temperature_initial_comparison.json')
    if temperature_control and temperature_control.get('complete'):
        page += ('<h2>Collection control: the same initial actor at different temperatures</h2>'
                 '<table><tr><th>Action selection</th><th>Mean score on 100 games</th></tr>'
                 f'<tr><td>Sample at T=1</td><td>{temperature_control["temperature1_mean"]:,.2f}</td></tr>'
                 f'<tr><td>Sample at T=0.25</td><td>{temperature_control["temperature025_mean"]:,.2f}</td></tr>'
                 f'<tr><td>Greedy argmax</td><td>{temperature_control["greedy_mean"]:,.2f}</td></tr></table>'
                 '<p>No weights were updated. Lower temperature produced stronger initial sampled games in this check, but this is not a learning gain. The training experiment below must improve upon its own initial policy.</p>')
    sampling = read(base / 'reinforce_sampling_comparison.json')
    if sampling and sampling.get('complete'):
        page += '<h2>Did the sampled policy improve even when greedy play weakened?</h2><p>Yes in this fixed-policy check. REINFORCE optimizes scores while sampling from its probabilities; argmax selects a different policy. The table uses the same 100 selection game seeds, unchanged weights and no search. Sampling uses temperature 1 and an independent, recorded action RNG per game.</p><table><tr><th>Frozen policy</th><th>Sample moves from probabilities</th><th>Always take argmax</th></tr>'
        names = {'initial_actor':'Original supervised actor', 'reinforce_final':'Plain REINFORCE final',
                 'baseline_final':'REINFORCE with baseline final', 'ppo_final':'PPO final'}
        for r in sampling['runs']:
            page += f'<tr><td>{names.get(r["id"],r["id"])}</td><td>{r["sampled_mean"]:,.2f}</td><td>{r["greedy_mean"]:,.2f}</td></tr>'
        page += '</table><p>Plain REINFORCE raised sampled mean from 3,028.68 to 5,352.44; the baseline reached 5,451.16. This supports learning on the stochastic objective despite no clear greedy gain over initialization. It does not make sampling the stronger playing method: greedy remains higher for these same models. One training seed and one recorded action-sampling stream per game; reserved final tests remain unused. <a href="reinforce_sampling_comparison.json">Paired game intervals and protocol</a>.</p>'
    page += ('<img src="reinforce_curves.png" alt="Fixed-game mean score versus training moves and elapsed training time">'
             if plotted else '<p>Full-run learning curves will appear here after the short GPU verification.</p>')
    (base / 'reinforce.html').write_text(page + ''.join(parts) + '</html>')


if __name__ == '__main__':
    render()
