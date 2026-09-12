"""A focused report for full-value n-tuple to Transformer supervision."""
import json
from pathlib import Path
from html import escape
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


BASE=Path(__file__).resolve().parents[1]/'runs/research/scaled_transformer'


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def render(base=BASE):
    protocol=read(base/'transformer_teacher_protocol.json')
    if not protocol:return
    data=read(base/'transformer_teacher_data/manifest.json')
    status=read(base/'status.json',{})
    reference=read(base/'transformer_teacher_reference.json',{}).get('summary',{})
    sections=[]
    for identifier in protocol['jobs']:
        path=base/identifier;teacher=read(path/'teacher_curve.json',[]);curve=read(path/'curve.json',[])
        config=read(path/'config.json',{});result=read(path/'result.json',{})
        selected=read(path/'best_selection.json',{}).get('summary',{})
        state='complete' if result.get('complete') else 'running' if status.get('job')==identifier else 'queued'
        if config.get('diagnostic_only'):state+=' — fitting diagnostic, excluded from competitive rankings'
        latest=teacher[-1] if teacher else {}
        def number(x):return f'{x:,.2f}' if x is not None else 'Pending'
        sections.append(f'<h2>{escape(identifier)}</h2><p>Status: {state}. '
            f'Latest supervised update: {latest.get("updates",0):,}. No online RL transitions have been used in this stage.</p>'
            f'<p>Held-out value MSE: {number(latest.get("value_mse"))}; teacher move agreement: '
            f'{number(100*latest["teacher_action_agreement"]) if latest else "Pending"}%. '
            f'Final100-game score: {number(result.get("mean_score"))}; '
            f'monitor-selected100-game score: {number(selected.get("mean_score"))}.</p>')
        if config:
            metadata=read(path/'last/metadata.json',{})
            sections.append(f'<p>Network: width {config["width"]}, {config["depth"]} Transformer blocks, '
                f'{config["heads"]} attention heads; {metadata.get("parameter_count",0):,} learned parameters.</p>')
            policy_value=config.get('algorithm')=='teacher_policy_value'
            fitting=config.get('fitting_boards') if policy_value else config['fitting_afterstates']
            sections.append(f'<p>Fitting {"current boards" if policy_value else "afterstates"}: {fitting:,}. '
                f'Fixed label-validation sample: {config["validation_boards_used"]:,} boards from held-out games. '
                f'Training-set mean μ = {config["value_offset"]:,.3f}; standard deviation σ = {config["value_scale"]:,.3f}, '
                'both measured in points/128. The held-out data do not determine these constants.</p>')
            if policy_value:
                sections.append('<p><b>Different prediction task:</b> current board → shared Transformer → '
                    'four move logits plus one scalar value. The legal move logits are trained with cross entropy '
                    'against softmax(teacher Q / 4). The scalar is trained to predict the maximum legal teacher Q, '
                    'with normalized MSE weighted by0.05. Play uses legal argmax of the move logits. '
                    'The logits are not Q-values; no slides or search are used to calculate their scores. '
                    'All8 views share weights and their move outputs are mapped back before averaging. '
                    f'Initialization: {escape(config["initialization"])}. This is supervised imitation, not online RL.</p>')
                if config.get('fit_subset_size'):
                    sections.append(f'<p>This diagnostic samples only {config["fitting_boards"]} fixed fitting boards '
                        'to test whether the model can reproduce their teacher labels. Normalization remains that of the '
                        'original full fitting partition. It is not a proposed gameplay improvement.</p>')
                fit_curve=read(path/'fit_curve.json',[])
                if fit_curve:
                    fitted=fit_curve[-1]
                    sections.append(f'<p>Fixed fitting-sample teacher agreement: {100*fitted["teacher_action_agreement"]:.2f}%; '
                        f'KL divergence from the teacher probabilities: {fitted["policy_kl"]:.4f}. '
                        'KL subtracts the teacher’s own entropy from cross entropy; zero means the probabilities match.</p>')
            if config.get('grouped_actions'):
                sections.append(f'<p>This is a continuation from the first 16,384 supervised updates. '
                    f'Each new update samples {config["batch"]} whole boards and their legal alternatives. '
                    f'Loss: normalized value MSE + {config.get("action_loss_weight",0):g} × action cross entropy. '
                    'Teacher and student action probabilities use softmax((merge points/128 + U)/4). '
                    'The temperature corresponds to512 raw points, retaining near-tied moves. '
                    'The graphs below count additional updates in this phase.</p>')
            if config.get('stage_balanced'):
                sections.append('<p>Sampling change: choose one of four maximum-tile stages with equal probability, '
                    'then choose a fitting board from that stage. Stages are ≤256, 512–2048, 4096–8192 and ≥16384. '
                    'This changes training coverage only; teacher labels, normalization and model inputs stay the same.</p>')
        if teacher:
            fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
            axes[0].plot([r['updates'] for r in teacher],[r['normalized_value_mse'] for r in teacher]);axes[0].set_ylabel('Held-out normalized value MSE')
            axes[1].plot([r['updates'] for r in teacher],[100*r['teacher_action_agreement'] for r in teacher]);axes[1].set_ylabel('Teacher-best action agreement (%)')
            fitted=read(path/'fit_curve.json',[])
            if fitted:
                axes[1].lines[0].set_label('Held-out games')
                axes[1].plot([r['updates'] for r in fitted],[100*r['teacher_action_agreement'] for r in fitted],
                    '--',label='Fixed fitting sample');axes[1].legend()
            if curve:axes[2].plot([r['updates'] for r in curve],[r['mean_score'] for r in curve])
            axes[2].set_ylabel('Mean raw score:128 monitoring games')
            for ax in axes:ax.set_xlabel('Supervised gradient updates');ax.grid(alpha=.2)
            name=f'{identifier}_distillation.png';fig.savefig(base/name,dpi=130);plt.close(fig)
            sections.append(f'<img src="{name}" alt="Held-out value errors, move agreement and game score versus supervised updates">')
        if (path/'replay_best.html').exists():sections.append(f'<p><a href="{identifier}/replay_best.html">Watch the student</a></p>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="60"><title>N-tuple teacher → Transformer</title><style>body{font:16px/1.6 system-ui;background:#faf8ef;color:#544c44;max-width:1250px;margin:30px auto;padding:20px}a{color:#197970}img{width:100%}pre{white-space:pre-wrap;background:#eee9df;padding:20px;border-radius:10px}</style><h1>Teach the Transformer the complete n-tuple value</h1><p><a href="index.html">All research</a> · <a href="transformer_teacher_values/index.html">Inspect every teacher lookup for an example board</a> · <a href="transformer_teacher_protocol.json">Reason and experiment protocol</a> · <a href="transformer_teacher_benchmark.json">Measured Mac throughput</a></p>
<p>The frozen published teacher has8 pattern tables and8 views per pattern:64 lookups are added for each afterstate. We supervise that complete value. These are external pretrained weights already downloaded from Hung Guei; they were not trained on this Mac. The teacher value is an estimate, not an exact future score.</p>
<pre>Teacher: T(x) = sum over8patterns and8orientations of the selected table weight
Student: U(x) = μ + σ × mean over8orientations of Transformer(x)
Loss: mean[((U(x) - T(x)/128) / σ)²]
Play: choose the legal move with the largest immediate points/128 + U(afterstate)</pre>
<p>First scalar student:16 tile categories → learned128-dimensional tile vectors + learned positions →4 standard Transformer blocks, each with4 attention heads and a512-wide feed-forward MLP → one scalar. Shared weights average all8 views. Total799,745learned parameters. Later policy/value architectures and their sizes are specified per run below. No tuple lookups, snake features or teacher decisions enter the network input.</p>
<p>First stage: fixed teacher supervision only. Later online TD or student-state relabelling is a separate decision after checking copying quality. Smaller held-out value error does not guarantee correct action rankings or strong play. Gameplay monitoring uses128fixed games; endpoint evaluation uses100other fixed selection games. The reserved final test is untouched.</p>
'''
    page+=f'<p>Data: {len(data["episodes"]):,} fresh complete games, {data["source_transitions"]:,} source moves, {data["labelled_states"]:,} sampled boards and {data["labelled_afterstates"]:,} legal afterstate labels. Three quarters of collection games supply fitting data; the remaining games supply label validation.</p>'
    first=read(base/'transformer_teacher_first_summary.json')
    if first:
        page+='<p><b>First experiment completed:</b> '+escape(first['learning'])+'</p>'
        page+='<p>The next matched comparison retains the same frozen teacher labels, starting model and optimizer state. Both arms use grouped board sampling; only one adds direct supervision of move probabilities. The initial untrained checkpoint was the best gameplay checkpoint in the first run, so its3204 score must not be presented as successful learned imitation.</p>'
    if reference.get('complete'):
        page+=f'<p><b>Teacher reference with the same root-only action calculation:</b> {reference["mean_score"]:,.2f} over100selection games. No future-spawn search, downgrading or value floor. This differs from the teacher\'s stronger search-assisted deployment.</p>'
    comparison=read(base/'transformer_teacher_rank_comparison.json')
    if comparison:
        page+='<p><b>Completed matched objective comparison:</b> '+escape(comparison['learning'])+'</p>'
    coverage=read(base/'transformer_teacher_stage_comparison.json')
    if coverage:
        page+='<p><b>Completed coverage test:</b> '+escape(coverage['learning'])+'</p>'
    policy_comparison=read(base/'transformer_teacher_policy_comparison.json')
    if policy_comparison:
        page+='<p><b>Direct policy/value initialization comparison:</b> '+escape(policy_comparison['learning'])+'</p>'
    capacity=read(base/'transformer_teacher_capacity_comparison.json')
    if capacity:
        page+='<p><b>Completed capacity comparison:</b> '+escape(capacity['learning'])+'</p>'
    fit_result=read(base/'transformer_teacher_policy_fit512_seed0/result.json',{})
    if fit_result.get('completed_fit_goal'):
        page+='<p><b>Can the model fit known examples?</b> The 512-board fitting diagnostic reached 95.9% teacher agreement '
        page+='and KL divergence 0.00143 after 768 updates (31 training seconds). Held-out agreement stayed at58.0%. '
        page+='This demonstrates fitting capacity for a small set, not a gameplay improvement. The deliberately overfit model '
        page+='is excluded from competitive rankings. That check justified the larger-model comparison above.</p>'
    if (base/'transformer_online_protocol.json').exists():
        page+='<p><a href="transformer_online.html"><b>Next objective: learn from the student’s own games with PPO</b></a>. '
        page+='This preserves the pretrained actor, resets the teacher critic, and uses new on-policy rollouts without teacher labels.</p>'
    audit=read(base/'transformer_teacher_diagnostics/rank_replay_audit.json')
    if audit:
        final=audit['decisions'][-1]
        page+='<p><b>Check an actual decision:</b> the action-supervised student’s fixed diagnostic game scored '
        page+=f'{audit["summary"]["replay_score"]:,}. At its final board it chose right, with a 90% chance of immediate game over; '
        page+='left had a 0% immediate game-over probability. The teacher preferred left. '
        page+='Both networks of values can still be too optimistic on unfamiliar boards: these are estimates, not exact future returns. '
        page+='<a href="transformer_teacher_diagnostics/rank_replay_audit.json">All 170 recorded decisions, teacher/student values and exact immediate risks</a>. '
        page+='This one replay is a diagnostic, not an average-score estimate.</p>'
    page+=''.join(sections)+'''<p>Research context: <a href="https://arxiv.org/pdf/2212.11087">Guei reports625,377average points with multi-stage n-tuples and6-ply search</a>. This is not a promised Transformer result. <a href="https://www.jstage.jst.go.jp/article/ipsjjip/27/0/27_340/_article">Kondo and Matsuzaki demonstrate supervised training from strong2048players</a>; reproducing numerical values alone was challenging in that study too.</p></html>'''
    (base/'transformer_teacher.html').write_text(page)


if __name__=='__main__':render()
