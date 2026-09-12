"""Live report for completed runs and saved checkpoints of the active run."""
import json
from html import escape
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research.scaled_transformer import BASE,write


def render():
    if (BASE.parent/'endgame_tablebase').exists():
        from research.report_endgame import render as render_endgame
        render_endgame()
    if (BASE/'native_policy_protocol.json').exists():
        from research.report_native_policy import render as render_native_policy
        render_native_policy()
    if (BASE/'cnn_filtered_reinforce_protocol.json').exists():
        from research.report_cnn_filtered_reinforce import render as render_filtered
        render_filtered()
    if (BASE/'cnn_reinforce_protocol.json').exists():
        from research.report_cnn_reinforce import render as render_cnn_reinforce
        render_cnn_reinforce()
    if (BASE/'cnn_spawn_safety/protocol.json').exists():
        from research.report_cnn_safety import render as render_safety
        render_safety()
    if (BASE.parent/'tablebase_search/validation100/summary.json').exists():
        from research.report_tablebase import render as render_tablebase
        render_tablebase()
    if (BASE/'cnn_dagger_protocol.json').exists():
        from research.report_cnn_dagger import render as render_cnn
        render_cnn()
    if (BASE/'dagger_protocol.json').exists():
        from research.report_dagger import render as render_dagger
        render_dagger(BASE)
    if (BASE/'reinforce_protocol.json').exists():
        from research.report_reinforce import render as render_reinforce
        render_reinforce(BASE)
    if (BASE/'transformer_teacher_protocol.json').exists():
        from research.report_teacher_transformer import render as render_teacher
        render_teacher(BASE)
    if (BASE/'transformer_online_protocol.json').exists():
        from research.report_teacher_ppo import render as render_online
        render_online(BASE)
    if (BASE/'rotation_comparisons_protocol.json').exists():
        from research.report_rotation_comparisons import render as render_rotations
        render_rotations(BASE)
    manifest=json.loads((BASE/'manifest.json').read_text())
    results=json.loads((BASE/'results.json').read_text())
    status=json.loads((BASE/'status.json').read_text())
    # The full history is in the table/journal. A fifty-entry legend hides the
    # actual curves, so the live plot shows recent runs plus two fixed controls.
    available=[j['id'] for j in manifest['jobs'] if (BASE/j['id']/'curve.json').exists() and 'smoke' not in j['id'] and not j['config'].get('teacher_only')]
    plotted=set(available[-6:]) | {'qr_compare_qr_dqn_score_seed0','plateau_qr_dqn_score_seed0'}
    fig,axes=plt.subplots(1,3,figsize=(16,6))
    fig.subplots_adjust(bottom=.30,top=.90,left=.06,right=.98,wspace=.30)
    colors={name:plt.get_cmap('tab10')(i%10) for i,name in enumerate(j['id'] for j in manifest['jobs'] if j['id'] in plotted)}
    rows=[];evaluated_policies=[];notes=['# Larger Transformer research','',
        'User instruction: continue until explicitly stopped. One hour is an expected return time, not a hard deadline.','',
        'Primary sources: [TQL (2026)](https://arxiv.org/html/2602.01439v1), '
        '[compute-optimal value learning (2025)](https://value-scaling.github.io/), '
        '[SimbaV2 (2025)](https://arxiv.org/abs/2502.15280).','',
        'Implemented adaptation: layer-wise automatic attention entropy control on a standard Transformer with four discrete Q outputs, online Double DQN and replay. '
        'This is not a reproduction of offline TQL, its continuous action tokens, VALUE token, ensemble critics or flow actor. '
        'QR-DQN distributional targets are implemented; SimbaV2 remains a research lead.','',
        'Inputs are simple exponents for the original size study, or tile/max(tile) for the new relative-input controls; no engineered 80 features. '
        'Relative inputs map empty cells to0 and the largest tile to1, deliberately discarding absolute scale. Heads have dimension32. '
        'New reward controls compare points/128, points/pre-move maximum, and ln(1+points/pre-move maximum); reported game scores stay raw. '
        'Strategic reward candidates restore bottom-left plus weighted-snake potential differences at strengths2 and10, with original board inputs. '
        'QR-DQN controls predict51 quantiles/action and act on their mean, compared with scalar Double DQN at matching experience budgets. '
        'Initial size comparisons use AdamW decay0.01, LR0.0001, n=3, gamma0.99, batch512. '
        'Entropy control averages attention over heads before measuring entropy; target starts at 0.8 log(16), lower by0.5 in the last layer. '
        'A separate adaptive temperature is learned per layer; initial alpha0.01 is our experiment setting.','',
        'The same 128 validation games are run throughout training. The separate 100-game suite is also selection data. '
        'Seeds8920000..8920099 remain reserved for a final frozen choice. Single-seed scores are provisional. '
        'Do not compare different amounts of experience as if they were an isolated size experiment.','']
    for job in manifest['jobs']:
        path=BASE/job['id'];curve=[]
        if (path/'curve.json').exists():
            try:curve=json.loads((path/'curve.json').read_text())
            except json.JSONDecodeError:pass
        result=next((r for r in results if r['id']==job['id']),None)
        selection=result.get('result',{}).get('mean_score') if result else None
        if selection is not None and not job['config'].get('diagnostic_only') and not job['config'].get('spawn_safety'):
            evaluated_policies.append((selection,job['id'],'Final checkpoint',f'{job["id"]}/evaluation.json'))
        saved_selection=None
        # Latest evaluation of a monitor-selected checkpoint; never substitute
        # the128-game monitor score for a100-game selection result.
        for file in sorted(path.glob('best_selection*.json'),key=lambda f:f.stat().st_mtime,reverse=True):
            try: saved=json.loads(file.read_text())
            except json.JSONDecodeError: continue
            saved_selection=saved.get('summary',{}).get('mean_score')
            if saved_selection is not None:
                if not job['config'].get('diagnostic_only') and not job['config'].get('spawn_safety'):
                    evaluated_policies.append((saved_selection,job['id'],'Monitoring-selected checkpoint',f'{job["id"]}/{file.name}'))
                break
        phase=result['status'] if result else ('running' if job['id']==status.get('job') and status.get('phase')=='training' else job.get('status','pending'))
        if job['config'].get('diagnostic_only'):phase+=' (diagnostic only)'
        if result and result.get('result',{}).get('stop_reason')=='training_time_budget' and result['result'].get('complete'):
            phase='budget complete'
        if curve and job['id'] in plotted:
            steps=[r['transitions'] for r in curve];scores=[r['mean_score'] for r in curve]
            c=job['config'];family='QR' if c.get('algorithm')=='qr_dqn' else 'Double DQN'
            architecture=f'MLP {c["width"]}' if c['architecture']=='mlp_q' else 'Transformer'
            label=f'{family} {architecture} · n={c.get("n","—")} · γ={c["gamma"]} · seed {job["seed"]}'
            if c.get('algorithm')=='teacher_ppo':
                label=f'PPO Transformer {c["width"]} · '+('teacher actor' if c.get('initial_actor_checkpoint') else 'scratch')+f' · seed {job["seed"]}'
            if c.get('algorithm')=='dagger':
                label=f'Teacher corrections on student boards · seed {job["seed"]}'
                if c['architecture'] == 'pretrained_ml2048':
                    label=('Pretrained CNN · ' + ('student corrections' if c['data_source']=='student'
                                                 else 'fixed-data control') + f' · seed {job["seed"]}')
            if 'dagger_cost_' in job['id']:
                label=('Teacher actions + mistake cost' if c.get('teacher_gap_weight',0) else 'Matched teacher-action control')+f' · seed {job["seed"]}'
            if 'dagger_stage_' in job['id']:
                label=('Half stage-balanced sampling' if c.get('stage_sampling_fraction',0) else 'Matched uniform sampling')+f' · seed {job["seed"]}'
            if c.get('algorithm') in ('reinforce', 'reinforce_loo'):
                family='REINFORCE + baseline' if c['algorithm']=='reinforce_loo' else 'REINFORCE'
                label=f'{family} Transformer {c["width"]} · B={c["episode_batch"]} · T={c.get("action_temperature",1):g} · '+('teacher actor' if c.get('initial_actor_checkpoint') else 'scratch')+f' · seed {job["seed"]}'
                if c['architecture'] == 'pretrained_ml2048':
                    label=f'{family} pretrained CNN · B={c["episode_batch"]} · seed {job["seed"]}'
                    if c.get('spawn_safety'):
                        label += ' · exact spawn-risk filter'
            if c.get('algorithm')=='neural_afterstate':
                model={'afterstate_cnn':'CNN','afterstate_sym_mlp':'D4 mean MLP'}.get(c['architecture'],'MLP')
                label=f'Afterstate {model} {c["width"]} · {c["input_encoding"]} · γ={c["gamma"]} · seed {job["seed"]}'
                if c.get('spawn_target')=='expected':label+=' · exact spawn target'
            if c.get('resume'):label+=' · continued'
            if c.get('resume_dagger'):label+=' · continued'
            if c.get('augment_symmetry'):label+=' · symmetry'
            if c.get('epsilon_steps',0)>500000:label+=' · slower exploration'
            if c.get('priority_alpha') is not None:label+=' · prioritized replay'
            if job['id'].startswith('qr_late_exploration_') or (c.get('resume') and c.get('epsilon_end')==.005):label+=f' · ε={c["epsilon_end"]}'
            color=colors[job['id']]
            axes[0].plot(steps,scores,'o-',ms=3,label=label,color=color)
            axes[1].plot([r['training_seconds'] for r in curve],scores,'o-',ms=3,color=color)
            if 'attention_entropy_fraction' in curve[0]:
                axes[2].plot(steps,[r['attention_entropy_fraction'] for r in curve],'o-',ms=3,color=color)
        meta={}
        if (path/'last/metadata.json').exists():
            try:meta=json.loads((path/'last/metadata.json').read_text())
            except json.JSONDecodeError:pass
        rows.append('<tr>'+''.join(f'<td>{escape(str(x))}</td>' for x in [job['id'],phase,
            f"{meta.get('parameter_count',0):,}" if meta else '—',
            f"{curve[-1]['transitions']:,}" if curve else '—',
            f"{curve[-1]['mean_score']:,.0f}" if curve else '—',
            f"{max(r['mean_score'] for r in curve):,.0f}" if curve else '—',
            f"{selection:,.2f}" if selection is not None else '—',
            f"{saved_selection:,.2f}" if saved_selection is not None else '—'])+'</tr>')
        if result:
            r=result.get('result',{})
            notes += [f"## {job['id']}",f"Testing: {job['question']}",
                f"Status: {result['status']}; completed transitions {r.get('training_transitions')}; final selection-suite mean {r.get('mean_score')}.",
                'Learning: '+(f"Within-run validation changed from {curve[0]['mean_score']:.1f} to {curve[-1]['mean_score']:.1f}; best {max(x['mean_score'] for x in curve):.1f}. " if curve else 'No complete learning curve available. ')+
                'Single-seed evidence; compare matching transition counts and repeat promising configurations before claiming an architecture gain.','']
    for ax in axes:ax.grid(alpha=.2)
    axes[0].set(xlabel='New training transitions',ylabel='128-game mean raw score',title='Learning with experience')
    axes[1].set(xlabel='Training seconds, monitoring excluded',ylabel='128-game mean raw score',title='Learning with compute')
    axes[2].set(xlabel='New training transitions',ylabel='Attention entropy / log(16)',title='Fixed-board attention diagnostic',ylim=(0,1.05))
    if axes[0].lines:
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.01),ncol=2,fontsize=8,frameon=False)
    fig.savefig(BASE/'curves.png',dpi=130);plt.close(fig)
    if (BASE/'supervision.md').exists():
        notes+=['','## Supervision reviews','',(BASE/'supervision.md').read_text()]
    (BASE/'journal.md').write_text('\n'.join(notes))
    symmetry_path=BASE.parent/'overnight_qr_search/afterstate_symmetry_29m/d4_mean.json'
    if symmetry_path.exists():
        symmetry=json.loads(symmetry_path.read_text())['summary']
        if symmetry.get('complete') and symmetry.get('episodes')==100 and not symmetry.get('truncated_episodes'):
            evaluated_policies.append((symmetry['mean_score'],'MLP: eight rotations/reflections, frozen 29.36M weights',
                'Eight-view value ensemble plus exact root slides; higher inference cost',
                '../overnight_qr_search/afterstate_symmetry_29m/d4_mean.json'))
    cnn_symmetry_path=BASE.parent/'overnight_qr_search/afterstate_cnn_long_symmetry_mps/d4_mean.json'
    if cnn_symmetry_path.exists():
        cnn_symmetry=json.loads(cnn_symmetry_path.read_text())['summary']
        if cnn_symmetry.get('complete') and cnn_symmetry.get('episodes')==100 and not cnn_symmetry.get('truncated_episodes'):
            evaluated_policies.append((cnn_symmetry['mean_score'],'CNN: eight rotations/reflections, frozen12.70M weights, MPS',
                'Eight-view value ensemble plus exact root slides; higher inference cost',
                '../overnight_qr_search/afterstate_cnn_long_symmetry_mps/d4_mean.json'))
    highlight=''
    if evaluated_policies:
        score,job_id,checkpoint_kind,url=max(evaluated_policies,key=lambda r:r[0])
        highlight=(f'<div class="highlight"><strong>Highest evaluated policy without future-spawn search: '
            f'{score:,.2f} points</strong><br>'
            f'{escape(job_id)} · {checkpoint_kind} · mean over 100 selection games · '
            f'<a href="{escape(url,quote=True)}">Saved evaluation</a></div>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="60">
<title>2048 · Larger Transformer research</title><style>body{font:16px/1.6 system-ui;max-width:1400px;margin:35px auto;padding:0 24px;background:#faf8ef;color:#544c44}img{width:100%}a{color:#197970}td,th{text-align:left;padding:10px;border-bottom:1px solid #d9d2c5}table{width:100%;font-size:14px}.scroll{overflow:auto}.highlight{background:#e1efe8;border-left:5px solid #197970;padding:18px;margin:20px 0;overflow-wrap:anywhere}.highlight strong{font-size:24px}pre{white-space:pre-wrap;background:#eee9df;padding:20px;border-radius:10px}</style>
<h1>Larger Transformers learning 2048</h1><p>One GPU training queue, CPU game simulation, frozen-policy evaluation on the same 128 games. The study continues until you request a stop. This page refreshes every minute; saved curves update at training checkpoints and supervisor reviews.</p>
'''+highlight+'''
<p><b>Research:</b> <a href="https://arxiv.org/abs/2602.01439">TQL (2026)</a> motivates adaptive attention entropy control; <a href="https://value-scaling.github.io/">value-learning scaling studies (2025)</a> motivate measuring model size, batch size and compute together. This is an online, discrete-action adaptation, not a reproduction of those papers.</p>
<p>Original Q-network runs: tile inputs → learned scalar projections and positions → standard attention + feed-forward Transformer blocks → four Q-values. Original runs use exponents divided by16; relative-input controls use actual tile / largest tile, with empty cells0. No engineered 80-feature input, teacher copying or search. Input controls keep the original points/128 reward. Further reward controls test points/pre-move maximum and ln(1+points/pre-move maximum). Reported game scores always use actual merge points. Single-seed rankings are exploratory.</p>
<p><b>Broader formulation tests:</b> the new afterstate MLPs predict one future value per board after a slide. Action selection enumerates the four exact root slides and adds their actual merge points; it does not search future random spawns. Simple tile inputs and learned embeddings are compared in separate bounded runs. This changes the whole prediction recipe, with different parameter counts and one-step scalar MSE targets. <a href="afterstate_mlp_protocol.md">Equations, data flow, reason and budget</a>.</p>
<p><a href="status.json">Process status</a> · <a href="journal.md">Questions, results and learnings</a> · <a href="benchmark.json">Mac size/batch benchmark</a> · <a href="../policy_curves/index.html">Earlier every-update diagnostics</a></p>
<p><b>Two separate game sets:</b> the latest and best monitoring means use 128 fixed games throughout training. The final evaluation uses 100 different fixed games, so its mean can differ for the same model. Both are selection data; the reserved final test has not been used. The final evaluation column refers to the last checkpoint, not necessarily the best saved checkpoint. QR-DQN predicts 51 quantiles per action and chooses using their mean.</p>
<p>“Saved best evaluation” is the latest100-game evaluation of a checkpoint chosen using the128-game monitor. A dash means it has not been evaluated separately. These checkpoints can outperform the final model; both remain distinguished below.</p>
<div class="scroll"><table><tr><th>Experiment</th><th>Status</th><th>Parameters</th><th>New transitions</th><th>Latest monitoring<br>128 games</th><th>Best monitoring<br>128 games</th><th>Final evaluation<br>100 games</th><th>Saved best evaluation<br>100 games</th></tr>'''+''.join(rows)+'''</table></div><p>The plot focuses on the six most recent runs with curves, plus the original QR control and its longer continuation. The full experiment history remains in the table and journal. A continued run starts from previously trained weights: equal new transitions do not mean equal lifetime training experience.</p><img src="curves.png" alt="Recent policy scores versus data and compute, and attention entropy; full history in table"><pre>'''+escape('\n'.join(notes))+'</pre></html>'
    if (BASE/'dagger_protocol.json').exists():
        page=page.replace('<h1>Larger Transformers learning 2048</h1>',
            '<h1>Larger Transformers learning 2048</h1><p><b>New comparison:</b> <a href="dagger.html">Teacher corrections on student boards versus fixed-data imitation</a></p>')
    if (BASE/'rotation_comparisons_protocol.json').exists():
        page=page.replace('<h1>Larger Transformers learning 2048</h1>',
            '<h1>Larger Transformers learning 2048</h1><p><b>Current comparisons:</b> <a href="rotation_comparisons.html">Replay size, smaller symmetric MLPs, and n-tuple teacher initialization</a></p>')
    if (BASE/'transformer_teacher_protocol.json').exists():
        page=page.replace('<h1>Larger Transformers learning 2048</h1>',
            '<h1>Larger Transformers learning 2048</h1><p><b>New main experiment:</b> <a href="transformer_teacher.html">Full n-tuple value supervision for a standard Transformer</a></p>')
    if (BASE/'reinforce_protocol.json').exists():
        page=page.replace('<h1>Larger Transformers learning 2048</h1>',
            '<h1>Larger Transformers learning 2048</h1><p><b>Latest requested test:</b> <a href="reinforce.html">Plain REINFORCE: complete games, no critic</a> · <a href="transformer_online.html">PPO comparison</a></p>')
    if (BASE/'cnn_dagger_protocol.json').exists():
        page=page.replace('<h1>Larger Transformers learning 2048</h1>',
            '<h1>Larger Transformers learning 2048</h1><p><b>Current research:</b> <a href="native_policy.html">Learning the stronger search player’s moves</a> · <a href="cnn_filtered_reinforce.html">Learning with the safety policy</a> · <a href="cnn_spawn_safety/index.html">Frozen CNN with immediate-risk check</a> · <a href="../tablebase_search/index.html">Million-point search games and GPT comparison</a></p>')
    if (BASE/'cnn_reinforce_protocol.json').exists():
        page=page.replace('<h1>Larger Transformers learning 2048</h1>',
            '<h1>Larger Transformers learning 2048</h1><p><b>Actual own-game learning:</b> <a href="cnn_reinforce.html">Pretrained CNN with REINFORCE and a return baseline</a></p>')
    if (BASE/'cnn_filtered_reinforce_protocol.json').exists():
        page=page.replace('<h1>Larger Transformers learning 2048</h1>',
            '<h1>Larger Transformers learning 2048</h1><p><b>New controlled test:</b> <a href="cnn_filtered_reinforce.html">Can training improve the already useful safety policy?</a></p>')
    latest_search=BASE.parent/'overnight_qr_search/afterstate_symmetry_49m_two_move/evaluation.json'
    selected_search=BASE.parent/'overnight_qr_search/afterstate_best53m_two_move/evaluation/evaluation.json'
    selected_complete=False
    if selected_search.exists():
        selected=json.loads(selected_search.read_text())
        selected_complete=selected.get('complete') and len(selected.get('episodes',[]))==100
    if selected_complete:
        latest_search=selected_search
    if latest_search.exists():
        planned=json.loads(latest_search.read_text())
        if planned.get('complete') and len(planned['episodes'])==100 and all(e['complete'] and not e.get('truncated') for e in planned['episodes']):
            replay_root=('afterstate_best53m_two_move' if selected_complete else 'afterstate_symmetry_49m_two_move')
            context=('Selected 53.18M symmetric MLP; the gain over the same CPU planner\'s 60,724 reference remains uncertain. '
                     if selected_complete else 'Longer-trained symmetric MLP. ')
            page=page.replace('<p><b>Research:</b>',
                f'<div class="highlight"><strong>With two-move planning: {planned["mean_score"]:,.2f} points</strong><br>'
                +context+'Mean of 100 selection games, CPU. This includes search and is separate from the direct-policy result above. '
                f'<a href="../overnight_qr_search/{replay_root}/comparison.json">Same-planner comparison</a> · '
                f'<a href="../overnight_qr_search/{replay_root}/audit/replay.html">Watch</a> · '
                f'<a href="../overnight_qr_search/{replay_root}/audit/index.html">Decision audit</a></div><p><b>Research:</b>',1)
    if (BASE/'td_horizon_study.json').exists():
        from research.qr_td_horizon import render as render_horizons
        render_horizons(BASE)
        page=page.replace('<h1>Larger Transformers learning 2048</h1>',
            '<h1>Larger Transformers learning 2048</h1><p><a href="td_horizon.html">New comparison: one-, three-, and five-step TD with QR-DQN</a></p>')
    if (BASE.parent/'overnight_qr_search/direct.json').exists():
        from research.report_overnight_qr_search import render as render_search
        render_search()
        page=page.replace('<h1>Larger Transformers learning 2048</h1>',
            '<h1>Larger Transformers learning 2048</h1><p><a href="../overnight_qr_search/index.html">Overnight: frozen QR-DQN with planning</a></p>')
    comparison_file = BASE / 'qr_scalar_three_seed_comparison.json'
    if comparison_file.exists():
        comparison = json.loads(comparison_file.read_text())
        qr, scalar = comparison['qr_dqn'], comparison['double_dqn']
        summary = (f'<p><b>Completed three-seed algorithm comparison:</b> QR-DQN averaged '
            f'{qr["mean"]:,.0f} points (training-seed SD {qr["sample_standard_deviation"]:,.0f}); '
            f'scalar Double DQN averaged {scalar["mean"]:,.0f} (SD {scalar["sample_standard_deviation"]:,.0f}). '
            'Each run used 4.19 million transitions and the same 100 evaluation games. These are final-checkpoint scores. '
            'QR was higher for each of the three paired training seeds, but has a larger output head; this is not a parameter-matched comparison. '
            '<a href="qr_scalar_three_seed_comparison.json">Full results and limits</a></p>')
        page = page.replace('<div class="scroll">', summary + '<div class="scroll">', 1)
    gamma_file=BASE/'gamma_three_seed_comparison.json'
    if gamma_file.exists():
        comparison=json.loads(gamma_file.read_text())
        ordinary,undiscounted=comparison['groups']['0.99'],comparison['groups']['1.0']
        summary=(f'<p><b>Completed discount comparison:</b> gamma 1 averaged {undiscounted["mean"]:,.0f} '
            f'points across three training seeds (SD {undiscounted["sample_standard_deviation"]:,.0f}), '
            f'versus gamma 0.99 at {ordinary["mean"]:,.0f} (SD {ordinary["sample_standard_deviation"]:,.0f}). '
            'Same 4.19M transitions per run. Gamma 1 was higher in two of three pairs; the largest gain came from seed 1. '
            '<a href="gamma_three_seed_comparison.json">Paired results and limits</a></p>')
        page=page.replace('<div class="scroll">',summary+'<div class="scroll">',1)
    exploration=BASE/'afterstate_symmetry_exploration_comparison.json'
    if exploration.exists():
        c=json.loads(exploration.read_text());control,low=c['records']
        summary=(f'<p><b>Less random exploration in a mature symmetric MLP:</b> final100-game mean '
            f'{control["final100_mean"]:,.0f} with5% random legal actions versus {low["final100_mean"]:,.0f} with0.5%. '
            'Both start from the same33.55M checkpoint and receive4.19M new moves. '
            f'The monitor-selected control scores {control["selected100_mean"]:,.0f}, so its gap to the low-exploration model is smaller. '
            f'Training4096 coverage rises from {control["training_4096"]:.1%} to {low["training_4096"]:.1%}. '
            'This supports testing less disruptive late exploration, with only one shared-parent training lineage so far. '
            '<a href="afterstate_symmetry_exploration_comparison.json">Paired results and limits</a></p>')
        page=page.replace('<div class="scroll">',summary+'<div class="scroll">',1)
    later_exploration=BASE/'afterstate_49m_exploration_comparison.json'
    if later_exploration.exists():
        c=json.loads(later_exploration.read_text());control,low=c['records']
        summary=(f'<p><b>The later-stage exploration check is mixed:</b> from the49.51M model, '
            f'final100means are {control["final100_mean"]:,.0f} at5% random actions and {low["final100_mean"]:,.0f} at0.5%. '
            f'The monitor-selected checkpoints instead score {control["selected100_mean"]:,.0f} and {low["selected100_mean"]:,.0f}. '
            'The earlier final-score advantage did not repeat. Preserve the stronger selected checkpoint and test planning separately; '
            'do not infer a universal exploration winner or plateau. '
            '<a href="afterstate_49m_exploration_comparison.json">Full result</a></p>')
        page=page.replace('<div class="scroll">',summary+'<div class="scroll">',1)
    mps_reference=BASE.parent/'overnight_qr_search/afterstate_symmetry_49m_mps_depth2_confirmation/evaluation.json'
    if mps_reference.exists():
        r=json.loads(mps_reference.read_text())
        if r.get('complete'):
            summary=(f'<p><b>Same frozen two-move model on MPS:</b> {r["mean_score"]:,.0f} over100selection games. '
                'CPU scored60,724. No weights changed: small numerical differences in tied actions changed seeded trajectories. '
                'This shallow GPU evaluation took440summed seconds versus406onCPU; the observed GPU speed benefit was in deeper three-move planning. '
                '<a href="../overnight_qr_search/afterstate_symmetry_49m_mps_depth2_confirmation/evaluation.json">MPS evaluation</a></p>')
            page=page.replace('<div class="scroll">',summary+'<div class="scroll">',1)
    symmetry_training=BASE/'afterstate_symmetry_finetune_comparison.json'
    if symmetry_training.exists():
        c=json.loads(symmetry_training.read_text())
        training=c['final_comparisons'][1]
        summary=(f'<p><b>Symmetry during training versus only at evaluation:</b> after the same 4.19M additional moves, '
            f'the ordinary MLP with eight-view averaging scored {training["means"][0]:,.0f}, '
            f'versus {training["means"][1]:,.0f} for the model trained with that averaging. '
            'Both are fresh CPU evaluations on the same 100 games. The extra training gain is inconclusive '
            '(paired game interval crosses zero), although both improve over ordinary one-view inference. '
            'The trained model scored 32,663 on MPS; tiny rounding differences can choose different moves '
            'when symmetric actions tie, so device and training comparisons are kept separate. '
            '<a href="afterstate_symmetry_finetune_comparison.json">All controls and uncertainty</a> · '
            '<a href="audits/afterstate_symmetry_trained_33551360/replay.html">Fixed replay</a> · '
            '<a href="../overnight_qr_search/afterstate_symmetry_33m_trained/first_device_divergence.json">First CPU/MPS divergence</a></p>')
        page=page.replace('<div class="scroll">',summary+'<div class="scroll">',1)
    cnn_long=BASE/'afterstate_cnn_long_comparison.json'
    if cnn_long.exists():
        c=json.loads(cnn_long.read_text())
        summary=(f'<p><b>Longer CNN training:</b> final MPS 100-game score rose from {c["means"][0]:,.0f} '
            f'to {c["means"][1]:,.0f}, after 8.51M additional moves in 30 training minutes. '
            'The run reached 12.70M lifetime moves; budget expiry is not a proven plateau. '
            '<a href="afterstate_cnn_long_comparison.json">Paired result and limits</a></p>')
        page=page.replace('<div class="scroll">',summary+'<div class="scroll">',1)
    cnn_average=BASE.parent/'overnight_qr_search/afterstate_cnn_long_symmetry_mps/comparison.json'
    if cnn_average.exists():
        c=json.loads(cnn_average.read_text())
        summary=(f'<p><b>Eight views also help the frozen CNN:</b> MPS 100-game mean {c["means"][1]:,.0f} '
            f'versus {c["means"][0]:,.0f}, with identical weights and no future-spawn search. '
            'It reached 2048 in78% and4096 in11%. The verified completed GPU control was reused. '
            'The earlier CPU attempt finished only16/100games in180seconds and remains unranked; '
            'MPS completed the eight-view evaluation in31.45seconds. '
            '<a href="../overnight_qr_search/afterstate_cnn_long_symmetry_mps/comparison.json">Result and compute limits</a> · '
            '<a href="../overnight_qr_search/afterstate_cnn_long_symmetry_mps/replay_d4_mean.html">Fixed replay</a></p>')
        page=page.replace('<div class="scroll">',summary+'<div class="scroll">',1)
    exploration_file=BASE/'late_exploration_comparison.json'
    if exploration_file.exists():
        comparison=json.loads(exploration_file.read_text())
        control=comparison['qr_late_exploration_control005_seed0']['final_evaluation']
        low=comparison['qr_late_exploration_low0005_seed0']['final_evaluation']
        summary=(f'<p><b>Completed late-exploration comparison:</b> reducing random actions from 5% to 0.5% '
            f'gave a final mean of {low["mean_score"]:,.0f} versus {control["mean_score"]:,.0f} points. '
            'Both continued the same saved policy for 4.19M moves. The lower-exploration policy reached '
            '4096 in 5 of 100 selection games; the control reached none. This is one matched starting policy, '
            'not yet a result across independent training seeds. '
            '<a href="late_exploration_comparison.json">Results and training-data coverage</a> · '
            '<a href="../overnight_qr_search/audits/qr_low_exploration_38273024/replay.html">Fixed-seed replay</a> · '
            '<a href="../overnight_qr_search/audits/qr_low_exploration_38273024/index.html">Decision audit</a></p>')
        page=page.replace('<div class="scroll">',summary+'<div class="scroll">',1)
    afterstate_file=BASE/'afterstate_three_seed_comparison.json'
    if afterstate_file.exists():
        comparison=json.loads(afterstate_file.read_text())
        summary=(f'<p><b>Afterstate MLP seed replication:</b> the learned-embedding model averaged '
            f'{comparison["mean"]:,.0f} points across three training seeds (SD {comparison["sample_standard_deviation"]:,.0f}) '
            'after 4.19M transitions each. This model estimates one future value after each exact slide; '
            'it has different targets, inputs and parameter count from the QR Transformer. '
            'A follow-up replaces one sampled spawn label with an exact average over all possible spawns during learning; '
            'its acting rule stays unchanged. '
            '<a href="afterstate_three_seed_comparison.json">Seed results</a> · '
            '<a href="afterstate_mlp_protocol.md">Equations and expected-target experiment</a></p>')
        page=page.replace('<div class="scroll">',summary+'<div class="scroll">',1)
    afterstate_long=BASE/'afterstate_long_comparison.json'
    if afterstate_long.exists():
        comparison=json.loads(afterstate_long.read_text())
        summary=(f'<p><b>Longer embedding afterstate training:</b> final 100-game mean '
            f'{comparison["long_final_mean"]:,.0f}, versus {comparison["early_final_mean"]:,.0f} '
            'at its earlier 4.19M checkpoint. The continuation reached 29.36M lifetime moves '
            'and its 30-minute training cap; this is not a claimed plateau. '
            'A CNN with learned local filters is queued as a separate architecture screen. '
            '<a href="afterstate_long_comparison.json">Paired results</a> · '
            '<a href="audits/afterstate_embedding_29357056/replay.html">Fixed-seed replay</a> · '
            '<a href="audits/afterstate_embedding_29357056/index.html">Decision audit</a></p>')
        page=page.replace('<div class="scroll">',summary+'<div class="scroll">',1)
    expected_file=BASE/'afterstate_expected_comparison.json'
    if expected_file.exists():
        c=json.loads(expected_file.read_text())
        summary=(f'<p><b>Exact spawn targets:</b> {c["means"][1]:,.0f} versus {c["means"][0]:,.0f} '
            'points on 100 games at the same 4.19M real moves. The small paired gain is inconclusive, '
            'while observed training took about 3.55 times as long. Preserved without extending this change now. '
            '<a href="afterstate_expected_comparison.json">Result and uncertainty</a></p>')
        page=page.replace('<div class="scroll">',summary+'<div class="scroll">',1)
    cnn_file=BASE/'afterstate_cnn_comparison.json'
    cnn_seeds=BASE/'afterstate_cnn_three_seed_comparison.json'
    if cnn_seeds.exists():
        c=json.loads(cnn_seeds.read_text())
        summary=(f'<p><b>Completed three-seed CNN comparison:</b> average {c["cnn_mean"]:,.0f} '
            f'versus MLP {c["mlp_mean"]:,.0f}, at 4.19M real moves per training seed. '
            'The CNN won all three pairs, with 5.54% more parameters and longer training time. '
            'Its learned local filters look useful at matched experience; equal-compute superiority is unproven. '
            'A 30-minute continuation of seed0 is queued after the matched symmetry experiment. '
            '<a href="afterstate_cnn_three_seed_comparison.json">All seed results and limits</a></p>')
        page=page.replace('<div class="scroll">',summary+'<div class="scroll">',1)
    elif cnn_file.exists():
        c=json.loads(cnn_file.read_text())
        summary=(f'<p><b>Convolutional afterstate screen:</b> {c["means"][1]:,.0f} versus {c["means"][0]:,.0f} '
            'points for the MLP at the same 4.19M moves. The CNN learns local patterns with shared filters. '
            'It took substantially more compute; two independent training seeds are queued before scaling it. '
            '<a href="afterstate_cnn_comparison.json">Matched result and limits</a></p>')
        page=page.replace('<div class="scroll">',summary+'<div class="scroll">',1)
    if symmetry_path.exists():
        summary=('<p><b>Eight views of the same frozen MLP:</b> averaging afterstate values over rotations and reflections '
            'raised the 100-game CPU mean from 19,178 to 26,590, with 66% reaching 2048. '
            'No weights changed and no future spawns were searched; inference uses eight views per afterstate. '
            'A matched fine-tuning experiment will test sharing values this way during learning too. '
            '<a href="../overnight_qr_search/afterstate_symmetry_29m/comparison.json">Comparison</a> · '
            '<a href="../overnight_qr_search/afterstate_symmetry_29m/replay_d4_mean.html">Fixed replay</a></p>')
        page=page.replace('<div class="scroll">',summary+'<div class="scroll">',1)
    (BASE/'index.html').write_text(page)
    root=BASE.parent/'index.html'
    if root.exists():
        old=root.read_text()
        if 'scaled_transformer/index.html' not in old:
            old=old.replace('<nav>','<nav><a href="scaled_transformer/index.html">Live larger Transformer training</a><a href="policy_curves/index.html">128-game policy learning curves</a>',1)
            root.write_text(old)


if __name__=='__main__':render()
