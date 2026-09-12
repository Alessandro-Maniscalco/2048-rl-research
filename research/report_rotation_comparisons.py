"""Focused, automatically refreshed report for the user-approved comparisons."""
import json
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]/'runs/research/scaled_transformer'


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def paired(first, second):
    a, b = first['episodes'], second['episodes']
    if len(a) != 100 or len(b) != 100:
        raise ValueError('Only completed 100-game endpoints can be compared')
    if [x['seed'] for x in a] != [x['seed'] for x in b]:
        raise ValueError('Paired evaluation requires identical seeds')
    delta = np.array([y['score']-x['score'] for x,y in zip(a,b)])
    rng = np.random.default_rng(712)
    low, high = np.quantile(delta[rng.integers(0,100,(5000,100))].mean(1), [.025,.975])
    return dict(mean_difference=float(delta.mean()), paired_game_interval=[float(low),float(high)],
                wins=int((delta>0).sum()), games=100)


def render(base=BASE):
    protocol = read(base/'rotation_comparisons_protocol.json')
    if not protocol:
        return
    manifest = read(base/'manifest.json')
    jobs = {j['id']:j for j in manifest['jobs']}
    status = read(base/'status.json', {})
    results = {r['id']:r for r in read(base/'results.json', [])}
    groups = [('Replay capacity', [protocol['buffer']['control']]+protocol['buffer']['variants'],
        'Same frozen 33.55M parent, collection seed and 4.19M new moves. Only capacity changes; batch512 and one update per128 new moves stay fixed.'),
        ('MLP size', protocol['size']['jobs'],
        'Every model starts from scratch. Eight views, learned embeddings and two hidden layers are shared. Compare equal experience and measured runtime.'),
        ('Teacher initialization', [protocol['teacher']['control']]+protocol['teacher'].get('jobs', []),
        'Same width256 online learner and new-game budget. The teacher arm first fits frozen n-tuple values, then uses only its own online TD. Teacher data and fitting cost are extra, reported separately.')]
    if protocol.get('teacher_objective'):
        objective=protocol['teacher_objective']
        groups.append(('What the teacher should teach', [objective['control']]+objective['variants'],
            'Both arms sample the same whole boards and legal alternatives. One fits absolute values only; the other also penalizes errors in relative action values. The subsequent online TD recipe is identical. This controls the sampling change separately from the first flat-regression teacher test.'))
    if protocol.get('replay_reuse'):
        reuse=protocol['replay_reuse']
        groups.append(('Replay reuse', [reuse['control']]+reuse['variants'],
            'Same mature starting model, one-million buffer and4.19M new moves. Compare one versus two512-sample gradient updates after each128newmoves. More optimizer and target updates are intentional and costed; this is separate from buffer capacity.'))
    if protocol.get('learning_rate'):
        rate=protocol['learning_rate']
        groups.append(('Learning rate', [rate['control']]+rate['variants'],
            'Same mature starting policy, target network, optimizer moments, collection seed and 4.19M new moves. Compare learning rates 0.0001 and 0.00003, with one update per collection. The completed replay-reuse control is reused exactly.'))
    if protocol.get('long_size'):
        groups.append(('Longer model-size training', protocol['long_size']['jobs'],
            'Continue each width from its own scratch-seed-0 checkpoint at 4.19M moves. Give both 16.78M additional moves, reaching 20.97M lifetime moves, with the same training recipe and new collection seed. This checks whether the smaller model remains competitive at a longer budget.'))
    sections, analysis = [], {}
    for group_index,(title, identifiers, reason) in enumerate(groups):
        rows, evaluations, curves = [], {}, []
        for identifier in identifiers:
            path = base/identifier; job = jobs[identifier]; c=job['config']
            label = (f"{c['capacity']:,} transitions" if group_index==0 else
                     f"Width {c['width']}" if group_index==1 else
                     'Teacher warm start' if c.get('teacher_pretrain') else 'From scratch')
            if group_index==3:
                label=('Values + action gaps' if c['teacher_pretrain']['gap_weight'] else 'Values only; grouped control')
            if title=='Replay reuse':
                label=f"{c['updates_per_collection']} updates per128moves"
            if title=='Learning rate':
                label=f"Learning rate {c['lr']:.5f}"
            if title=='Longer model-size training':
                label=f"Width {c['width']}"
            curve = read(path/'curve.json', [])
            if curve: curves.append((label, curve, read(path/'teacher_pretrain.json', {})))
            evaluation = read(path/'evaluation.json')
            valid = (evaluation and evaluation['summary'].get('complete')
                     and evaluation['summary'].get('episodes')==100
                     and not evaluation['summary'].get('truncated_episodes'))
            if valid: evaluations[identifier] = evaluation
            state = results.get(identifier,{}).get('status', 'running' if status.get('job')==identifier else 'queued')
            endpoint = f"{evaluation['summary']['mean_score']:,.0f}" if valid else 'Pending'
            selected = read(path/'best_selection.json', {}).get('summary', {})
            selected_score = (f"{selected['mean_score']:,.0f}" if selected.get('complete')
                              and selected.get('episodes')==100 else 'Pending')
            meta = read(path/'result.json', {})
            pretrain = read(path/'teacher_pretrain.json', {})
            seconds = meta.get('training_seconds')
            timing = f'{seconds:,.1f}s' if seconds is not None else 'Pending'
            if pretrain: timing += f" + {pretrain['seconds']:.1f}s teacher fit"
            monitoring = f"{curve[-1]['mean_score']:,.0f}" if curve else 'Pending'
            transitions = f"{curve[-1]['transitions']:,}" if curve else '0'
            links = f'<a href="{identifier}/config.json">Config</a>' if (path/'config.json').exists() else ''
            if (path/'replay_best.html').exists(): links += f' · <a href="{identifier}/replay_best.html">Replay</a>'
            rows.append(f'<tr><td>{escape(label)}</td><td>{escape(state)}</td><td>{transitions}</td><td>{monitoring}</td><td>{endpoint}</td><td>{selected_score}</td><td>{timing}</td><td>{links}</td></tr>')
        figure = ''
        if curves:
            fig, axes = plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
            for label, curve, pretrain in curves:
                score = [r['mean_score'] for r in curve]
                axes[0].plot([r['transitions'] for r in curve],score,label=label)
                axes[1].plot([r['training_seconds']+pretrain.get('seconds',0) for r in curve],score,label=label)
            for ax in axes:
                ax.set_ylabel('Mean raw score · 128 fixed games'); ax.grid(alpha=.2); ax.legend()
            axes[0].set_xlabel('New online environment transitions')
            axes[1].set_xlabel('Training seconds + teacher fit (evaluation excluded)')
            name=f'rotation_comparison_{group_index}.png'
            fig.savefig(base/name,dpi=130); plt.close(fig)
            figure=f'<img src="{name}" alt="{escape(title)}: fixed-game scores against experience and compute">'
        conclusions=[]
        control = identifiers[0]
        for identifier in identifiers[1:]:
            if control not in evaluations or identifier not in evaluations: continue
            result = paired(evaluations[control], evaluations[identifier])
            analysis[identifier] = result | dict(control=control)
            low, high = result['paired_game_interval']
            conclusions.append(f"{identifier}: mean difference {result['mean_difference']:+,.0f} points against control; paired-game 95% bootstrap interval [{low:+,.0f}, {high:+,.0f}], {result['wins']}/100 wins. "
                + ('Direction remains uncertain in this game sample.' if low<=0<=high else 'Replicate across training seeds before treating this as a robust improvement.'))
        learning = '<p><b>Learning so far:</b> '+escape(' '.join(conclusions) if conclusions else 'The matched endpoints are not complete yet; no winner is claimed.')+'</p>'
        if group_index==2 and (base/'rotation_teacher_seed0_summary.json').exists():
            teacher=read(base/'rotation_teacher_seed0_summary.json')
            learning+='<p>'+escape(teacher['learning'])+'</p>'
        if title=='What the teacher should teach' and (base/'rotation_teacher_objective_summary.json').exists():
            teacher=read(base/'rotation_teacher_objective_summary.json')
            learning+='<p>'+escape(teacher['learning'])+'</p>'
        if title=='MLP size' and (base/'rotation_size_three_seed_summary.json').exists():
            sizes=read(base/'rotation_size_three_seed_summary.json')
            learning+='<p>'+escape(sizes['learning'])+'</p>'
        sections.append(f'<h2>{escape(title)}</h2><p>{escape(reason)}</p><div class="scroll"><table><tr><th>Condition</th><th>Status</th><th>New moves</th><th>Latest128 mean</th><th>Final100 mean</th><th>Monitor-selected100 mean</th><th>Training time</th><th>Files</th></tr>'+''.join(rows)+'</table></div>'+figure+learning)
    if protocol['size'].get('replication_jobs'):
        rows=[]
        for width in (256,512):
            scores=[];cells=[]
            for seed in (0,1,2):
                result=read(base/f'afterstate_size_w{width}_scratch_seed{seed}'/'result.json',{})
                valid=result.get('complete') and result.get('training_transitions')==4194304
                cells.append(f'{result["mean_score"]:,.0f}' if valid else 'Pending')
                if valid:scores.append(result['mean_score'])
            aggregate=(f'{np.mean(scores):,.0f} ± {np.std(scores,ddof=1):,.0f}' if len(scores)==3 else 'Pending all3seeds')
            rows.append(f'<tr><td>{width}</td>'+''.join(f'<td>{cell}</td>' for cell in cells)+f'<td>{aggregate}</td></tr>')
        sections.append('<h2>Independent training seeds for the size comparison</h2><p>Each fresh run uses4.19M moves. These replications measure training-seed variation, unlike paired intervals across evaluation games. The mean and sample standard deviation appear only after all three seeds finish.</p><table><tr><th>Width</th><th>Seed0</th><th>Seed1</th><th>Seed2</th><th>Mean ± training-seed SD</th></tr>'+''.join(rows)+'</table>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="60"><title>2048 · Rotation comparisons</title><style>body{font:16px/1.6 system-ui;background:#faf8ef;color:#544c44;max-width:1200px;margin:30px auto;padding:0 20px}a{color:#197970}img{width:100%}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:10px;border-bottom:1px solid #d9d2c5}.scroll{overflow:auto}</style><h1>Keep rotations: compare memory, model size and a teacher</h1><p><a href="index.html">All research</a> · <a href="rotation_comparisons_protocol.json">Protocol and decisions</a></p><p>All scores use ordinary game points. Eight-view averaging stays enabled. Monitoring uses128 fixed games; final endpoints use100 different fixed selection games. These are tuning sets. The reserved final test remains untouched. Only complete endpoints are compared; the paired intervals do not measure training-seed uncertainty.</p>'''+''.join(sections)+'</html>'
    (base/'rotation_comparisons.html').write_text(page)
    (base/'rotation_comparisons_results.json').write_text(json.dumps(analysis,indent=2))


if __name__=='__main__': render()
