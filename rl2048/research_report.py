"""Refresh local experiment charts and the readable result ledger."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path('runs/research')


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        return None


def save_panels(fig, axes, prefix):
    """Separate panels stay readable when the dashboard is narrow."""
    fig.canvas.draw()
    fig.set_layout_engine(None)  # Keep positions fixed while hiding neighbors.
    panels = list(np.asarray(axes).ravel())
    for index, axis in enumerate(panels):
        for other in panels:
            other.set_visible(other is axis)
        box = axis.get_tightbbox(fig.canvas.get_renderer()).expanded(1.04,1.08)
        fig.savefig(ROOT/f'{prefix}_{index}.png',dpi=160,
                    bbox_inches=box.transformed(fig.dpi_scale_trans.inverted()))
    for axis in panels:
        axis.set_visible(True)


def build():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    html = Path(__file__).with_name('research_dashboard.html').read_text()
    if (ROOT/'local_v2_test_record_replay.html').exists():
        html = html.replace('<nav>', '<nav><a href="local_v2_test_record_replay.html">Newest local test record</a>')
    (ROOT/'index.html').write_text(html)
    results=[]
    for path in sorted(ROOT.glob('*/validation.json')):
        data=read_json(path)
        if not data:
            continue
        summaries=[(path.parent.name,data['summary'])] if 'summary' in data else [(path.parent.name+' '+(str(depth).replace('_',' ') if str(depth).startswith('depth') else f'depth {depth}'),v['summary']) for depth,v in data.items()]
        for name,summary in summaries:
            results.append({'name':name,'mean_score':summary['mean_score'],
                            'max_score':summary.get('max_score'), 'games':summary.get('games',summary.get('episodes')),
                            'source':'local','path':str(path.relative_to(ROOT))})
    for path in ROOT.glob('**/search_validation.json'):
        data=read_json(path)
        if data:
            for depth,result in data.items():
                summary=result['summary']
                name=str(path.parent.relative_to(ROOT))+' depth '+depth
                results.append({'name':name,'mean_score':summary['mean_score'],'max_score':summary['max_score'],
                                'games':summary['games'],'source':'published' if 'published' in name else 'local',
                                'path':str(path.relative_to(ROOT))})
    published=read_json(ROOT/'published/checkpoint/validation.json')
    if published:
        for depth,result in published.items():
            summary=result['summary']
            results.append({'name':'published reference depth '+depth,'mean_score':summary['mean_score'],
                            'max_score':summary['max_score'],'games':summary['games'],'source':'published',
                            'path':'published/checkpoint/validation.json'})
    finalists=read_json(ROOT/'local_finalist_validation.json')
    if finalists:
        for name,result in finalists.items():
            summary=result['summary']
            results.append({'name':name,'mean_score':summary['mean_score'],'max_score':summary['max_score'],
                            'games':summary['games'],'source':'local','path':'local_finalist_validation.json'})
    downgrade=read_json(ROOT/'downgrade_validation.json')
    if downgrade:
        for name,result in downgrade.items():
            summary=result['summary']
            results.append({'name':name,'mean_score':summary['mean_score'],'max_score':summary['max_score'],
                            'games':summary['games'],'source':'published' if 'published' in name else 'local',
                            'path':'downgrade_validation.json'})
    results.sort(key=lambda x:x['mean_score'],reverse=True)
    (ROOT/'results.json').write_text(json.dumps(results,indent=2))
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    neural_curves = {'ppo_control','ppo_cpu300','ppo_cnn300','sac_180',
                     'dqn16_exponents','dqn16_corner_snake','dqn16_long','dqn16_wide1024_long'}
    for path in sorted(ROOT.glob('*/progress.csv')):
        try:
            if 'smoke' in path.parent.name:
                continue
            with path.open() as file:
                rows=list(csv.DictReader(file))
            if not rows:
                continue
            if 'validation_score' not in rows[0] and path.parent.name not in neural_curves:
                continue
            x=np.array([float(r.get('transitions') or r.get('training_examples') or 0) for r in rows])/1e6
            elapsed=np.array([float(r['seconds']) for r in rows])
            score_key='validation_score' if 'validation_score' in rows[0] else 'mean_score_100'
            if score_key in rows[0]:
                axis=axes[0,0] if score_key=='validation_score' else axes[0,1]
                axis.plot(x,[float(r[score_key]) for r in rows],label=path.parent.name)
            for key in ('policy_loss','q_loss','value_loss','bellman_loss'):
                if key in rows[0]:
                    axes[1,0].plot(elapsed,[float(r[key]) for r in rows],label=path.parent.name+' '+key,alpha=.7)
                    break
        except (ValueError,KeyError):
            continue # another process may be refreshing this CSV
    for ax,title in [(axes[0,0],'N-tuple validation score'),(axes[0,1],'Neural training score · last 100 games')]:
        ax.set(title=title,xlabel='Transitions in this training phase (millions)',ylabel='Raw score'); ax.legend(fontsize=6)
    axes[1,0].set(title='Diagnostic losses · scales differ by algorithm',xlabel='Training seconds',ylabel='Loss')
    axes[1,0].set_yscale('symlog',linthresh=1); axes[1,0].legend(fontsize=5,ncol=2)
    top=[r for r in results if 'smoke' not in r['name']][:10]
    axes[1,1].barh([r['name'] for r in top][::-1],[r['mean_score'] for r in top][::-1],
                   color=['#b07748' if r['source']=='published' else '#167d8d' for r in top][::-1])
    axes[1,1].set(title='Validation means · published model shown in brown',xlabel='Raw score')
    axes[1,1].tick_params(axis='y',labelsize=7)
    fig.savefig(ROOT/'training_overview.png',dpi=160)
    save_panels(fig,axes,'training_panel'); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for name in sorted(neural_curves):
        path = ROOT/name/'episodes.csv'
        if not path.exists():
            continue
        with path.open() as file:
            rows = list(csv.DictReader(file))
        if len(rows) < 100:
            continue
        # Nonoverlapping windows keep the chart readable; no smoothing across runs.
        windows = [rows[i:i+100] for i in range(0,len(rows)-99,100)]
        x = [int(w[-1]['transitions'])/1e6 for w in windows]
        axes[0].plot(x,[np.mean([int(r['length']) for r in w]) for w in windows],label=name)
        axes[1].plot(x,[np.mean([int(r['max_tile'])>=2048 for r in w]) for w in windows],label=name)
    axes[0].set(title='Episode length · batches of 100 training games',ylabel='Mean legal moves')
    axes[1].set(title='Reaching 2048 · batches of 100 training games',ylabel='Fraction of games',ylim=(0,1))
    for axis in axes:
        axis.set_xlabel('Transitions in this training phase (millions)')
        axis.legend(fontsize=6)
    save_panels(fig,axes,'episode_panel');plt.close(fig)
    dqn_results=read_json(ROOT/'dqn16_comparison.json')
    if dqn_results:
        fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
        for record in dqn_results:
            folder=ROOT/record['name']
            path=folder/'validation_curve.csv'
            if path.exists():
                with path.open() as file: rows=list(csv.DictReader(file))
                axes[0].plot([float(r['transitions'])/1e6 for r in rows],
                             [float(r['validation_score']) for r in rows],label=record['name'].replace('dqn16_',''))
            path=folder/'progress.csv'
            if path.exists():
                with path.open() as file: rows=list(csv.DictReader(file))
                axes[2].plot([float(r['updates']) for r in rows],[float(r['q_loss']) for r in rows],
                             label=record['name'].replace('dqn16_',''))
        axes[0].set(title='Frozen policy: 20 validation games',xlabel='Training transitions (millions)',ylabel='Raw score')
        axes[0].legend(fontsize=6)
        axes[1].barh([r['name'].replace('dqn16_','') for r in dqn_results],
                     [r['mean_score'] for r in dqn_results],color='#167d8d')
        axes[1].set(title='Final policy: 100 validation games',xlabel='Mean raw score')
        axes[2].set(title='TD Huber loss',xlabel='Optimizer updates',ylabel='Loss')
        axes[2].legend(fontsize=6)
        fig.savefig(ROOT/'dqn16_overview.png',dpi=160)
        save_panels(fig,axes,'dqn16_panel');plt.close(fig)
    groups = {
        'Three training seeds · 3M transitions each': {
            'Merge reward': ['dqn16_exponents','dqn16_control_seed1','dqn16_control_seed2'],
            'Corner + snake': ['dqn16_corner_snake','dqn16_winner_seed1','dqn16_winner_seed2'],
            'Dueling + shaping': ['dueling16_seed0','dueling16_seed1','dueling16_seed2'],
            'Width 1,024 + shaping': ['dqn16_wide1024','dqn16_wide1024_seed1','dqn16_wide1024_seed2'],
            'Merge reward, gamma 1': ['dqn16_gamma1','dqn16_gamma1_seed1','dqn16_gamma1_seed2'],
        },
        'Batch size · fixed 3M transitions, seed 0': {
            '1,024 CPU': ['dqn16_batch1024'], '4,096 CPU': ['dqn16_corner_snake'],
            '16,384 MPS': ['dqn16_batch16384'], '65,536 MPS': ['dqn16_batch65536'],
        },
        'Discount · merge reward, fixed 3M transitions': {
            '0.99': ['dqn16_exponents'], '0.999': ['dqn16_gamma999'], '1.0': ['dqn16_gamma1'],
        },
    }
    comparisons = {}
    fig, axes = plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for axis, (title, variants) in zip(axes, groups.items()):
        records = []
        for label, names in variants.items():
            saved = [read_json(ROOT/name/'validation.json') for name in names]
            if not all(saved):
                continue
            scores = [data['summary']['mean_score'] for data in saved]
            records.append({'label':label, 'runs':names, 'scores':scores,
                            'mean':float(np.mean(scores)), 'seed_sd':float(np.std(scores,ddof=1)) if len(scores)>1 else 0.})
        comparisons[title] = records
        axis.barh([r['label'] for r in records], [r['mean'] for r in records],
                  xerr=[r['seed_sd'] for r in records], color='#167d8d', capsize=4)
        axis.invert_yaxis()
        axis.set(title=title, xlabel='Mean raw score · 100 validation games per run')
        axis.tick_params(axis='y', labelsize=8)
    fig.savefig(ROOT/'dqn16_followups.png',dpi=160)
    save_panels(fig,axes,'dqn16_followup_panel'); plt.close(fig)
    (ROOT/'dqn16_followups.json').write_text(json.dumps(comparisons,indent=2))
    planning = read_json(ROOT/'q_planning_comparison.json')
    if planning:
        fig, axis = plt.subplots(figsize=(8,4),layout='constrained')
        labels = list(planning)
        x = np.arange(len(labels))
        axis.barh(x-.18,[planning[k]['greedy']['mean_score'] for k in labels],.36,label='Direct argmax Q',color='#8d9b94')
        axis.barh(x+.18,[planning[k]['planning']['mean_score'] for k in labels],.36,label='One move + exact spawn expectation',color='#167d8d')
        axis.set(yticks=x,yticklabels=[k.replace('dqn16_','') for k in labels],xlabel='Mean raw score · same 100 validation seeds',
                 title='Same frozen weights, different action selection')
        axis.legend(loc='lower right',fontsize=8)
        fig.savefig(ROOT/'q_planning.png',dpi=160);plt.close(fig)
        deeper = read_json(ROOT/'q_planning_depth2_validation.json')
        if deeper and 'dqn16_long' in planning:
            summaries = [planning['dqn16_long']['greedy'],planning['dqn16_long']['planning'],deeper['summary']]
            labels = ['Direct Q','One move','Two moves']
            fig, axes = plt.subplots(1,2,figsize=(10,4),layout='constrained')
            axes[0].bar(labels,[s['mean_score'] for s in summaries],color=['#8d9b94','#52a69c','#167d8d'])
            axes[0].set(title='Same network · increasing search depth',ylabel='Mean raw score')
            x = np.arange(3)
            for offset,tile,color in [(-.18,'2048','#52a69c'),(.18,'4096','#167d8d')]:
                axes[1].bar(x+offset,[s['tile_reaching_rates'][tile] for s in summaries],.36,label=tile,color=color)
            axes[1].set(xticks=x,xticklabels=labels,title='Tile-reaching rates',ylabel='Fraction of 100 validation games',ylim=(0,1))
            axes[1].legend()
            save_panels(fig,axes,'q_depth_panel');plt.close(fig)
    wide_planning = read_json(ROOT/'wide_planning_validation.json')
    narrow_planning = read_json(ROOT/'q_planning_depth2_validation.json')
    wide_direct = read_json(ROOT/'dqn16_wide1024_long/validation.json')
    narrow_direct = read_json(ROOT/'dqn16_long/validation.json')
    if all([wide_planning,narrow_planning,wide_direct,narrow_direct]):
        fig, axis = plt.subplots(figsize=(8,4),layout='constrained')
        x = np.arange(2)
        axis.bar(x-.18,[narrow_direct['summary']['mean_score'],wide_direct['summary']['mean_score']],.36,
                 label='Direct Q argmax',color='#8d9b94')
        axis.bar(x+.18,[narrow_planning['summary']['mean_score'],wide_planning['summary']['mean_score']],.36,
                 label='Two-move planning',color='#167d8d')
        axis.set(xticks=x,xticklabels=['Width 256','Width 1,024'],ylabel='Mean raw score · 100 validation games',
                 title='A stronger direct policy can be a weaker search value')
        axis.legend()
        fig.savefig(ROOT/'width_and_planning.png',dpi=160);plt.close(fig)
    tests = []
    for path in ROOT.glob('*/test.json'):
        data = read_json(path)
        if data and 'summary' in data:
            summary = data['summary']
            tests.append({'name':path.parent.name, 'mean_score':summary['mean_score'],
                          'max_score':summary.get('max_score'), 'games':summary.get('games',summary.get('episodes')),
                          'path':str(path.relative_to(ROOT))})
    tests.sort(key=lambda r:r['mean_score'],reverse=True)
    (ROOT/'heldout_results.json').write_text(json.dumps(tests,indent=2))
    return results


if __name__=='__main__':
    print(json.dumps(build(),indent=2))
