"""Compare horizons using the same simulated policy and the same trajectories."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from research.afterstate_teacher import digest, write

BASE=Path('runs/research/endgame_tablebase')
HORIZONS=(8,32,128,512)


def analyze(folder):
    summary=json.loads((folder/'summary.json').read_text())
    assert summary['complete'] and all(json.loads((folder/'verified.json').read_text()).values())
    protocol=json.loads((folder/'protocol.json').read_text())
    outputs=[]
    for case in protocol['cases']:
        actions=[]
        for action in range(4):
            paths=sorted(folder.glob(f'case{case["id"]}_action{action}_seed*/trajectory.json'))
            values=[];second32768=0;two65536=0
            for path in paths:
                replay=json.loads(path.read_text());fs=replay['frames']
                values.append([fs[min(h,len(fs)-1)]['score']-fs[0]['score'] for h in HORIZONS])
                if np.max(case['board'])>=65536:
                    second32768+=any(any(v==32768 for row in f['board'] for v in row) for f in fs[1:])
                two65536+=any(sum(v==65536 for row in f['board'] for v in row)>=2 for f in fs)
            assert len(paths)==next(a['samples'] for c in summary['cases'] if c['case_id']==case['id'] for a in c['actions'] if a['action']==action)
            means=np.mean(values,axis=0).tolist()
            actions.append(dict(action=action,mean_points_by_horizon=dict(zip(map(str,HORIZONS),means)),
                samples=len(paths),built_another32768_after_existing65536=second32768,
                ever_had_two65536=two65536))
        outputs.append(dict(case_id=case['id'],source_move=case['source_move'],recorded_action=case['recorded_action'],actions=actions))
    payload=dict(complete=True,source_summary_sha256=digest(folder/'summary.json'),horizons=HORIZONS,
        cases=outputs,note='Same base policy and same future trajectories at every horizon; terminal reward padding is zero. '
            'This separates horizon effects from changes in future policy. The earlier exact8-step planner optimized '
            'future actions instead and is a different quantity. Actual raw points, not transformed rewards. '
            'Counterfactual saved-position continuations remain excluded from full-game score records.')
    write(folder/'horizon_analysis.json',payload)
    return payload


def main():
    discovery=analyze(BASE/'long_rollout_probe')
    validation=analyze(BASE/'long_rollout_validation')
    colors=['#2878b5','#d45b2a','#298461','#9b57b4']
    names=['Up','Right','Down','Left']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,case in zip(axes,validation['cases']):
        for action,color,name in zip(case['actions'],colors,names):
            ys=[action['mean_points_by_horizon'][str(h)] for h in HORIZONS]
            label=name+(' (recorded)' if action['action']==case['recorded_action'] else '')
            ax.plot(range(4),ys,'o-',label=label,color=color,linewidth=2)
        ax.set_xticks(range(4),HORIZONS)
        ax.set_xlabel('Number of simulated moves included')
        ax.set_ylabel('Mean additional raw points')
        ax.set_title(f'Before move {case["source_move"]:,}')
        ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('Delayed rewards: 128 fresh streams per action\nSame base player and trajectories at every horizon',fontsize=13)
    folder=BASE/'long_rollout_validation'
    fig.savefig(folder/'horizons.png',dpi=160)
    fig.savefig(folder/'horizons.svg')
    plt.close(fig)
    print(json.dumps(validation),flush=True)


if __name__=='__main__':main()
