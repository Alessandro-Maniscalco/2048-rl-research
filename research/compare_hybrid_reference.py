"""Compare two completed 100-game evaluations with independent uncertainty.

The historical C++ player uses libc RNG and the hybrid uses Python's game RNG.
These are different seed sets, not paired games, and timing setups also differ.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from research.afterstate_teacher import write


def main():
    root = Path('runs/research')
    out = root/'endgame_tablebase/hybrid_validation100'
    summary = json.loads((out/'summary.json').read_text())
    verification = json.loads((out/'verified.json').read_text())
    assert summary['complete'] and all(verification.values())
    current = json.loads((out/'games.json').read_text())
    historical = [json.loads(p.read_text()) for p in
                  sorted((root/'tablebase_search').glob('validation_depth8_seed*/result.json'))]
    assert len(historical) == len(current) == 100
    assert {r['seed'] for r in historical} == set(range(8949200, 8949300))
    assert {r['seed'] for r in current} == set(range(8975000, 8975100))
    assert all(r['complete'] and r['terminated'] and not r['truncated'] for r in historical+current)
    scores = [np.array([r['score'] for r in group]) for group in (historical, current)]
    rng = np.random.default_rng(9152)
    difference = (rng.choice(scores[1], (20000,100), replace=True).mean(1) -
                  rng.choice(scores[0], (20000,100), replace=True).mean(1))
    records = []
    for name, group, values in zip(('Historical depth-8', 'Frozen hybrid'), (historical,current), scores):
        records.append(dict(name=name, games=100, mean_score=float(values.mean()),
            median_score=float(np.median(values)), best_score=int(values.max()),
            reaching_16384=sum(r['max_tile']>=16384 for r in group),
            reaching_32768=sum(r['max_tile']>=32768 for r in group),
            reaching_65536=sum(r['max_tile']>=65536 for r in group)))
    comparison = dict(complete=True, arms=records,
        hybrid_minus_historical_mean=float(scores[1].mean()-scores[0].mean()),
        independent_bootstrap95=np.quantile(difference,[.025,.975]).tolist(),
        note='Two completed 100-game evaluations on different seed sets and different RNG implementations. '
             'Independent bootstrap, not paired. Search timing and implementation differ, so this does '
             'not isolate a single intervention or establish a like-for-like compute advantage. '
             'All scores are raw merge points from natural game endings. A selected maximum is not a mean.')
    write(out/'historical_comparison.json', comparison)
    fig, axes = plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    labels = [r['name'] for r in records]
    colors = ['#697c96','#22666a']
    for values, label, color in zip(scores, labels, colors):
        axes[0].step(np.sort(values)/1000, np.arange(1,101)/100,where='post',label=label,color=color)
    axes[0].set(xlabel='Final raw score (thousands)',ylabel='Fraction of games at or below score',
                title='100 natural games per player')
    axes[0].grid(alpha=.2);axes[0].legend()
    thresholds = (16384,32768,65536)
    for i,(group,label,color) in enumerate(zip((historical,current),labels,colors)):
        rates=[sum(r['max_tile']>=t for r in group) for t in thresholds]
        bars=axes[1].bar(np.arange(3)+(i-.5)*.34,rates,.34,label=label,color=color)
        axes[1].bar_label(bars,fmt='%d%%',padding=3)
    axes[1].set(xticks=np.arange(3),xticklabels=[f'{t:,}' for t in thresholds],
                ylabel='Games reaching tile (%)',ylim=(0,110),title='Tile-reaching rates')
    axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('Frozen evaluations on different seeds — not training curves',fontsize=12)
    fig.savefig(out/'historical_comparison.png',dpi=170)
    plt.close(fig)
    print(json.dumps(comparison),flush=True)


if __name__=='__main__': main()
