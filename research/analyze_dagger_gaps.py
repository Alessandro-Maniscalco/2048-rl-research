"""Measure teacher-estimated mistakes in already saved student collection rounds.

This reads recorded actions, never executes or changes a policy. Each block was
collected by that round's frozen student before its next fitting updates.
Teacher Q is an approximate root value, not the student's actual future return.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from research.afterstate_teacher import digest, write


def analyze(folder, out, last_rounds=4):
    folder, out = Path(folder), Path(out)
    if last_rounds < 1:
        raise ValueError('Choose at least one completed collection round')
    if out.exists():
        raise FileExistsError('Preserve completed diagnostics')
    manifest = json.loads((folder/'dataset_blocks.json').read_text())
    selected = manifest[-last_rounds:]
    if len(selected) != last_rounds:
        raise ValueError('Not enough completed collection rounds')
    gaps, agreements, blocks = [], [], []
    for block in selected:
        path = folder/block['file']
        if path.name != block['file'] or digest(path) != block['sha256']:
            raise ValueError('Collection block path or hash mismatch')
        with np.load(path, allow_pickle=False) as data:
            q = data['q'].astype(np.float64) * 128.
            actions, legal, best = data['played_actions'], data['legal'], data['best']
            rows = np.arange(len(actions))
            if len(rows) != block['boards'] or not legal[rows, actions].all():
                raise ValueError('Recorded counts or actions are invalid')
            if not np.isfinite(q[legal]).all():
                raise ValueError('Legal teacher values must be finite')
            gap = np.max(np.where(legal, q, -np.inf), axis=1)-q[rows, actions]
            agree = best[rows, actions]
            gaps.append(gap)
            agreements.append(agree)
        blocks.append({k:block[k] for k in ('file', 'sha256', 'collection_seed', 'boards')})
    gaps, agreements = np.concatenate(gaps), np.concatenate(agreements)
    # The training label permits small numerical ties. Diagnose its mistakes
    # using the saved accepted-action set, while retaining unmodified raw gaps.
    errors = gaps[~agreements]
    total = float(errors.sum())
    bins = []
    for low, high in zip((0, 16, 128, 1024, 10000), (16, 128, 1024, 10000, np.inf)):
        pick = (errors >= low) & (errors < high)
        bins.append(dict(lower_points=low, upper_points=None if np.isinf(high) else high,
            decisions=int(pick.sum()), fraction_of_disagreements=float(pick.mean()) if len(errors) else 0.,
            fraction_of_total_estimated_gap=float(errors[pick].sum()/total) if total else 0.))
    result = dict(source=str(folder.resolve()), blocks=blocks, decisions=len(gaps),
        teacher_action_agreement=float(agreements.mean()), disagreements=len(errors),
        mean_raw_teacher_gap_points=float(gaps.mean()),
        disagreement_gap_quantiles_points={str(p):float(np.quantile(errors, p))
            for p in (.25, .5, .75, .9, .95, .99)} if len(errors) else {},
        bins=bins,
        note='Recorded greedy actions from the last completed collection rounds, before their fitting updates. These are different sequential policies and correlated states, not an evaluation of the final checkpoint. Gaps are approximate frozen teacher root Q differences in raw point units, not measured losses of eventual game score. Accepted numerical ties are excluded from disagreement bins. No model, training data, reward or sampling rule changed.')
    assert sum(row['decisions'] for row in bins) == len(errors)
    out.parent.mkdir(parents=True, exist_ok=True)
    write(out, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--last-rounds', type=int, default=4)
    print(json.dumps(analyze(**vars(parser.parse_args())), indent=2))
