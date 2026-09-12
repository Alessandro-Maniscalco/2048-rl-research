"""Post-hoc diagnosis of sampling error; does not change frozen choices."""
import json
from pathlib import Path

import numpy as np

from research.afterstate_teacher import digest, write


def main():
    folder = Path('runs/research/endgame_tablebase/rollout_choice_study')
    assert json.loads((folder/'summary.json').read_text())['complete']
    files = ['choices.json', 'discovery/games.json', 'validation/games.json', 'summary.json']
    hashes = {name: digest(folder/name) for name in files}
    choices = json.loads((folder/'choices.json').read_text())['choices']
    games = json.loads((folder/'discovery/games.json').read_text())
    index = {(r['case_id'], r['future_seed'], r['first_action']): r['additional_points'] for r in games}
    rows = []
    for c in choices:
        seeds = sorted({s for i, s, _ in index if i == c['case_id']})
        diffs = np.array([index[c['case_id'], s, c['long_action']]
                          - index[c['case_id'], s, c['recorded_action']] for s in seeds], dtype=float)
        boot = np.random.default_rng(99021+c['case_id']).choice(diffs, (20000, len(diffs))).mean(1)
        rows.append(dict(case_id=c['case_id'], changed=c['long_action'] != c['recorded_action'],
            mean_discovery_advantage=float(diffs.mean()), paired_sd=float(diffs.std(ddof=1)),
            paired_bootstrap95=np.quantile(boot, [.025, .975]).tolist()))
    assert all(digest(folder/name) == value for name, value in hashes.items())
    write(folder/'selection_noise.json', dict(rows=rows, inputs_sha256=hashes, post_hoc=True,
        note='Exploratory diagnosis after seeing validation. These intervals are not adjusted '
             'for selecting the best sampled action or multiple comparisons. No new selector '
             'is retrospectively claimed as independently validated; frozen choices unchanged.',
        learning='Both discovery overrides have intervals spanning zero. Case0 chose an '
                 'apparent +227-point advantage with paired SD29828, then lost14117 points '
                 'on fresh validation. More reliable comparisons or retaining the base '
                 'action under uncertainty deserve a future preregistered test.'))


if __name__ == '__main__':
    main()
