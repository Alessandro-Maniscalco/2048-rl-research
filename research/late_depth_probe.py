"""Reproducible offline native queries for the isolated late-depth correction.

These are three saved boards queried twice, not additional evaluation games.
The installed player remains unchanged. Adaptive timing state is fresh for both
variants except for the recorded previous search sum and depth.
"""
import json
from pathlib import Path
from types import MethodType

import numpy as np

from research.afterstate_teacher import write
from research.late_depth_diagnostic import corrected_function
from rl2048.agents.endgame import EndgameAgent
from rl2048.game import legal_actions


def probe(output):
    output=Path(output)
    if output.exists():raise FileExistsError('Choose a new output for a new timed probe')
    base=Path('runs/research/endgame_tablebase')
    diagnostic=json.loads((base/'late_depth_diagnostic.json').read_text())
    selected=[];seen=set();results=[]
    for row in diagnostic['records']:
        if row['folder'] not in seen:selected.append(row);seen.add(row['folder'])
    for row in selected:
        for variant in ('original','corrected'):
            agent=EndgameAgent(1.)
            agent.logic.last_sum=row['previous_sum'];agent.logic.last_depth=row['previous_depth']
            if variant=='corrected':
                agent.logic.calculate_step=MethodType(corrected_function(),agent.logic)
            board=np.asarray(row['board'])
            agent.act(board,legal_actions(board))
            results.append(dict(folder=row['folder'],board_index=row['board_index'],mass=row['mass'],
                variant=variant,actual_decision=agent.last_decision))
    write(output,dict(complete=True,results=results,
        note='Offline board queries; original previous sum/depth, reset time_ratio for both variants. '
            'No game restarts or installed policy changes. Higher depth or changed timings do not prove higher game scores.'))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    probe(parser.parse_args().out)
