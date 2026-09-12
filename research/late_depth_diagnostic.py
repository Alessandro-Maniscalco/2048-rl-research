"""Inspect a shadowed late-game search setting without running new games.

The original function assigns a special configuration for board mass65381..65500,
then an independent if/elif/else always overwrites it. Compare that control flow
with an isolated one-character elif correction. Stub the search itself and keep
the actual recorded previous search depth/sum. No installed source is modified.
"""
import inspect
import json
from pathlib import Path
import textwrap
from types import SimpleNamespace

import numpy as np

from research.afterstate_teacher import write,digest
from rl2048.agents.endgame import load_native,pack_ranks,ROOT,VENDOR


def corrected_function():
    _,Logic=load_native()
    source=textwrap.dedent(inspect.getsource(Logic.calculate_step))
    before='        if 65260 < board_sum <= 65380:'
    after='        elif 65260 < board_sum <= 65380:'
    assert source.count(before)==1
    namespace={}
    exec(compile(source.replace(before,after),'<isolated-depth-correction>','exec'),
         Logic.calculate_step.__globals__,namespace)
    return namespace['calculate_step']


def inspect_configuration(board,last_sum=0,last_depth=4,corrected=False):
    _,Logic=load_native()
    raw=np.asarray(board,dtype=np.int64)
    ranks=np.zeros(16,dtype=np.uint8);flat=raw.reshape(16)
    ranks[flat>0]=np.log2(flat[flat>0]).astype(np.uint8)
    assert ranks.max()<16
    counts=np.bincount(ranks,minlength=16)
    logic=Logic();logic.last_sum=last_sum;logic.last_depth=last_depth
    encoded=pack_ranks(ranks)
    probe=logic.manager.probe(encoded,counts,int(raw.sum()))
    # These boards came from actual SEARCH decisions: the table stage either
    # missed or declined its candidate. Isolate the subsequent fallback policy,
    # retaining probe's evil-position flag but skipping table validation search.
    logic.manager=SimpleNamespace(probe=lambda *args:(0,*probe[1:]))
    configurations=[]
    def capture(player,initial_depth,max_depth,time_limit):
        configurations.append(dict(initial_depth=int(initial_depth),max_depth=int(max_depth),
            time_limit=float(time_limit),prune=int(player.prune)))
        return 1,initial_depth,[0,0,0,0]
    logic.perform_iterative_search=capture
    player=SimpleNamespace(board=encoded)
    function=corrected_function() if corrected else Logic.calculate_step
    function(logic,player,raw,counts)
    assert len(configurations)==1
    return configurations[0]


def main():
    base=ROOT/'runs/research/endgame_tablebase'
    folders=[base/'hybrid_comparison/compact_seed8973003',
             base/'hybrid_validation100/seed8975019',base/'hybrid_validation100/seed8975023']
    result=[]
    for folder in folders:
        replay=json.loads((folder/'replay.json').read_text())
        decisions=json.loads((folder/'decisions.json').read_text())
        previous_sum,previous_depth=0,4
        for i,d in enumerate(decisions):
            board=replay['frames'][i]['board'];mass=sum(sum(row) for row in board)
            if d['mode']=='search':
                if 65380<mass<=65500:
                    result.append(dict(folder=folder.name,board_index=i,board=board,mass=mass,
                        recorded_depth=d['depth'],previous_sum=previous_sum,previous_depth=previous_depth,
                        original=inspect_configuration(board,previous_sum,previous_depth),
                        corrected=inspect_configuration(board,previous_sum,previous_depth,True)))
                previous_sum=mass;previous_depth=d['depth']
    payload=dict(complete=True,source_sha256=digest(VENDOR/'engine_core/AIPlayer.py'),
        records=result,changed_configurations=sum(r['original']!=r['corrected'] for r in result),
        limitation='Configuration control-flow diagnostic only: actual searches were stubbed. '
            'Force fallback because recorded move came from search, preserve current probe evil flag and previous recorded search sum/depth. '
            'The special assignment is certainly overwritten; a score improvement from correcting it remains untested. '
            'No active player source, checkpoint, cache or evaluation was changed.')
    write(base/'late_depth_diagnostic.json',payload)
    print(json.dumps({k:v for k,v in payload.items() if k!='records'}))
    for r in result[:4]:print(json.dumps(r))


if __name__=='__main__':main()
