"""Freeze the validation-selected local model and use fresh held-out seeds."""
import json
from pathlib import Path
from rl2048.agents.ntuple import NTupleAgent
from rl2048.ntuple_train import evaluate_fast
from rl2048.fast_replay import save_fast_replay

if __name__=='__main__':
    root=Path('runs/research')
    agent=NTupleAgent.load(root/'local_tc_numba/best',mmap_mode='r')
    agent.depth=3;agent.downgrade_threshold=16384
    agent.display_name='Locally trained afterstate TD · depth 3 + tile downgrading'
    agent.metadata.update({'provenance':'Locally trained from zero; no published weights',
        'lineage':[{'run':'zero4','games':175200},{'run':'native_otd','games':1008000},
                   {'run':'local_tc_numba/best','games':122400}],
        'selected_using':'6000000..6000099 validation games',
        'native_transition_count':'not recovered; training_transitions does not include native phase'})
    agent.training_games=175200+1008000+122400
    agent.save(root/'local_selected')
    results={}
    summary,episodes=evaluate_fast(agent,range(7_000_000,7_000_100),workers=12)
    results={'summary':summary,'episodes':episodes,'provenance':agent.metadata,'split':'Fresh held-out test; not used to select this checkpoint'}
    (root/'local_selected/test.json').write_text(json.dumps(results,indent=2))
    print('HELD OUT',{k:v for k,v in summary.items() if k!='seeds'},flush=True)
    best=max(episodes,key=lambda r:r['score'])
    print('REPLAY',save_fast_replay(agent,root/'local_test_record_replay.html',best['seed']),flush=True)
    # Also preserve the previously observed validation record, explicitly selected.
    validation=json.loads((root/'local_finalist_validation.json').read_text())
    record=max(validation['local_tc_numba/best/depth3/threshold16384']['episodes'],key=lambda r:r['score'])
    print('VALIDATION REPLAY',save_fast_replay(agent,root/'local_record_replay.html',record['seed']),flush=True)
