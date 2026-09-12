"""Select the matched TD variant on validation, then use fresh test seeds."""
import json
from pathlib import Path
from rl2048.agents.ntuple import NTupleAgent
from rl2048.ntuple_train import evaluate_fast
from rl2048.fast_replay import save_fast_replay

if __name__=='__main__':
    root=Path('runs/research')
    candidates={name:json.loads((root/name/'validation.json').read_text())['depth3_downgrade16384']
                for name in ('native_td_control','native_td_lambda')}
    winner=max(candidates,key=lambda name:candidates[name]['summary']['mean_score'])
    agent=NTupleAgent.load(root/winner,mmap_mode='r')
    agent.depth=3;agent.downgrade_threshold=16384
    agent.metadata.update({'selected_using':'Fixed 100 validation games6000000..6000099',
                           'held_out_seeds':'7100000..7100099, not previously used',
                           'selected_variant':winner})
    agent.display_name='Locally trained afterstate TD · additional training · depth3'
    folder=root/'local_selected_v2';agent.save(folder)
    print('FROZEN',winner,flush=True)
    summary,episodes=evaluate_fast(agent,range(7100000,7100100),workers=12)
    (folder/'test.json').write_text(json.dumps({'summary':summary,'episodes':episodes,
        'provenance':agent.metadata,'split':'Fresh held-out test after validation selection'},indent=2))
    print('TEST',{k:v for k,v in summary.items() if k!='seeds'},flush=True)
    best=max(episodes,key=lambda row:row['score'])
    print('REPLAY',save_fast_replay(agent,root/'local_v2_test_record_replay.html',best['seed']),flush=True)
    record=max(candidates[winner]['episodes'],key=lambda row:row['score'])
    print('VALIDATION RECORD',save_fast_replay(agent,root/'local_v2_validation_record_replay.html',record['seed']),flush=True)
    from rl2048.research_report import build
    build()
