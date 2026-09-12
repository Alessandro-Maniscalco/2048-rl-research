"""Compare final/best wide Q policies, then time actual CPU/MPS search games."""
import json
from pathlib import Path
import torch
from rl2048.agents.neural import NeuralAgent
from rl2048.agents.q_planning import PlanningQAgent
from rl2048.evaluate import evaluate

if __name__=='__main__':
    root=Path('runs/research');results={};torch.set_num_threads(1)
    for version in ('best','last'):
        a=NeuralAgent.load(root/'dqn16_wide1024_long'/version)
        s,e=evaluate(a,seeds=range(6300000,6300100),max_steps=40000)
        results[version]={'summary':s,'episodes':e}
        print(version,s['mean_score'],flush=True)
    (root/'wide_policy_validation.json').write_text(json.dumps(results,indent=2))
    # Pick using validation, before observing the separate timing games.
    selected=max(results,key=lambda k:results[k]['summary']['mean_score'])
    timing={}
    for device in ('cpu','mps'):
        a=PlanningQAgent(NeuralAgent.load(root/'dqn16_wide1024_long'/selected,device),reward_mode='corner_snake',depth=2)
        s,e=evaluate(a,seeds=range(6300000,6300005),max_steps=40000)
        timing[device]={'summary':s,'episodes':e,'seconds_per_move':s['elapsed_seconds']/s['environment_transitions']}
        print('TIMING',device,s['mean_score'],timing[device]['seconds_per_move'],flush=True)
    device=min(timing,key=lambda k:timing[k]['seconds_per_move'])
    (root/'compute/wide_planning_games.json').write_text(json.dumps(timing,indent=2))
    a=PlanningQAgent(NeuralAgent.load(root/'dqn16_wide1024_long'/selected,device),reward_mode='corner_snake',depth=2)
    s,e=evaluate(a,seeds=range(6300000,6300100),max_steps=40000)
    (root/'wide_planning_validation.json').write_text(json.dumps({'summary':s,'episodes':e,
        'network':selected,'device':device,'depth':2,'split':'validation'},indent=2))
    print('PLANNING',s['mean_score'],s['tile_reaching_rates'],flush=True)
