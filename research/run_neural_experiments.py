"""Finite experiment queue. Each child saves its own config/results/checkpoint."""
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1]
base=root/'runs/research'
experiments=[]
for algorithm in ('bc','awr','iql','cql'):
    experiments.append((algorithm+'_180',['-m','rl2048.offline_train','--algorithm',algorithm,
        '--data','runs/research/offline_dataset','--seconds','180']))
experiments.append(('sac_180',['-m','rl2048.sac_train','--seconds','180']))
for name,extra in [('ppo_control',[]),('ppo_batch16384',['--batch','16384']),
                   ('ppo_gamma999',['--gamma','.999']),('ppo_gamma1',['--gamma','1']),
                   ('ppo_bottom_left',['--reward-mode','bottom_left']),('ppo_top_right',['--reward-mode','top_right']),
                   ('ppo_constant',['--reward-mode','constant'])]:
    experiments.append((name,['-m','rl2048.ppo_train','--seconds','180']+extra))
ledger=[]
for name,command in experiments:
    output=base/name
    if output.exists():
        print('Already exists, skipping:',name,flush=True)
        continue
    command=[sys.executable]+command+['--out',str(output),'--device','mps']
    print('START',name,flush=True)
    start=time.time()
    with (base/'logs'/f'{name}.log').open('w') as log:
        result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    ledger.append({'name':name,'command':command,'started':start,'elapsed':time.time()-start,'returncode':result.returncode})
    (base/'neural_queue.json').write_text(json.dumps(ledger,indent=2))
    print('FINISHED',name,result.returncode,flush=True)
    if result.returncode:
        print('Stopping queue so failure can be inspected.',flush=True)
        break
