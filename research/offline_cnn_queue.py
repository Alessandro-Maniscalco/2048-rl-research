"""Controlled CNN/expert-data follow-up to the existing BC CNN control."""
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
base=root/'runs/research'
ledger=[]
for algorithm in ('awr','iql','cql'):
    name=algorithm+'_expert_cnn_symmetry'
    out=base/name
    if out.exists():
        continue
    command=[sys.executable,'-m','rl2048.offline_train','--algorithm',algorithm,
             '--data',str(base/'offline_dataset'),'--out',str(out),'--device','mps',
             '--seconds','180','--data-selection','expert','--architecture','cnn','--augment']
    print('START',name,flush=True)
    with (base/'logs'/f'{name}.log').open('w') as log:
        result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    ledger.append({'name':name,'command':command,'returncode':result.returncode})
    (base/'offline_cnn_queue.json').write_text(json.dumps(ledger,indent=2))
    print('FINISHED',name,result.returncode,flush=True)
    if result.returncode:
        break
