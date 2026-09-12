"""Repeat the wider scalar network, then extend its first training run."""
from pathlib import Path
import subprocess,sys,json
root=Path(__file__).resolve().parents[1];base=root/'runs/research'
experiments=[('dqn16_wide1024_seed1',['--seed','1']),
             ('dqn16_wide1024_seed2',['--seed','2']),
             ('dqn16_wide1024_long',['--resume','runs/research/dqn16_wide1024','--steps','10000000'])]
ledger=[]
for name,extra in experiments:
    command=[sys.executable,'-m','rl2048.dqn_train','--out',str(base/name),'--device','mps',
             '--width','1024','--reward-mode','corner_snake','--seconds','900']+extra
    print('START',name,flush=True)
    with (base/'logs'/f'{name}.log').open('w') as log:
        result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    ledger.append({'name':name,'command':command,'returncode':result.returncode})
    (base/'wide_dqn_queue.json').write_text(json.dumps(ledger,indent=2))
    print('FINISHED',name,result.returncode,flush=True)
    if result.returncode:break
