"""Adaptive, single-factor DQN experiment: encoding first, reward second."""
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1]
base=root/'runs/research'
results=[]

def run(name,encoding='exponents',reward='score',seed=0,double=True):
    out=base/name
    command=[sys.executable,'-m','rl2048.dqn_train','--out',str(out),'--device','cpu',
             '--steps','3000000','--seconds','300','--batch','4096','--input-encoding',encoding,
             '--reward-mode',reward,'--seed',str(seed)]
    if not double:command.append('--no-double')
    print('START',name,flush=True)
    if not out.exists():
        with (base/'logs'/f'{name}.log').open('w') as log:
            completed=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f'{name} failed; inspect its log')
    data=json.loads((out/'validation.json').read_text())
    result={'name':name,'encoding':encoding,'reward':reward,'seed':seed,'double':double,
            'mean_score':data['summary']['mean_score'],'command':command}
    results.append(result)
    (base/'dqn16_comparison.json').write_text(json.dumps(results,indent=2))
    print('RESULT',json.dumps({k:v for k,v in result.items() if k!='command'}),flush=True)
    subprocess.run([sys.executable,'-m','rl2048.research_report'],cwd=root,stdout=subprocess.DEVNULL)
    return result

if __name__=='__main__':
    encodings=[run('dqn16_'+encoding,encoding) for encoding in ('exponents','raw','relative')]
    best=max(encodings,key=lambda r:r['mean_score'])
    rewards=[best]+[run('dqn16_'+reward,best['encoding'],reward) for reward in ('corner','snake','corner_snake','dense_corner')]
    winner=max(rewards,key=lambda r:r['mean_score'])
    run('dqn16_plain',best['encoding'],double=False)
    for seed in (1,2):
        run('dqn16_winner_seed'+str(seed),winner['encoding'],winner['reward'],seed)
    (base/'dqn16_selection.json').write_text(json.dumps({'best_encoding':best,'winner':winner},indent=2))
    print('DQN comparisons completed',flush=True)
