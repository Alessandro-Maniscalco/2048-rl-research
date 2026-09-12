"""Finish GPU planning evaluation, then fill a reward-by-discount comparison."""
import json,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];base=root/'runs/research'
deadline=time.monotonic()+1800
while not (base/'wide_planning_validation.json').exists():
    if time.monotonic()>deadline:raise RuntimeError('Preceding GPU evaluation did not finish.')
    time.sleep(10)
experiments=[('dqn16_wide1024_gamma1',['--reward-mode','corner_snake','--gamma','1']),
             ('dqn16_wide1024_score',['--reward-mode','score']),
             ('dqn16_wide1024_score_gamma1',['--reward-mode','score','--gamma','1'])]
results=[]
for name,extra in experiments:
    command=[sys.executable,'-m','rl2048.dqn_train','--out',str(base/name),
             '--device','mps','--width','1024','--steps','3000000','--seconds','900']+extra
    print('START',name,flush=True)
    with (base/'logs'/f'{name}.log').open('w') as log:
        result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    results.append({'name':name,'command':command,'returncode':result.returncode})
    (base/'wide_reward_queue.json').write_text(json.dumps(results,indent=2))
    print('FINISHED',name,result.returncode,flush=True)
    if result.returncode:break
