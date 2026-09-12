from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess,sys,json
root=Path(__file__).resolve().parents[1]
experiments=[('dqn16_control_seed1',['--reward-mode','score','--seed','1']),
             ('dqn16_control_seed2',['--reward-mode','score','--seed','2']),
             ('dqn16_batch1024',['--reward-mode','corner_snake','--batch','1024']),
             ('dqn16_long',['--reward-mode','corner_snake','--batch','1024','--steps','10000000',
                            '--resume','runs/research/dqn16_corner_snake'])]
def run(item):
    name,extra=item
    command=[sys.executable,'-m','rl2048.dqn_train','--out',f'runs/research/{name}',
             '--device','cpu','--seconds','900']+extra
    with (root/'runs/research/logs'/f'{name}.log').open('w') as log:
        result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    print(name,result.returncode,flush=True)
    return {'name':name,'returncode':result.returncode,'command':command}
if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(run,experiments))
    (root/'runs/research/dqn_followups.json').write_text(json.dumps(results,indent=2))
