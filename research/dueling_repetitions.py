from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess,sys,json
root=Path(__file__).resolve().parents[1]
def run(seed):
    name=f'dueling16_seed{seed}';out=root/'runs/research'/name
    command=[sys.executable,'-m','rl2048.dqn_train','--out',str(out),'--dueling','--reward-mode','corner_snake',
             '--input-encoding','exponents','--steps','3000000','--seconds','600','--device','cpu','--seed',str(seed)]
    with (root/'runs/research/logs'/f'{name}.log').open('w') as log:
        result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    print(name,result.returncode,flush=True)
    return {'name':name,'returncode':result.returncode,'command':command}
if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(run,[0,1,2]))
    (root/'runs/research/dueling_repetitions.json').write_text(json.dumps(results,indent=2))
