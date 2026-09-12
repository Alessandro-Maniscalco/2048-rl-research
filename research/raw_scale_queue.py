"""Compare fixed raw-tile scales; only the input divisor changes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess,sys,json
root=Path(__file__).resolve().parents[1]
base=root/'runs/research'
def run(divisor):
    name=f'dqn16_raw_scale{divisor}'
    command=[sys.executable,'-m','rl2048.dqn_train','--out',str(base/name),
             '--input-encoding','raw','--raw-divisor',str(divisor),
             '--device','cpu','--steps','3000000','--seconds','900']
    with (base/'logs'/f'{name}.log').open('w') as log:
        result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    print(name,result.returncode,flush=True)
    return {'name':name,'command':command,'returncode':result.returncode}
if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(run,[16,256,2048]))
    (base/'raw_scale_queue.json').write_text(json.dumps(results,indent=2))
