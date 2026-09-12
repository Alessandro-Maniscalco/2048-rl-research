"""Change TC base rate and symmetry normalization from the same warm start."""
import json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];base=root/'runs/research'
experiments=[('local_v2_tc_alpha01',['--tc-alpha','.1']),
             ('local_v2_tc_normalized',['--normalize-collisions']),
             ('local_v2_tc_normalized_alpha01',['--tc-alpha','.1','--normalize-collisions'])]
results=[]
for name,extra in experiments:
    command=[sys.executable,'-m','rl2048.ntuple_train','--resume',str(base/'native_td_control'),
             '--out',str(base/name),'--seconds','300','--workers','12','--tc-fraction','1','--seed','17']+extra
    print('START',name,flush=True)
    with (base/'logs'/f'{name}.log').open('w') as log:
        result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    results.append({'name':name,'command':command,'returncode':result.returncode})
    (base/'tc_stability_queue.json').write_text(json.dumps(results,indent=2))
    print('FINISHED',name,result.returncode,flush=True)
    if result.returncode:break
