"""Finish the current GPU run, test data/representation, then resume PPO queue."""
from pathlib import Path
import os,signal,subprocess,sys,time,json
root=Path(__file__).resolve().parents[1];base=root/'runs/research'
# The earlier queue is intentionally paused; its already running child finishes.
while not (base/'ppo_gamma1/validation.json').exists():
    time.sleep(5)
experiments=[('bc_expert',['--data-selection','expert']),
             ('bc_expert_early',['--data-selection','expert','--early-fraction','.5']),
             ('bc_expert_cnn',['--data-selection','expert','--architecture','cnn']),
             ('bc_expert_cnn_symmetry',['--data-selection','expert','--architecture','cnn','--augment'])]
ledger=[]
try:
    for name,extra in experiments:
        out=base/name
        if out.exists():continue
        command=[sys.executable,'-m','rl2048.offline_train','--algorithm','bc','--data',str(base/'offline_dataset'),
                 '--out',str(out),'--device','mps','--seconds','180']+extra
        print('START',name,flush=True)
        with (base/'logs'/f'{name}.log').open('w') as log:
            result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        ledger.append({'name':name,'returncode':result.returncode,'command':command})
        (base/'representation_queue.json').write_text(json.dumps(ledger,indent=2))
        print('FINISHED',name,result.returncode,flush=True)
        if result.returncode:break
finally:
    # Resume our original, verified queue process, not any unrelated process.
    check=subprocess.run(['ps','-p','95368','-o','command='],capture_output=True,text=True)
    if 'research/run_neural_experiments.py' in check.stdout:
        os.kill(95368,signal.SIGCONT)
