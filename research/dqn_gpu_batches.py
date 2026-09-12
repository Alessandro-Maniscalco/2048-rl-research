from pathlib import Path
import subprocess,sys,time,os,signal,json
root=Path(__file__).resolve().parents[1];base=root/'runs/research'
while not (base/'ppo_bottom_left/validation.json').exists():time.sleep(5)
results=[]
try:
    for batch in (16384,65536):
        name=f'dqn16_batch{batch}'
        command=[sys.executable,'-m','rl2048.dqn_train','--out',str(base/name),'--device','mps',
                 '--reward-mode','corner_snake','--batch',str(batch),'--steps','3000000','--seconds','900']
        print('START',name,flush=True)
        with (base/'logs'/f'{name}.log').open('w') as log:
            result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        results.append({'name':name,'command':command,'returncode':result.returncode})
        (base/'dqn_gpu_batches.json').write_text(json.dumps(results,indent=2))
        print('FINISHED',name,result.returncode,flush=True)
        if result.returncode:break
finally:
    check=subprocess.run(['ps','-p','95368','-o','command='],capture_output=True,text=True)
    if 'research/run_neural_experiments.py' in check.stdout:os.kill(95368,signal.SIGCONT)
