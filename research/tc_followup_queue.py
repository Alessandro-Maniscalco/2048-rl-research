"""Finish the frozen test, then improve the validation-selected TD checkpoint."""
from pathlib import Path
import subprocess,sys,time
root=Path(__file__).resolve().parents[1];base=root/'runs/research'
deadline=time.monotonic()+1800
while not (base/'local_selected_v2/test.json').exists():
    if time.monotonic()>deadline:
        raise RuntimeError('Timed out waiting for the preceding evaluation.')
    time.sleep(10)
command=[sys.executable,'-m','rl2048.ntuple_train','--resume',str(base/'native_td_control'),
         '--out',str(base/'local_v2_tc'),'--seconds','300','--workers','12',
         '--tc-fraction','1','--seed','17']
print('START',command,flush=True)
with (base/'logs/local_v2_tc.log').open('w') as log:
    result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
print('FINISHED',result.returncode,flush=True)
sys.exit(result.returncode)
