"""Finish the fixed one-hour study; never extend the persisted deadline."""
import json,os,signal,subprocess,time,sys,fcntl,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'runs/research/fair_afterstate_hour'
def write(n,d):
 p=B/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
manifest=json.loads((B/'manifest.json').read_text());deadline=manifest['deadline_epoch']
lock=(B/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
child=None;stopped=False
signal.signal(signal.SIGTERM,lambda *_:globals().update(stopped=True))
signal.signal(signal.SIGINT,lambda *_:globals().update(stopped=True))
awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
def execute(cmd,limit):
 global child
 with (B/'final_execution.log').open('a') as out:
  child=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
  write('status.json',dict(phase='running',pid=os.getpid(),child_pid=child.pid,command=cmd,deadline=deadline))
  end=min(deadline,time.time()+limit)
  while child.poll() is None and time.time()<end and not stopped and not (B/'STOP').exists():time.sleep(.2)
  if child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=2)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  code=child.returncode;child=None;return code
try:
 if not (B/'paired_seed0/training.json').exists() and time.time()<deadline-60:
  execute([sys.executable,str(ROOT/'research/fair_afterstate_train.py'),'--out',str(B/'paired_seed0'),'--steps','1000000','--seconds','35','--eval-every','10000000','--deadline',str(deadline-65)],45)
 if time.time()<deadline-10 and not stopped and not (B/'STOP').exists():
  execute([sys.executable,str(ROOT/'research/fair_afterstate_evaluate.py')],deadline-time.time()-3)
 write('status.json',dict(phase='stopped' if stopped or (B/'STOP').exists() else 'complete',pid=os.getpid(),finished_epoch=time.time(),deadline=deadline))
finally:
 if child and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
 awake.terminate()
