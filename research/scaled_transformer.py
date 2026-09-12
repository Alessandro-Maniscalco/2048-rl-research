"""One supervised GPU queue for sustained Transformer research; honors STOP.

Initialize once: python -m research.scaled_transformer init
Run queue:      python -m research.scaled_transformer run
Append explicit experiments to manifest.json while it runs. Old studies stay
paused. A heartbeat reviews results; the runner never invents a winning claim.
"""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import torch
import numpy as np
from research.transformer_td_experiment import config,train_one,tensor_batch
from rl2048.agents.dqn import DQN
from rl2048.agents.neural import tensor_boards

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/research/scaled_transformer'


def write(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(path)


def initial_jobs():
    jobs=[]
    for width,depth,entropy in [(128,2,0.),(512,2,0.),(512,2,.01),
                                (256,2,0.),(256,2,.01),(1024,2,0.),(1024,2,.01),(256,4,.01)]:
        label=f'w{width}_d{depth}_'+('entropy' if entropy else 'plain')+'_seed0'
        c=config('transformer_q','exponents',3)|dict(width=width,depth=depth,heads=width//32,
            attention_entropy=entropy,entropy_target=.8,weight_decay=.01,lr=1e-4,
            capacity=1000000,warmup=16384,epsilon_steps=500000,epsilon_end=.05,
            monitor_games=128,monitor_interval=131072,monitor_initial=True,dense_metrics=True,
            monitor_seed_start=8900000,final_seed_start=8910000,evaluation_device='mps',
            save_best=True,save_training_at_monitor=True,stop_file=str(BASE/'STOP'))
        c['attention_probe']=str(BASE/'attention_probe.npz')
        c['monitor_report']='research.report_scaled_transformer'
        jobs.append(dict(id=label,seed=0,steps=1048576,config=c,
            question=f'Test {width}-wide, {depth}-block Transformer with '+
                ('adaptive attention entropy control.' if entropy else 'ordinary Double DQN.'),
            status='pending'))
    return jobs


def initialize():
    BASE.mkdir(parents=True,exist_ok=False)
    write(BASE/'manifest.json',dict(started_epoch=time.time(),instruction='Continue until the user says stop; roughly one hour before their return is not a hard deadline.',
        jobs=initial_jobs(),validation_seeds=[8900000,8900127],selection_seeds=[8910000,8910099],
        fresh_final_test_seeds_reserved=[8920000,8920099]))
    write(BASE/'results.json',[])
    write(BASE/'status.json',dict(phase='initialized',updated_epoch=time.time()))
    from rl2048.agents.ntuple import encode
    frames=json.loads((ROOT/'runs/research/deeper_q/replay_latest_student.json').read_text())['frames']
    ids=np.linspace(0,len(frames)-1,64,dtype=int)
    boards=np.array([encode(np.array(frames[i]['board'])).reshape(16) for i in ids],np.uint8)
    np.savez_compressed(BASE/'attention_probe.npz',boards=boards)
    write(BASE/'attention_probe.json',dict(source='deeper_q/replay_latest_student.json',seed=8630000,
        frame_indices=ids.tolist(),usage='Fixed diagnostic states only; never used as training data or policy evaluation.'))


def benchmark():
    torch.set_num_threads(2)
    records=[]
    for width in (128,256,512,1024):
        for batch_size in (256,512,1024):
            if (BASE/'STOP').exists():return
            torch.manual_seed(0)
            learner=DQN(architecture='transformer_q',input_encoding='exponents',width=width,
                depth=2,heads=width//32,device='mps',attention_entropy=.01,weight_decay=.01)
            b=np.random.default_rng(0).integers(0,12,(batch_size,16),dtype=np.uint8)
            batch=dict(states=b,next_states=b,actions=np.arange(batch_size)%4,
                rewards=np.ones(batch_size,np.float32),terminated=np.zeros(batch_size,bool),
                next_masks=np.ones((batch_size,4),bool))
            durations=[]
            for i in range(8):
                start=time.perf_counter()
                with torch.no_grad():learner.policy(tensor_boards(b[:128],'mps')).cpu().numpy()
                metrics=learner.update(tensor_batch(batch,'mps'));float(metrics['q_loss'].cpu())
                if i>=3:durations.append(time.perf_counter()-start)
            r=dict(width=width,batch=batch_size,parameters=sum(p.numel() for p in learner.policy.parameters()),
                seconds_per_cycle=float(np.median(durations)),examples_per_second=batch_size/float(np.median(durations)),
                allocated_bytes=torch.mps.current_allocated_memory(),driver_bytes=torch.mps.driver_allocated_memory())
            records.append(r);write(BASE/'benchmark.json',records);print(json.dumps(r),flush=True)
            del learner,metrics,batch,b;gc.collect();torch.mps.empty_cache()


def worker(job_id):
    torch.set_num_threads(2)
    job=next(j for j in json.loads((BASE/'manifest.json').read_text())['jobs'] if j['id']==job_id)
    out=BASE/job_id
    if out.exists():raise RuntimeError('Existing attempt preserved; use a new job ID or explicit resume.')
    trainer=train_one
    if job['config'].get('algorithm')=='native_behavior_cloning':
        from research.native_policy_experiment import train_one as trainer
    if job['config'].get('algorithm')=='dagger':
        from research.dagger_experiment import train_one as trainer
    if job['config'].get('algorithm')=='policy_sampling_audit':
        from research.policy_sampling_experiment import train_one as trainer
    if job['config'].get('algorithm') in ('reinforce', 'reinforce_loo'):
        from research.reinforce_experiment import train_one as trainer
    if job['config'].get('algorithm')=='teacher_ppo':
        from research.teacher_ppo_experiment import train_one as trainer
    if job['config'].get('algorithm')=='teacher_policy_value':
        from research.teacher_policy_experiment import train_one as trainer
    if job['config'].get('algorithm')=='neural_afterstate':
        from research.neural_afterstate_experiment import train_one as trainer
        if job['config'].get('teacher_only'):
            from research.teacher_transformer_experiment import train_one as trainer
    result=trainer(job['config'],job['seed'],job['steps'],out,'mps')
    write(out/'result.json',result)


def run():
    lock=(BASE/'runner.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def stopping(*_):(BASE/'STOP').touch()
    signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
    awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    child=None
    try:
        while not (BASE/'STOP').exists():
            manifest=json.loads((BASE/'manifest.json').read_text())
            results=json.loads((BASE/'results.json').read_text())
            done={r['id'] for r in results}
            job=next((j for j in manifest['jobs'] if j['id'] not in done and j.get('status','pending')=='pending'),None)
            if job is None:
                write(BASE/'status.json',dict(phase='awaiting_next_experiment',pid=os.getpid(),completed=len(results),updated_epoch=time.time()))
                time.sleep(5);continue
            folder=BASE/job['id']
            if folder.exists():
                # A prior crash is evidence, not a successful run. Supervisor
                # must choose a new ID and an explicit saved-state continuation.
                results.append(dict(id=job['id'],status='blocked_existing_attempt',question=job['question']))
                write(BASE/'results.json',results);continue
            command=[sys.executable,'-m','research.scaled_transformer','worker','--job',job['id']]
            with (BASE/(job['id']+'.log')).open('w') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write(BASE/'status.json',dict(phase='training',pid=os.getpid(),child_pid=child.pid,job=job['id'],command=command,updated_epoch=time.time()))
                stop_start=None
                while child.poll() is None:
                    if (BASE/'STOP').exists():
                        stop_start=stop_start or time.time()
                        if time.time()-stop_start>60:
                            os.killpg(child.pid,signal.SIGTERM)
                            try:child.wait(timeout=10)
                            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                            break
                    time.sleep(1)
                row=dict(id=job['id'],question=job['question'],config=job['config'],seed=job['seed'],
                    returncode=child.returncode,finished_epoch=time.time(),status='failed')
                child=None
                if (folder/'result.json').exists():
                    row['result']=json.loads((folder/'result.json').read_text())
                    row['status']=('plateau' if row['result'].get('stop_reason')=='policy_plateau'
                        else 'budget_complete' if row['result'].get('stop_reason')=='training_time_budget' and row['result'].get('complete')
                        else 'complete' if row['result'].get('completed_budget') else 'interrupted')
                results.append(row);write(BASE/'results.json',results)
                subprocess.run([sys.executable,'-m','research.report_scaled_transformer'],cwd=ROOT,check=False)
        write(BASE/'status.json',dict(phase='stopped',pid=os.getpid(),updated_epoch=time.time()))
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
        awake.terminate()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['init','benchmark','run','worker'])
    p.add_argument('--job');args=p.parse_args()
    if args.mode=='worker':worker(args.job)
    else:globals()[{'init':'initialize'}.get(args.mode,args.mode)]()
