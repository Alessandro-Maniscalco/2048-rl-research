"""Brief before/paused/resumed timing probe for this study's CPU searches.

Never terminates a worker. Only verified local deeper_q_experiment processes
receive STOP/CONT signals, and a finally block resumes them, including when the
shared user STOP file appears (so those workers can save and exit normally).
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'runs/research/scaled_transformer'


def search_command(pid):
    result = subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
    command = result.stdout.strip()
    prefix = str(ROOT/'runs/research/overnight_qr_search')
    return command if f' -m research.deeper_q_experiment --checkpoint {prefix}' in command else None


def run(args):
    stop = BASE/'STOP'
    if stop.exists(): raise RuntimeError('Study is user-stopped')
    args.out.mkdir(parents=True,exist_ok=False)
    status = json.loads((BASE/'status.json').read_text())
    if status['phase'] != 'training': raise RuntimeError('No active GPU learner')
    job, gpu_pid = status['job'],status['child_pid']
    commands = {pid:search_command(pid) for pid in args.pids}
    if not all(commands.values()): raise RuntimeError('A requested PID is not an owned search worker')
    report = dict(job=job,gpu_pid=gpu_pid,cpu_commands=commands,started_epoch=time.time(),phases=[],
                  limits='Short changing-training-state observation, not an isolated device benchmark. Training seconds exclude monitor time. Discard comparison if GPU job changes.')
    paused = []

    def interrupted(*_): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)

    def observe(label):
        rows=[];start=time.time()
        with (BASE/f'{job}.log').open() as log:
            log.seek(0,2)
            while time.time()-start < args.seconds:
                current=json.loads((BASE/'status.json').read_text())
                if stop.exists() or current.get('job')!=job or current.get('child_pid')!=gpu_pid:
                    raise RuntimeError('STOP or GPU job change; discard timing comparison')
                line=log.readline()
                if not line: time.sleep(.1);continue
                if line.startswith('TRAIN') and '{' in line:
                    try: row=json.loads(line[line.index('{'):])
                    except json.JSONDecodeError: continue
                    rows.append({k:row[k] for k in ('transitions','training_seconds')})
        phase=dict(label=label,wall_seconds=time.time()-start,rows=rows)
        if len(rows)>=2:
            dt=rows[-1]['training_seconds']-rows[0]['training_seconds']
            phase['training_transitions_per_second']=(rows[-1]['transitions']-rows[0]['transitions'])/dt
        report['phases'].append(phase)
        (args.out/'benchmark.json').write_text(json.dumps(report,indent=2))
        print(label,phase.get('training_transitions_per_second'),flush=True)

    try:
        observe('CPU searches running')
        for pid,command in commands.items():
            if search_command(pid)==command:
                os.kill(pid,signal.SIGSTOP);paused.append(pid)
        observe('CPU searches temporarily paused')
    except BaseException as error:
        report['error']=str(error)
        raise
    finally:
        for pid in paused:
            if search_command(pid)==commands[pid]: os.kill(pid,signal.SIGCONT)
        report['search_workers_resumed']=paused
        (args.out/'benchmark.json').write_text(json.dumps(report,indent=2))
    observe('CPU searches resumed')
    report['complete']=True
    (args.out/'benchmark.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pids',type=int,nargs='+',required=True)
    parser.add_argument('--seconds',type=float,default=30.)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args())
