"""Temporarily lend one owned CPU search slot to a bounded afterstate check.

The original process keeps its exact in-memory game and is resumed in finally,
including on shared STOP so it can save and exit normally. Its wall budget and
recorded game time include this pause. The protocol records both process IDs.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from research.benchmark_training_contention import ROOT, BASE, search_command


def run(args):
    stop = BASE/'STOP'
    if stop.exists():
        raise RuntimeError('Study is user-stopped')
    command = search_command(args.pid)
    if command is None:
        raise ValueError('The requested PID is not an owned Q-search process')
    args.out.mkdir(parents=True, exist_ok=False)
    # Preserve the venv path: resolving its symlink selects bare uv Python,
    # which does not have the project's installed dependencies.
    child_command = [str(ROOT/'.venv/bin/python'), '-m', 'research.afterstate_search_experiment',
        '--checkpoint', str(args.checkpoint.resolve()), '--out', str(args.out/'evaluation'),
        '--games', str(args.games), '--seed-start', str(args.seed_start), '--depth', str(args.depth),
        '--spawn-batch', '256', '--seconds', str(args.seconds), '--stop-file', str(stop)]
    if getattr(args,'nonnegative_leaf',False):
        child_command.append('--nonnegative-leaf')
    protocol = dict(status='preparing', supervisor_pid=os.getpid(), paused_pid=args.pid,
        paused_command=command, command=child_command, seconds=args.seconds,
        started_epoch=time.time(), hypothesis='Does the stronger trained symmetric scalar leaf '
        'improve cheap exact two-move planning on100common games? Preserve its8-view forward '
        'and all spawn probabilities. Compare with ordinary29M afterstate two-move mean34734.64. '
        'This combines training and inference improvements, not a one-factor architecture claim.',
        budget_reason='Previous ordinary scalar100games cost74seconds. Eight-view inference '
        'may require several minutes; give900seconds as a resumable check. Temporarily borrow '
        'one of four original depth4 slots to keep six active CPU studies, then automatically '
        'resume that original process. Its recorded wall time includes the pause.')
    if args.hypothesis:
        protocol['hypothesis']=args.hypothesis
    if args.budget_reason:
        protocol['budget_reason']=args.budget_reason
    protocol.update(games=args.games,seed_start=args.seed_start,depth=args.depth,
                    nonnegative_leaf=getattr(args,'nonnegative_leaf',False))
    def write():
        temporary=args.out/'protocol.tmp'
        temporary.write_text(json.dumps(protocol,indent=2)+'\n')
        temporary.replace(args.out/'protocol.json')
    def interrupt(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupt)
    signal.signal(signal.SIGINT,interrupt)
    paused=False
    child=None
    write()
    try:
        if search_command(args.pid)!=command:
            raise RuntimeError('Worker identity changed before pause')
        os.kill(args.pid,signal.SIGSTOP)
        paused=True
        with (args.out/'run.log').open('ab') as log:
            child=subprocess.Popen(child_command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            protocol.update(status='running',child_pid=child.pid,paused_epoch=time.time())
            write()
            deadline=time.monotonic()+args.seconds+120
            while child.poll() is None:
                if time.monotonic()>deadline:
                    raise TimeoutError('Afterstate worker exceeded its review budget and exit allowance')
                time.sleep(1)
        protocol.update(returncode=child.returncode,status='worker_finished')
        if child.returncode:
            raise RuntimeError(f'Afterstate worker exited with status {child.returncode}; see run.log')
        result=args.out/'evaluation/evaluation.json'
        if result.exists():
            data=json.loads(result.read_text())
            protocol['complete']=data['complete']
            protocol['status']='complete' if data['complete'] else 'saved_partial'
    except BaseException as error:
        protocol.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=repr(error))
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        raise
    finally:
        if paused and search_command(args.pid)==command:
            os.kill(args.pid,signal.SIGCONT)
            protocol['original_worker_resumed']=True
        protocol['finished_epoch']=time.time()
        write()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid',type=int,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=900)
    parser.add_argument('--games',type=int,default=100)
    parser.add_argument('--seed-start',type=int,default=8910000)
    parser.add_argument('--depth',type=int,choices=[2,3],default=2)
    parser.add_argument('--nonnegative-leaf',action='store_true')
    parser.add_argument('--hypothesis')
    parser.add_argument('--budget-reason')
    run(parser.parse_args())
