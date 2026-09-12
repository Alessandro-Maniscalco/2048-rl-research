"""Bounded, sequential MLP Q-learning comparisons with a journal per experiment.

python research/overnight_mlp.py --hours 8
The runner owns its child process groups, prevents duplicate runners with a
file lock, honors STOP, and never uses held-out scores to choose experiments.
"""
import argparse
import csv
import fcntl
import html
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/research/mlp_overnight'
PYTHON=str(ROOT/'.venv/bin/python')

def write_json(path,data):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False));temp.replace(path)

def utc(epoch=None):
    return datetime.fromtimestamp(epoch or time.time(),timezone.utc).isoformat()

def journal(manifest,results):
    lines=['# Overnight MLP Q-learning experiments','',
           f"Started: {manifest['started_utc']}. Hard deadline: {manifest['deadline_utc']}.",'',
           'All models output four Q-values; evaluation uses legal argmax without planning. '
           'Validation selects experiments. Fresh tests are reserved for the final frozen selection. '
           'Results below use the final checkpoint; intermediate 20-game best checkpoints remain saved.','']
    for r in results:
        lines += [f"## {r['name']}",'',f"**Testing:** {r['question']}",'',
                  f"**Configuration:** `{json.dumps(r['config'],sort_keys=True)}`",'',
                  f"**Result:** {r['result']}",'',f"**Learning:** {r['learning']}",'',
                  f"[Run files]({r['name']}/)",'']
    (BASE/'journal.md').write_text('\n'.join(lines))
    cards=''.join('<article><h2>'+html.escape(r['name'])+'</h2><p><b>Testing:</b> '+html.escape(r['question'])+
                  '</p><p><b>Result:</b> '+html.escape(r['result'])+'</p><p><b>Learning:</b> '+html.escape(r['learning'])+
                  '</p><details><summary>Settings and files</summary><pre>'+html.escape(json.dumps(r['config'],indent=2))+
                  '</pre><a href="'+r['name']+'/">Run files</a></details></article>' for r in reversed(results))
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><meta http-equiv="refresh" content="60">
<title>Overnight MLP experiments</title><style>body{font:17px/1.6 system-ui;max-width:1000px;margin:35px auto;padding:0 20px;background:#f5f2ed;color:#232323}article{background:white;padding:24px;margin:20px 0;border-radius:16px}pre{overflow:auto;font-size:13px}img{max-width:100%}a{color:#075c92}</style>
<h1>Overnight MLP Q-learning</h1><p>Four Q-values → highest legal value → move. No planning.</p>'''
    page+='<p>Deadline: '+manifest['deadline_utc']+' (UTC). <a href="status.json">Live status</a> · <a href="journal.md">Journal</a> · <a href="../index.html">Research dashboard</a></p>'
    page+='<p>All figures are validation, unless explicitly labeled fresh test. Intermediate rankings are exploratory.</p><img src="curves.png" alt="Validation scores and TD loss by training transitions">'+cards
    if (BASE/'final_test.json').exists():
        d=json.loads((BASE/'final_test.json').read_text())
        page+='<article><h2>Frozen winner: fresh test</h2><pre>'+html.escape(json.dumps(d['summary'],indent=2))+'</pre><a href="replay.html">Watch a test game</a></article>'
    (BASE/'index.html').write_text(page)
    plot_curves(results)

def plot_curves(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(10,8),layout='constrained')
    for r in results[-12:]:
        for ax,filename,y in [(axes[0],'validation_curve.csv','validation_score'),(axes[1],'progress.csv','q_loss')]:
            p=BASE/r['name']/filename
            if p.exists():
                rows=list(csv.DictReader(p.open()))
                ax.plot([float(x['transitions']) for x in rows],[float(x[y]) for x in rows],label=r['name'])
    axes[0].set(ylabel='20-game validation mean score',xlabel='New training transitions')
    axes[1].set(ylabel='Sampled Huber TD loss',xlabel='New training transitions')
    if axes[0].lines:axes[0].legend(fontsize=7,ncol=2)
    fig.savefig(BASE/'curves.png',dpi=120);plt.close(fig)

def summarize(name,question,config,path,elapsed,returncode,reference):
    r=dict(name=name,question=question,config=config,finished_utc=utc(),seconds=elapsed,returncode=returncode)
    if returncode!=0 or not (path/'validation.json').exists():
        r.update(result=f'Incomplete or failed (exit {returncode}); see train.log. No score promoted.',
                 learning='No performance conclusion: this run did not finish evaluation.')
        return r
    d=json.loads((path/'validation.json').read_text())
    meta=json.loads((path/'last/metadata.json').read_text())['experiment']
    s=d['summary'];r.update(score=s['mean_score'],steps=meta['transitions'],total_steps=meta['total_transitions'],summary=s)
    r['result']=(f"100-game validation mean {s['mean_score']:,.1f}, score SD {s['score_std']:,.1f}; "
                 f"2048 reached {100*s['tile_reaching_rates']['2048']:.0f}%; mean length {s['mean_episode_length']:.1f}; "
                 f"{r['steps']:,} new transitions ({r['total_steps']:,} cumulative), {elapsed:.1f} seconds including startup/evaluation.")
    r['learning']='Baseline for comparisons. One training seed does not establish reproducibility.'
    if reference and 'score' in reference:
        import numpy as np
        other=json.loads((BASE/reference['name']/'validation.json').read_text())
        scores={v['seed']:v['score'] for v in other['episodes']}
        paired=np.array([v['score']-scores[v['seed']] for v in d['episodes'] if v['seed'] in scores])
        delta=r['score']-reference['score']
        uncertainty=1.96*paired.std(ddof=1)/(len(paired)**.5)
        r['reference']=reference['name'];r['paired_difference']=float(delta)
        r['paired_normal_95_interval']=[float(delta-uncertainty),float(delta+uncertainty)]
        r['learning']=(f"Against {reference['name']}, mean score changed {delta:+,.1f}; approximate paired 95% interval "
                       f"[{delta-uncertainty:+,.1f}, {delta+uncertainty:+,.1f}]. "
                       'This describes variation across these validation games, not across training seeds; repeated selection can overstate gains.')
        r['learning']+= (' The interval includes zero: no clear gain in this comparison.' if abs(delta)<=uncertainty
                         else ' This is a promising improvement to repeat.' if delta>0 else ' This setting performed worse in this comparison.')
        if r['steps']!=reference['steps'] or config.get('resume')!=reference['config'].get('resume'):
            r['learning']+=' Training budgets/lineage differ; this is not an isolated architecture comparison.'
    return r

def scale_probe(path):
    """Measure, never enforce, response similarity under doubling nonempty tiles."""
    import numpy as np
    import torch
    from rl2048.agents.neural import NeuralAgent
    torch.set_num_threads(2)
    agent=NeuralAgent.load(path/'last')
    rng=np.random.default_rng(81001)
    boards=rng.integers(0,10,(128,16));boards[:,:3]=[2,2,1]
    doubled=np.where(boards>0,boards+1,0)
    b=torch.as_tensor(np.concatenate([boards,doubled]))
    with torch.no_grad():
        x=agent.policy.encode_inputs(b)
        h=agent.policy.layers[:-1](x);q=agent.policy(b)
        cos=torch.nn.functional.cosine_similarity(h[:128],h[128:]).mean().item()
    write_json(path/'scale_probe.json',dict(hidden_cosine_mean=cos,
        q_argmax_agreement=float((q[:128].argmax(1)==q[128:].argmax(1)).float().mean()),
        note='Synthetic boards, unmasked Q argmax, whole-board doubling; not a gameplay metric or proof of generalization. Absolute reward/spawn rules differ.'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--hours',type=float,default=8);p.add_argument('--device',default='mps')
    p.add_argument('--refine',action='store_true',help='Run fixed-parent stability comparisons before further continuations')
    args=p.parse_args();BASE.mkdir(parents=True,exist_ok=True)
    lock=(BASE/'runner.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Another overnight runner owns this directory.')
    if (BASE/'manifest.json').exists():
        manifest=json.loads((BASE/'manifest.json').read_text())
    else:
        start=time.time();manifest=dict(started_utc=utc(start),deadline_utc=utc(start+args.hours*3600),
            started_epoch=start,deadline_epoch=start+args.hours*3600,device=args.device,
            validation_seeds=[6300000,6300099],fresh_test_seeds=[7600000,7600099],
            objective='Compare whole-board MLP encodings, depth, width, optimization; all four-Q-output direct policies.')
        write_json(BASE/'manifest.json',manifest)
    deadline=manifest['deadline_epoch'];results=json.loads((BASE/'results.json').read_text()) if (BASE/'results.json').exists() else []
    stopped=False;child=None
    def on_stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,on_stop);signal.signal(signal.SIGINT,on_stop)
    def status(phase,**extra):
        write_json(BASE/'status.json',dict(phase=phase,pid=os.getpid(),updated_utc=utc(),deadline_utc=manifest['deadline_utc'],completed=len(results),**extra))
    def execute(cmd,log,limit):
        nonlocal child
        with log.open('w') as f:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            status('running',child_pid=child.pid,command=cmd)
            end=min(deadline,time.time()+limit)
            while child.poll() is None and time.time()<end and not stopped and not (BASE/'STOP').exists():
                time.sleep(1)
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=max(.1,min(15,deadline-time.time())))
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            code=child.returncode;child=None
            return code
    # macOS: inhibit idle system sleep only while this runner exists.
    awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    default=dict(architecture='mlp_q',input_encoding='exponents',width=512,depth=2,
        embedding_dim=16,lr=.0003,batch=4096,gamma=.99,reward_mode='score',seed=0,
        target_tau=.005,device=manifest['device'],steps=3_000_000,seconds=600)
    def run(name,question,changes=None,reference=None):
        previous=next((r for r in results if r['name']==name),None)
        if previous:return previous
        if stopped or (BASE/'STOP').exists() or time.time()>deadline-240:return None
        cfg=default|dict(changes or {});cfg['seconds']=min(cfg['seconds'],int(deadline-time.time()-180))
        path=BASE/name
        if path.exists():
            # An interrupted attempt must remain available for diagnosis.
            path.rename(BASE/(name+'_incomplete_'+str(int(time.time()))))
        cmd=[PYTHON,'-m','rl2048.dqn_train','--out',str(path)]
        for key,value in cfg.items():
            if isinstance(value,bool):
                if value:cmd.append('--'+key.replace('_','-'))
            else:cmd += ['--'+key.replace('_','-'),str(value)]
        log=BASE/(name+'.log');t=time.time();code=execute(cmd,log,cfg['seconds']+150)
        r=summarize(name,question,cfg,path,time.time()-t,code,reference)
        if 'score' in r:
            try:
                scale_probe(path)
                probe=json.loads((path/'scale_probe.json').read_text())
                r['learning']+=f" Synthetic doubled-board hidden-feature cosine similarity: {probe['hidden_cosine_mean']:.3f}; descriptive only, not a success metric."
            except Exception as error:r['probe_error']=str(error)
        results.append(r);write_json(BASE/'results.json',results)
        write_json(BASE/(name+'_report.json'),r)
        journal(manifest,results);status('between_experiments')
        print(json.dumps(r),flush=True)
        return r
    def best(rows):
        good=[r for r in rows if r and 'score' in r and r['steps']>=r['config']['steps']]
        return max(good,key=lambda r:r['score']) if good else None
    def settings(r):
        return {k:v for k,v in r['config'].items() if k not in ('seconds','steps','resume')}
    try:
        status('starting');journal(manifest,results)
        enc=[]
        for encoding in ['exponents','one_hot','embedding','relational']:
            enc.append(run('encoding_'+encoding,'Does '+encoding+' improve learning with the same 512-wide, two-layer MLP?',
                           dict(input_encoding=encoding),enc[0] if enc else None))
        winner=best(enc)
        if winner:
            baseline=winner;arch=[baseline];cfg=settings(baseline)
            for width,depth,residual in [(256,2,False),(1024,2,False),(512,4,False),(512,8,False),(512,4,True),(512,8,True),(288,4,False),(192,8,False)]:
                arch.append(run(f'architecture_w{width}_d{depth}_r{int(residual)}',
                    'Test width/depth or residual connections. The narrower deep variants approximately match the baseline parameter budget; exact counts are in checkpoint metadata.',
                    cfg|dict(width=width,depth=depth,residual=residual),baseline))
            winner=best(arch) or winner
            baseline=winner;cfg=settings(baseline);tuning=[baseline]
            for key,values in [('lr',[.0001,.001]),('batch',[1024]),('gamma',[.999,1.]),('target_tau',[.001]),('reward_mode',['corner_snake'])]:
                for value in values:
                    tuning.append(run(f'tune_{key}_{value}',f'Change only {key} to {value} relative to the selected architecture.',cfg|{key:value},baseline))
            winner=best(tuning) or winner
            # Repeated seeds on top two configurations, including each original seed 0.
            candidates=sorted([r for r in tuning if r and 'score' in r and r['steps']>=r['config']['steps']],key=lambda r:r['score'],reverse=True)[:2]
            groups=[]
            for i,candidate in enumerate(candidates):
                group=[candidate]
                for seed in [1,2]:
                    group.append(run(f'repeat_candidate{i}_seed{seed}',
                        'Does the selected configuration reproduce across independent training seeds?',settings(candidate)|dict(seed=seed),candidate))
                good=[r for r in group if r and 'score' in r and r['steps']>=r['config']['steps']]
                if len(good)==3:groups.append((sum(r['score'] for r in good)/3,candidate,[r['name'] for r in good]))
            if groups:
                groups.sort(key=lambda item:item[0],reverse=True);winner=groups[0][1]
                write_json(BASE/'seed_comparison.json',[dict(mean=mean,configuration=r['config'],runs=names) for mean,r,names in groups])
            # Longer continuations refill replay: this is documented, not exact resume.
            continuation=winner;round_number=0;prefix='continuation'
            plan_path=BASE/'refinement_plan.json'
            if args.refine or plan_path.exists():
                if not plan_path.exists():
                    parent=best(results)
                    controls=[r for r in results if 'score' in r and r['config'].get('resume')==str(BASE/parent['name'])
                              and r['steps']>=10_000_000 and r['config']['lr']==parent['config']['lr']
                              and r['config']['batch']==parent['config']['batch']]
                    write_json(plan_path,dict(parent=parent['name'],control=controls[0]['name'] if controls else None,
                        created_utc=utc(),reason='Later continuations declined; compare stability settings from one frozen parent.',
                        deadline_utc=manifest['deadline_utc']))
                plan=json.loads(plan_path.read_text())
                parent=next(r for r in results if r['name']==plan['parent'])
                cfg=settings(parent)|dict(resume=str(BASE/parent['name']),steps=10_000_000,seconds=900)
                control=next((r for r in results if r['name']==plan['control']),None)
                if control is None:
                    control=run('refine_control','Same-parent 10M-transition control before changing stability settings.',cfg,parent)
                refinements=[control]
                for label,change in [('lr_0001',{'lr':.0001}),('lr_00003',{'lr':.00003}),('batch1024',{'batch':1024})]:
                    refinements.append(run('refine_'+label,
                        'From the same frozen parent as the control, test '+str(change)+' for 10M transitions; all other settings unchanged.',
                        cfg|change,control))
                continuation=best(refinements) or parent
                winner=best(results) or winner
                prefix='refined_continuation'
                write_json(BASE/'refinement_selection.json',dict(chosen_continuation=continuation['name'],
                    best_overall=winner['name'],criterion='Complete-budget validation mean; retain the best previous checkpoint too.'))
            adaptive_path=BASE/'adaptive_plan.json'
            if adaptive_path.exists():
                plan=json.loads(adaptive_path.read_text())
                parent=next(r for r in results if r['name']==plan['parent'])
                cfg=settings(parent)|dict(resume=str(BASE/parent['name']),steps=10_000_000,seconds=1200)
                control=run('adaptive_control','Frozen late-stage parent, 10M more transitions with unchanged settings.',cfg,parent)
                branches=[control]
                for label,change in [('gamma_0999',{'gamma':.999}),('gamma_1',{'gamma':1.}),
                                     ('tau_0001',{'target_tau':.001}),('batch1024_double_steps',{'batch':1024,'steps':20_000_000})]:
                    question=('From the same frozen parent, change '+str(change)+'. '
                              'Gamma changes the Bellman target and requires adaptation; this is late-stage fine-tuning. '
                              if label.startswith('gamma') else
                              'From the same frozen parent, change '+str(change)+'. ')
                    if label.startswith('batch'):
                        question+='Give the faster small batch twice as many transitions; compare measured time as well as score, not equal sample efficiency.'
                    branches.append(run('adaptive_'+label,question,cfg|change,control))
                continuation=best(branches) or parent
                winner=best(results) or winner
                prefix='adaptive_continuation'
                write_json(BASE/'adaptive_selection.json',dict(chosen_continuation=continuation['name'],
                    best_overall=winner['name'],criterion='Validation only; complete planned budgets. Small-batch branch has twice the transitions.'))
            exploration_path=BASE/'exploration_plan.json'
            if exploration_path.exists():
                plan=json.loads(exploration_path.read_text())
                parent=next(r for r in results if r['name']==plan['parent'])
                control=next(r for r in results if r['name']==plan['control'])
                cfg=settings(parent)|dict(resume=str(BASE/parent['name']),steps=10_000_000,seconds=900)
                branches=[control]
                for label,change in [('epsilon001',{'epsilon_end':.01}),('epsilon010',{'epsilon_end':.1}),
                                     ('epsilon020',{'epsilon_end':.2}),('capacity200k',{'capacity':200_000})]:
                    branches.append(run('explore_'+label,
                        'From the same frozen parent as the existing 10M control, change '+str(change)+
                        '. Training exploration/replay changes; validation remains greedy legal argmax.',cfg|change,control))
                continuation=best(branches) or parent
                winner=best(results) or winner
                prefix='exploration_continuation'
                write_json(BASE/'exploration_selection.json',dict(chosen_continuation=continuation['name'],
                    best_overall=winner['name'],criterion='Validation only; same-parent 10M budgets; preserve the previous best too.'))
            repeat_path=BASE/'robustness_plan.json'
            if repeat_path.exists():
                plan=json.loads(repeat_path.read_text())
                parent=next(r for r in results if r['name']==plan['parent'])
                cfg=settings(parent)|dict(resume=str(BASE/parent['name']),steps=10_000_000,seconds=900)
                new_runs=[];groups={}
                for epsilon,original in [(.05,'adaptive_continuation_00'),(.01,'explore_epsilon001')]:
                    group=[next(r for r in results if r['name']==original)]
                    for seed in [1,2]:
                        r=run(f'robust_epsilon{epsilon}_seed{seed}',
                            'Repeat the 10M same-parent exploration comparison with a new fine-tuning seed. '
                            'The inherited parent is shared; these are not independent end-to-end training seeds.',
                            cfg|dict(epsilon_end=epsilon,seed=seed),group[0])
                        group.append(r);new_runs.append(r)
                    complete=[r for r in group if r and 'score' in r and r['steps']>=r['config']['steps']]
                    if len(complete)==3:
                        import statistics
                        groups[str(epsilon)]=dict(runs=[r['name'] for r in complete],
                            mean=statistics.mean(r['score'] for r in complete),
                            sample_sd=statistics.stdev(r['score'] for r in complete))
                write_json(BASE/'robustness_comparison.json',dict(groups=groups,
                    note='Three fine-tuning seeds from one shared trained parent; not three independent full-training runs.'))
                group_mode=plan.get('selection_mode')=='group_mean'
                candidates=new_runs
                if group_mode and len(groups)==2:
                    chosen_epsilon=float(max(groups,key=lambda key:groups[key]['mean']))
                    candidates=[r for r in new_runs if r and r['config']['epsilon_end']==chosen_epsilon]
                continuation=best(candidates) or parent
                winner=best(results) or winner
                prefix='consensus_continuation' if group_mode else 'robust_continuation'
                write_json(BASE/'robustness_selection.json',dict(chosen_continuation=continuation['name'],
                    best_overall=winner['name'],criterion=('Select epsilon by the three-seed mean, then its best new fine-tuning run.' if group_mode else
                    'Continue the highest-scoring new fine-tuning run.')+' Retain every previous best for final validation selection.'))
            while not stopped and not (BASE/'STOP').exists() and time.time()<deadline-300:
                cfg=settings(continuation)|dict(resume=str(BASE/continuation['name']),steps=10_000_000,seconds=900)
                r=run(f'{prefix}_{round_number:02}',
                    'Does more training improve the selected MLP? Resume weights and optimizer; refill replay, preserve cumulative epsilon schedule.',cfg,continuation)
                if not r or 'score' not in r:break
                continuation=r
                if r['score']>winner['score']:winner=r
                round_number+=1
            if winner and not stopped and not (BASE/'STOP').exists() and time.time()<deadline-30:
                import hashlib
                selection=dict(run=winner['name'],selected_utc=utc(),criterion='Validation only; final checkpoint',
                    sha256=hashlib.sha256((BASE/winner['name']/'last/policy.pt').read_bytes()).hexdigest(),seeds=list(range(7600000,7600100)))
                write_json(BASE/'selection.json',selection)
                cmd=[PYTHON,str(ROOT/'research/overnight_mlp_test.py'),'--run',str(BASE/winner['name']/'last'),'--out',str(BASE)]
                execute(cmd,BASE/'final_test.log',max(1,deadline-time.time()-5))
        journal(manifest,results)
        status('stopped' if stopped or (BASE/'STOP').exists() else 'complete')
    except Exception as error:
        status('failed',error=repr(error));raise
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
        awake.terminate()

if __name__=='__main__':main()
