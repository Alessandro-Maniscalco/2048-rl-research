"""Online Double DQN: plain/embedded MLP vs standard Transformer; n=1/3.

Run: python -m research.transformer_td_experiment --out runs/research/transformer_td
All new models start from scratch: no engineered features, teacher, or search.
The core factorial changes architecture, input encoding, and TD horizon; two
extra variants change only the reward transform. This is an initial screen.
"""
import argparse
from copy import deepcopy
import csv
import itertools
import json
from pathlib import Path
import time
import numpy as np
import torch
from rl2048.agents.dqn import DQN
from rl2048.agents.neural import NeuralAgent, tensor_boards
from rl2048.agents.ntuple import row_tables
from rl2048.offline_train import tensor_batch, training_state
from rl2048.n_step import NStepReplay
from rl2048.vector_game import VectorGame, legal_masks
from rl2048.rewards import learning_rewards
from rl2048.afterstate_compare import evaluate_batch
from rl2048.view import save_replay


def write_json(path, data):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2));temp.replace(path)


def config(architecture, encoding, n, reward='score'):
    return dict(architecture=architecture,input_encoding=encoding,n=n,reward_mode=reward,
        width=64 if architecture=='transformer_q' else 256,depth=2,heads=4,embedding_dim=16,
        lr=1e-4,gamma=.99,target_tau=.005,double=True,envs=128,batch=512,
        warmup=4096,capacity=200000,epsilon_end=.1,epsilon_steps=100000)


def configs():
    rows=[config(a,e,n) for a,e,n in itertools.product(
        ('mlp_q','transformer_q'),('exponents','embedding'),(1,3))]
    rows += [config(a,'embedding',3,'log_score') for a in ('mlp_q','transformer_q')]
    return rows


def name(c):
    prefix='qr_dqn_' if c.get('algorithm')=='qr_dqn' else ''
    return prefix+f"{c['architecture']}_{c['input_encoding']}_n{c['n']}_{c['reward_mode']}"


def benchmark(out):
    """Time collection-sized inference AND replay updates, including transfers."""
    rows=[]
    for arch,device in itertools.product(('mlp_q','transformer_q'),('cpu','mps')):
        if device=='mps' and not torch.backends.mps.is_available():continue
        torch.manual_seed(0)
        c=config(arch,'embedding',3);learner=DQN(device=device,**c)
        b=np.random.default_rng(0).integers(0,12,(512,16),dtype=np.uint8)
        batch=dict(states=b,next_states=b,actions=np.arange(512)%4,rewards=np.ones(512,np.float32),
            terminated=np.zeros(512,bool),next_masks=np.ones((512,4),bool),discounts=np.full(512,.99**3,np.float32))
        elapsed=[]
        for i in range(25):
            t=time.perf_counter()
            with torch.no_grad():learner.policy(tensor_boards(b[:128],device)).cpu().numpy()
            loss=learner.update(tensor_batch(batch,device))
            float(loss['q_loss'].cpu()) # synchronize GPU before stopping timer
            if i>=5:elapsed.append(time.perf_counter()-t)
        rows.append(dict(architecture=arch,device=device,seconds_per_cycle=float(np.median(elapsed)),
            parameter_count=sum(p.numel() for p in learner.policy.parameters())))
        print('BENCHMARK',json.dumps(rows[-1]),flush=True)
    write_json(out/'benchmark.json',rows)
    return {arch:min((r for r in rows if r['architecture']==arch),key=lambda r:r['seconds_per_cycle'])['device']
            for arch in ('mlp_q','transformer_q')}


def evaluate_model(model, width, device, seeds):
    algorithm='qr_dqn' if getattr(model,'architecture',None) in ('transformer_qr','mlp_qr') else 'double_dqn'
    agent=NeuralAgent(deepcopy(model).to(device).eval(),algorithm,device,width)
    @torch.no_grad()
    def decision(b):
        q=agent.policy(tensor_boards(b,device)).cpu().numpy()
        return np.where(legal_masks(b,*row_tables()),q,-np.inf)
    result=evaluate_batch(None,seeds,decision=decision)
    assert result['summary']['complete'] and not result['summary']['truncated_episodes']
    result['summary']['tile_reaching_rates']={str(t):float(np.mean([e['max_tile']>=t for e in result['episodes']]))
        for t in (128,256,512,1024,2048,4096,8192)}
    return result


def train_one(c,seed,steps,out,device):
    if c.get('augment_symmetry',False) and c['reward_mode']!='score':
        raise ValueError('This symmetry experiment requires orientation-invariant score rewards')
    if c.get('priority_alpha') is not None and c.get('algorithm')!='qr_dqn':
        raise ValueError('This prioritized-replay comparison requires QR-DQN weighted loss')
    if c.get('priority_alpha') is not None and c.get('priority_beta_steps',steps)<=0:
        raise ValueError('The replay importance-weight schedule needs positive steps')
    out.mkdir(parents=True,exist_ok=False)
    c=c|dict(seed=seed,steps=steps,device=device,initialization='resume' if c.get('resume') else 'scratch',teacher=False,
        search_at_training=False,search_at_evaluation=False,off_policy_correction=False)
    write_json(out/'config.json',c)
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    # Separate stream: augmentation must not consume exploration/replay randomness.
    augmentation_rng=np.random.default_rng([seed,104729]) if c.get('augment_symmetry',False) else None
    algorithm=c.get('algorithm','double_dqn')
    if algorithm=='qr_dqn':
        from rl2048.agents.qr_dqn import QRDQN
        learner=QRDQN(**c)
    elif algorithm=='double_dqn':
        learner=DQN(**c)
    else:
        raise ValueError(f'Unknown online Q algorithm: {algorithm}')
    prior_steps=0
    if c.get('resume'):
        previous=Path(c['resume'])
        old=json.loads((previous/'config.json').read_text())
        if old.get('algorithm','double_dqn')!=algorithm:
            raise ValueError('Resume must preserve the learning algorithm')
        if algorithm=='qr_dqn' and old.get('num_quantiles',51)!=c.get('num_quantiles',51):
            raise ValueError('Resume must preserve the quantile count')
        for key in ('architecture','input_encoding','width','depth','heads','attention_entropy','weight_decay'):
            if old.get(key,0)!=c.get(key,0):raise ValueError(f'Resume must preserve {key}')
        state=torch.load(previous/'training.pt',map_location='cpu',weights_only=True)
        for key,weights in state.items():getattr(learner,key).load_state_dict(weights)
        for group in learner.optimizer.param_groups:group['lr']=c['lr']
        old_meta=json.loads((previous/'last/metadata.json').read_text())['experiment']
        prior_steps=old_meta.get('total_transitions',old_meta['transitions'])
    env=VectorGame(c['envs'],c.get('collection_seed',seed))
    replay=NStepReplay(c['capacity'],c['envs'],c['n'],c['gamma'],
        priority_alpha=c.get('priority_alpha'),priority_beta=c.get('priority_beta_start',.4),
        priority_epsilon=c.get('priority_epsilon',1e-6))
    progress=[];episodes=[];curve=[];updates=0;start=time.perf_counter();eval_seconds=0
    game_seconds=0.;update_seconds=0.
    processed_steps=0;stop_reason=None;best_monitor=-float('inf')

    def monitor(step):
        nonlocal eval_seconds,best_monitor
        elapsed_training=time.perf_counter()-start-eval_seconds
        monitor_start=time.perf_counter()
        first=c.get('monitor_seed_start',8640000)
        check=evaluate_model(learner.policy,c['width'],c.get('evaluation_device','cpu'),range(first,first+c.get('monitor_games',20)))
        curve.append(check['summary']|dict(transitions=step,updates=updates,
            training_seconds=elapsed_training,evaluation_transitions=check['summary']['transitions']))
        if c.get('attention_probe'):
            probe=np.load(c['attention_probe'])['boards']
            with torch.no_grad():_,entropy=learner.policy.forward_with_entropy(tensor_boards(probe,device))
            curve[-1]['attention_entropy_by_layer']=entropy.cpu().tolist()
            curve[-1]['attention_entropy_fraction']=float(entropy.mean().cpu())/np.log(16)
            if learner.entropy_control is not None:
                curve[-1]['attention_alpha_by_layer']=learner.entropy_control.log_alpha.detach().exp().cpu().tolist()
        write_json(out/'curve.json',curve)
        if check['summary']['mean_score']>best_monitor:
            best_monitor=check['summary']['mean_score']
            if c.get('save_best',False):
                NeuralAgent(learner.policy,algorithm,device,c['width']).save(out/'best',
                    c|dict(transitions=step,total_transitions=prior_steps+step,updates=updates))
                write_json(out/'best_monitor.json',check|dict(transitions=step))
        if c.get('save_training_at_monitor',False):
            NeuralAgent(learner.policy,algorithm,device,c['width']).save(out/'last',
                c|dict(transitions=step,total_transitions=prior_steps+step,updates=updates))
            torch.save(training_state(learner),out/'training.pt')
        if c.get('save_monitor_checkpoints',False):
            folder=out/'checkpoints'/f'{step:09d}'
            NeuralAgent(learner.policy,algorithm,device,c['width']).save(folder,c|dict(transitions=step,updates=updates))
            write_json(folder/'monitor_evaluation.json',check)
        if progress:
            with (out/'progress.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(progress[0]));w.writeheader();w.writerows(progress)
        if c.get('monitor_report'):
            import subprocess,sys
            subprocess.run([sys.executable,'-m',c['monitor_report']],check=False)
        eval_seconds+=time.perf_counter()-monitor_start

    if c.get('monitor_initial',False):monitor(0)
    for step in range(c['envs'],steps+1,c['envs']):
        if c.get('stop_file') and Path(c['stop_file']).exists():
            stop_reason='stop_file';break
        if c.get('max_training_seconds') and time.perf_counter()-start-eval_seconds>=c['max_training_seconds']:
            stop_reason='training_time_budget';break
        states=env.boards.copy();masks=env.masks()
        epsilon=1-(1-c['epsilon_end'])*min(1,(step+prior_steps)/c['epsilon_steps'])
        with torch.no_grad():q=learner.policy(tensor_boards(states,device)).cpu().numpy()
        actions=np.where(masks,q,-np.inf).argmax(1)
        random_actions=np.where(masks,rng.random(masks.shape),-np.inf).argmax(1)
        actions=np.where(rng.random(c['envs'])<epsilon,random_actions,actions)
        game_start=time.perf_counter()
        raw,next_states,term,trunc,completed=env.step(actions)
        game_seconds+=time.perf_counter()-game_start
        rewards=learning_rewards(raw,states,next_states,term,c['gamma'],c['reward_mode'],
            scale=c.get('shaping_scale',10.))
        replay.add(states=states,actions=actions,rewards=rewards,next_states=next_states,masks=masks,
            next_masks=legal_masks(next_states,env.rows,env.row_rewards),terminated=term,truncated=trunc)
        processed_steps=step
        for score,length,tile in completed[completed[:,1]>0]:
            episodes.append(dict(transitions=step,score=int(score),length=int(length),max_tile=int(tile)))
        if step>=c['warmup'] and replay.size>=c['batch']:
            update_start=time.perf_counter()
            if c.get('priority_alpha') is not None:
                beta_start=c.get('priority_beta_start',.4)
                replay.replay.beta=beta_start+(1-beta_start)*min(1,(step+prior_steps)/c.get('priority_beta_steps',steps))
            batch=replay.sample(c['batch'],rng)
            if augmentation_rng is not None:
                from rl2048.symmetry import transform_batch
                batch=transform_batch(batch,rotation=int(augmentation_rng.integers(4)),
                    reflect=bool(augmentation_rng.integers(2)))
            losses=learner.update(tensor_batch(batch,device));updates+=1
            if '_priorities' in losses:
                replay.update_priorities(batch['indices'],losses.pop('_priorities').cpu().numpy())
            if c.get('dense_metrics',False):
                # Synchronize for accurate GPU timing and log every update.
                loss_values={k:float(v.cpu()) for k,v in losses.items()}
                update_seconds+=time.perf_counter()-update_start
            if c.get('dense_metrics',False) or updates%64==0 or step==steps:
                row=dict(transitions=step,updates=updates,training_seconds=time.perf_counter()-start-eval_seconds,
                    epsilon=epsilon,mean_training_score_100=float(np.mean([e['score'] for e in episodes[-100:]])) if episodes else None,
                    **(loss_values if c.get('dense_metrics',False) else {k:float(v.cpu()) for k,v in losses.items()}))
                if c.get('dense_metrics',False):
                    row.update(mean_live_score_128=float(env.scores.mean()),
                        mean_live_moves_128=float(env.lengths.mean()),
                        mean_completed_score_128=float(np.mean([e['score'] for e in episodes[-128:]])) if episodes else None,
                        completed_window_count=min(128,len(episodes)),
                        mean_immediate_merge_points_128=float(raw.mean()),
                        cpu_game_step_seconds=game_seconds,update_seconds=update_seconds)
                if c.get('priority_alpha') is not None:
                    row.update(replay_beta=replay.replay.beta,
                        maximum_replay_priority=replay.replay.maximum_priority)
                progress.append(row)
                if updates%64==0 or step==steps:print('TRAIN',name(c),seed,json.dumps(row),flush=True)
        interval=c.get('monitor_interval')
        if (interval and (step%interval==0 or step==steps)) or (not interval and step in (steps//2,steps)):
            monitor(step)
            if c.get('plateau'):
                from research.plateau import plateau_status
                plateau=plateau_status(curve,**c['plateau'])
                write_json(out/'plateau.json',plateau|dict(transitions=step,total_transitions=prior_steps+step))
                if plateau['reached']:
                    stop_reason='policy_plateau'
                    break
    replay.flush()
    training_seconds=time.perf_counter()-start-eval_seconds
    metadata=c|dict(transitions=processed_steps,total_transitions=prior_steps+processed_steps,
        prior_transitions=prior_steps,updates=updates,training_seconds=training_seconds,
        completed_budget=processed_steps==steps,stop_reason=stop_reason,
        best_monitor_score=best_monitor if np.isfinite(best_monitor) else None,
        parameter_count=sum(p.numel() for p in learner.policy.parameters()),replay_records=replay.size,
        monitoring_seconds=eval_seconds,cpu_game_step_seconds=game_seconds,
        update_seconds=update_seconds if c.get('dense_metrics',False) else None)
    NeuralAgent(learner.policy,algorithm,device,c['width']).save(out/'last',metadata)
    torch.save(training_state(learner),out/'training.pt')
    for filename,rows in (('progress',progress),('episodes',episodes)):
        if rows:
            with (out/f'{filename}.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if stop_reason=='stop_file':
        return metadata|dict(name=name(c),seed=seed,config=c,training_transitions=processed_steps,
            evaluation_transitions=0,complete=False,mean_score=None)
    # Validate the actual saved artifact, not only its in-memory precursor.
    saved=NeuralAgent.load(out/'last')
    first=c.get('final_seed_start',8650000)
    result=evaluate_model(saved.policy,c['width'],c.get('evaluation_device','cpu'),range(first,first+100))
    write_json(out/'evaluation.json',result);write_json(out/'curve.json',curve)
    for filename,rows in (('progress',progress),('episodes',episodes)):
        if rows:
            with (out/f'{filename}.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return metadata|result['summary']|dict(name=name(c),seed=seed,config=c,
        training_transitions=processed_steps,evaluation_transitions=result['summary']['transitions'])


def report(out,results,status):
    write_json(out/'results.json',dict(status=status,runs=results))
    lines=['# Plain inputs, Transformers, and online n-step TD','',
        'Initial screen from scratch. No engineered relational inputs, teacher, planning, corner or snake shaping.',
        'All variants receive the same transition and update budgets. Architectures have different parameter counts; runtime is reported.',
        'Each checkpoint plays 100 fixed held-out seeds. These become validation results if used to choose further variants.',
        'Three training seeds quantify initialization and experience variation. Short runs cannot establish the best final architecture.',
        'Log reward = log2(1 + merge points) / log2(129); score reward = merge points / 128.',
        'n-step replay uses observed exploratory trajectories without off-policy correction; no sequence crosses resets.','',
        f'Status: {status}','','| Experiment | Seeds | Mean raw score ± SD across training seeds | Training seconds per seed |',
        '|---|---:|---:|---:|']
    for key in dict.fromkeys(r['name'] for r in results):
        group=[r for r in results if r['name']==key];scores=[r['mean_score'] for r in group]
        sd=np.std(scores,ddof=1) if len(scores)>1 else 0
        lines.append(f"| {key} | {len(group)} | {np.mean(scores):.1f} ± {sd:.1f} | {np.mean([r['training_seconds'] for r in group]):.1f} |")
    lines+=['','## Matched contrasts','',
        'Differences below are mean raw score changes on the same held-out game seeds, averaged over matching training seeds. They describe this short budget only.','']
    contrasts=[]
    for a,e in itertools.product(('mlp_q','transformer_q'),('exponents','embedding')):
        contrasts.append((f'{a}, {e}: three-step minus one-step',name(config(a,e,1)),name(config(a,e,3))))
    for a,n in itertools.product(('mlp_q','transformer_q'),(1,3)):
        contrasts.append((f'{a}, n={n}: embedding minus simple inputs',name(config(a,'exponents',n)),name(config(a,'embedding',n))))
    for e,n in itertools.product(('exponents','embedding'),(1,3)):
        contrasts.append((f'{e}, n={n}: Transformer minus MLP',name(config('mlp_q',e,n)),name(config('transformer_q',e,n))))
    for a in ('mlp_q','transformer_q'):
        contrasts.append((f'{a}, embedding n=3: logarithmic minus proportional reward',name(config(a,'embedding',3)),name(config(a,'embedding',3,'log_score'))))
    for label,left,right in contrasts:
        l={r['seed']:r for r in results if r['name']==left};r={r['seed']:r for r in results if r['name']==right}
        shared=sorted(l.keys()&r.keys())
        if shared:
            diffs=[r[s]['mean_score']-l[s]['mean_score'] for s in shared]
            lines.append(f'- {label}: {np.mean(diffs):+.1f} points across {len(shared)} paired training seeds; per-seed differences {", ".join(f"{x:+.1f}" for x in diffs)}.')
    lines+=['','## Experiment journal','']
    for r in results:
        ref=None
        if r['reward_mode']=='log_score':
            ref=name(config(r['architecture'],r['input_encoding'],r['n']))
        elif r['n']==3:
            ref=name(config(r['architecture'],r['input_encoding'],1))
        elif r['input_encoding']=='embedding':
            ref=name(config(r['architecture'],'exponents',r['n']))
        elif r['architecture']=='transformer_q':
            ref=name(config('mlp_q',r['input_encoding'],r['n']))
        match=next((x for x in results if x['name']==ref and x['seed']==r['seed']),None)
        learning=(f"Learning: {r['mean_score']-match['mean_score']:+.2f} raw points versus {ref} with the same training seed. "
            'This is an early-learning contrast; assess consistency using all three training seeds.' if match else
            'Learning: this plain-input one-step run establishes the reference for the later single-factor comparisons.')
        lines += [f"### {r['name']} · seed {r['seed']}",
            f"Testing: {r['architecture']}, {r['input_encoding']} inputs, {r['n']}-step TD, {r['reward_mode']} reward.",
            f"Result: mean raw score {r['mean_score']:.2f}, mean length {r['mean_length']:.1f}, 2048 rate {r['reaching_2048']:.1%}; {r['parameter_count']:,} parameters; {r['transitions']:,} evaluation transitions.",
            learning,'']
    (out/'journal.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=131072)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);p.add_argument('--limit',type=int,default=10)
    p.add_argument('--benchmark-only',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    if args.steps<8192 or args.steps%256:raise ValueError('Use >=8192 steps divisible by 256.')
    torch.set_num_threads(2)
    devices=benchmark(args.out)
    if args.benchmark_only:return
    results=[];report(args.out,results,'running')
    for c in configs()[:args.limit]:
        for seed in args.seeds:
            r=train_one(c,seed,args.steps,args.out/name(c)/f'seed{seed}',devices[c['architecture']])
            results.append(r);report(args.out,results,'running')
            print('RESULT',r['name'],seed,r['mean_score'],flush=True)
    # Fixed fresh seed, not chosen for an unusually high-scoring game.
    best=max(results,key=lambda r:np.mean([x['mean_score'] for x in results if x['name']==r['name']]))
    agent=NeuralAgent.load(args.out/best['name']/f"seed{best['seed']}"/'last')
    agent.display_name=f"{best['name']} · initial screen · training seed {best['seed']}"
    replay=save_replay(agent,args.out/'replay.html',seed=8660000)
    write_json(args.out/'replay_selection.json',dict(experiment=best['name'],training_seed=best['seed'],result=replay,
        selection='Highest mean across training seeds; first training seed checkpoint. Fresh fixed replay seed.'))
    report(args.out,results,'finished')


if __name__=='__main__':main()
