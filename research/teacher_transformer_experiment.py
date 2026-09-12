"""Supervise a Transformer on the complete frozen n-tuple afterstate value.

The first stage fits values; matched continuations can add action-distribution
supervision. Neither uses bootstrapped targets, online RL or teacher inputs.
We measure held-out errors, action rankings and complete games before deciding
whether to start the student's own TD learning.
"""
import json
from pathlib import Path
import time

import numpy as np
import torch

from research.afterstate_teacher import digest, validation_metrics, write
from research.neural_afterstate_experiment import evaluate
from rl2048.agents.afterstate_mlp import AfterstateLearner, AfterstateMLPAgent
from rl2048.agents.neural import optimize, tensor_boards
from rl2048.offline_train import training_state
from rl2048.view import save_replay


def action_distillation_loss(prediction,target,gains,legal,temperature):
    """Cross entropy of legal-action distributions derived from scalar values.

    Known immediate rewards are included on both sides. The temperature uses
    points/128 units; soft targets preserve near-tied alternatives. Terminal
    rows and illegal actions contribute no loss or gradient.
    """
    if temperature<=0:raise ValueError('Action temperature must be positive')
    valid=legal.any(1)
    if not valid.any():return prediction.sum()*0
    prediction,target,gains,legal=(x[valid] for x in (prediction,target,gains,legal))
    teacher_q=(target+gains/128)/temperature
    student_q=(prediction+gains/128)/temperature
    teacher_prob=torch.softmax(teacher_q.masked_fill(~legal,-torch.inf),dim=1).detach()
    log_prob=torch.log_softmax(student_q.masked_fill(~legal,-torch.inf),dim=1)
    return -(teacher_prob*torch.where(legal,log_prob,0.)).sum(1).mean()


def stage_pools(states):
    """Indices of fitting boards in four maximum-tile stages.

    Boards store tile exponents: <=256, 512--2048, 4096--8192, and >=16384.
    Stages change sampling only; no stage or handcrafted feature enters the model.
    """
    stage=np.digitize(states.reshape(-1,16).max(1),[8,11,13],right=True)
    return [np.flatnonzero(stage==i) for i in range(4)]


def sample_group_ids(rng,count,batch,pools=None):
    if pools is None:return rng.integers(count,size=batch)
    available=[p for p in pools if len(p)]
    if not available:raise ValueError('Need fitting boards to sample')
    chosen=rng.integers(len(available),size=batch)
    ids=np.empty(batch,dtype=np.int64)
    for i,pool in enumerate(available):
        positions=np.flatnonzero(chosen==i)
        ids[positions]=pool[rng.integers(len(pool),size=len(positions))]
    return ids


def prepare_data(folder, validation_boards=2048):
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text())
    if (not manifest['complete'] or manifest['gamma']!=1.
            or manifest['label_units']!='future raw merge points / 128'
            or digest(folder/'data.npz')!=manifest['data_sha256']):
        raise ValueError('Incomplete, changed, or incompatible teacher data')
    with np.load(folder/'data.npz',allow_pickle=False) as archive:
        data={k:archive[k] for k in archive.files}
    fit=~data['validation'];legal=data['legal'][fit]
    boards=data['afterstates'][fit][legal];labels=data['values'][fit][legal]
    if not len(boards) or not np.isfinite(labels).all():
        raise ValueError('Empty or nonfinite teacher labels')
    held=np.flatnonzero(data['validation'])
    if not len(held):raise ValueError('Need held-out games')
    rng=np.random.default_rng(8841)
    ids=np.sort(rng.choice(held,min(len(held),validation_boards),replace=False))
    validation={k:v[ids] for k,v in data.items()}
    mean=float(labels.mean(dtype=np.float64));scale=max(float(labels.std(dtype=np.float64)),1.)
    return boards,labels,validation,manifest,mean,scale


def train_one(c,seed,steps,out,device):
    if c.get('resume') or c.get('teacher_pretrain'):
        raise ValueError('This experiment starts fresh; explicit distillation continuation is separate')
    if c['gamma']!=1. or c['reward_mode']!='score':
        raise ValueError('Teacher labels use undiscounted future raw merge points')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    boards,labels,validation,manifest,mean,scale=prepare_data(c['teacher_dataset'],c.get('validation_boards',2048))
    c=c|dict(seed=seed,device=device,teacher=True,teacher_only=True,
        value_offset=mean,value_scale=scale,
        initialization='teacher_supervision_continuation' if c.get('resume_distillation') else 'frozen_ntuple_supervision',
        teacher_sha256=manifest['teacher_sha256'],data_sha256=manifest['data_sha256'],
        fitting_afterstates=len(boards),validation_boards_used=len(validation['legal']),
        label_units=manifest['label_units'],supervised_updates_requested=steps,
        target='Frozen n-tuple sum over all patterns and eight views; divided by128',
        search_at_training=False,search_at_evaluation=False,root_slide_enumeration=True)
    write(out/'config.json',c)
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    learner=AfterstateLearner(**c);agent=AfterstateMLPAgent(learner)
    prior_updates=0
    if c.get('resume_distillation'):
        parent=Path(c['resume_distillation']);old=json.loads((parent/'config.json').read_text())
        for key in ('architecture','width','depth','heads','input_encoding','value_offset','value_scale','data_sha256','teacher_sha256'):
            if old[key]!=c[key]:raise ValueError(f'Distillation continuation must preserve {key}')
        saved=torch.load(parent/'distillation_training.pt',map_location='cpu',weights_only=True)
        for key,state in saved.items():getattr(learner,key).load_state_dict(state)
        for group in learner.optimizer.param_groups:group['lr']=c['lr']
        history=json.loads((parent/'last/metadata.json').read_text())['experiment']
        prior_updates=history.get('total_supervised_updates',history['supervised_updates'])
    grouped=c.get('grouped_actions',False);groups=None;pools=None
    if c.get('stage_balanced') and not grouped:
        raise ValueError('Stage-balanced sampling requires grouped boards')
    if grouped:
        with np.load(Path(c['teacher_dataset'])/'data.npz',allow_pickle=False) as archive:
            fit=~archive['validation']
            groups={k:archive[k][fit] for k in ('afterstates','values','gains','legal')}
            if c.get('stage_balanced'):
                pools=stage_pools(archive['states'][fit])
                c['fitting_stage_counts']=[len(pool) for pool in pools]
                c['stage_sampling']='Uniform over nonempty maximum-tile stages, then uniform within each stage'
                write(out/'config.json',c)
    start=time.perf_counter();evaluation_seconds=0.;updates=0
    examples_seen=0
    best_score=-np.inf;best_mse=np.inf;stop_reason=None;curve=[];teacher_curve=[]
    last_policy_check=-1

    def elapsed():return time.perf_counter()-start-evaluation_seconds

    def metadata():
        return c|dict(training_transitions=0,total_transitions=0,transitions=0,
            supervised_updates=updates,total_supervised_updates=prior_updates+updates,
            supervised_examples_seen=examples_seen,
            training_seconds=elapsed(),parameter_count=sum(p.numel() for p in learner.policy.parameters()))

    def check(policy=False):
        nonlocal evaluation_seconds,best_score,best_mse,last_policy_check
        before=time.perf_counter();training_seconds=elapsed()
        learner.policy.eval()
        metrics=validation_metrics(learner.policy,validation,device,c['batch'])
        metrics['normalized_value_mse']=metrics['value_mse']/scale**2
        teacher_curve.append(dict(updates=updates,seconds=training_seconds,**metrics))
        write(out/'teacher_curve.json',teacher_curve)
        if metrics['value_mse']<best_mse:
            best_mse=metrics['value_mse'];agent.save(out/'best_value',metadata())
        if policy:
            result=evaluate(learner,range(c['monitor_seed_start'],c['monitor_seed_start']+c['monitor_games']))
            curve.append(result['summary']|dict(transitions=0,updates=updates,
                training_seconds=training_seconds,evaluation_transitions=result['summary']['transitions']))
            write(out/'curve.json',curve);last_policy_check=updates
            if result['summary']['mean_score']>best_score:
                best_score=result['summary']['mean_score'];agent.save(out/'best',metadata())
                write(out/'best_monitor.json',result|dict(supervised_updates=updates))
        agent.save(out/'last',metadata())
        torch.save(training_state(learner),out/'distillation_training.pt')
        print('TEACHER',json.dumps(teacher_curve[-1]|dict(policy_mean=curve[-1]['mean_score'] if policy else None)),flush=True)
        if c.get('monitor_report'):
            import subprocess,sys
            subprocess.run([sys.executable,'-m',c['monitor_report']],check=False)
        evaluation_seconds+=time.perf_counter()-before
        learner.policy.train()

    def stopped():return c.get('stop_file') and Path(c['stop_file']).exists()

    check(policy=True)
    for update in range(steps):
        if stopped():stop_reason='stop_file';break
        if elapsed()>=c['max_training_seconds']:stop_reason='training_time_budget';break
        if grouped:
            ids=sample_group_ids(rng,len(groups['legal']),c['batch'],pools)
            x=tensor_boards(groups['afterstates'][ids].reshape(-1,16),device)
            y=torch.as_tensor(groups['values'][ids],device=device)
            legal=torch.as_tensor(groups['legal'][ids],device=device)
            prediction=learner.policy(x).reshape(-1,4)
            loss=(((prediction-y)/scale).square()[legal]).mean()
            if c.get('action_loss_weight',0):
                loss=loss+c['action_loss_weight']*action_distillation_loss(prediction,y,
                    torch.as_tensor(groups['gains'][ids],device=device),legal,c['action_temperature'])
            examples_seen+=int(legal.sum())
        else:
            ids=rng.integers(len(boards),size=c['batch'])
            x=tensor_boards(boards[ids],device);y=torch.as_tensor(labels[ids],device=device)
            # Fixed residual scaling leaves the regression optimum unchanged.
            loss=((learner.policy(x)-y)/scale).square().mean()
            examples_seen+=len(ids)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite distillation update')
        optimize(learner.optimizer,loss,learner.policy.parameters());updates=update+1
        if updates%c['validation_interval']==0:
            check(policy=updates%c['policy_monitor_interval']==0)
    if not stopped() and last_policy_check!=updates:check(policy=True)
    agent.save(out/'last',metadata())
    # A future online TD continuation starts with a synchronized target and
    # fresh Adam moments. The separate file above preserves supervised state.
    learner.target.load_state_dict(learner.policy.state_dict());learner.optimizer.state.clear()
    torch.save(training_state(learner),out/'training.pt')
    result=metadata()|dict(completed_budget=updates==steps,stop_reason=stop_reason,
        monitoring_seconds=evaluation_seconds,complete=False,mean_score=None,
        best_monitor_score=best_score,final_validation=teacher_curve[-1],
        online_training=False,teacher_data_in_online_replay=False)
    if stopped():return result
    learner.policy.eval()
    seeds=range(c['final_seed_start'],c['final_seed_start']+100)
    final=evaluate(learner,seeds);write(out/'evaluation.json',final)
    selected=AfterstateMLPAgent.load(out/'best',device)
    chosen=evaluate(selected.learner,seeds);write(out/'best_selection.json',chosen|dict(
        checkpoint=str((out/'best').resolve()),selection='Highest128-game monitor score during supervised training'))
    replay=save_replay(selected,out/'replay_best.html',seed=8930100)
    write(out/'replay_selection.json',dict(device=device,result=replay))
    return result|final['summary']|dict(training_transitions=0,transitions=0,
        evaluation_transitions=final['summary']['transitions'],supervised_updates=updates,
        depth=c['depth'],final_selection_mean_score=final['summary']['mean_score'])
