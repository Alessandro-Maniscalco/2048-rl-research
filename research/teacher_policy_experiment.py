"""Supervised teacher policy and value heads, without Q bootstrapping.

Unlike the afterstate scalar student, this policy receives the current board
once and predicts move preferences directly. The auxiliary value estimates
the best teacher root Q. Full-game monitoring, label validation and fitting
use separate games. This is behavior distillation, not online reinforcement
learning, and its move logits must never be presented as calibrated Q-values.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from research.afterstate_teacher import write,digest
from research.teacher_transformer_experiment import prepare_data,stage_pools
from rl2048.agents.neural import NeuralAgent,tensor_boards,optimize,policy_logits
from rl2048.agents.ntuple import row_tables
from rl2048.agents.teacher_policy_transformer import TeacherPolicyTransformer,copy_value_encoder
from rl2048.afterstate_compare import evaluate_batch
from rl2048.vector_game import legal_masks
from rl2048.view import save_replay


def policy_value_loss(output,teacher_q,legal,scale,temperature,value_weight):
    if scale<=0 or temperature<=0 or value_weight<0:
        raise ValueError('Positive scale/temperature and nonnegative value weight required')
    valid=legal.any(1)
    if not valid.any():return output.sum()*0
    output,teacher_q,legal=(x[valid] for x in (output,teacher_q,legal))
    q=teacher_q.masked_fill(~legal,-torch.inf)
    teacher_probs=torch.softmax(q/temperature,1).detach()
    log_probs=torch.log_softmax(output[:,:4].masked_fill(~legal,-torch.inf),1)
    ce=-(teacher_probs*torch.where(legal,log_probs,0.)).sum(1).mean()
    value_mse=((output[:,4]-q.max(1).values)/scale).square().mean()
    return ce+value_weight*value_mse


@torch.no_grad()
def validate(model,data,device,batch,temperature,scale):
    result=[]
    for i in range(0,len(data['states']),batch):
        result.append(model(tensor_boards(data['states'][i:i+batch],device)).cpu().numpy())
    output=np.concatenate(result);legal=data['legal']
    q=np.where(legal,data['values']+data['gains']/128,-np.inf)
    choices=np.where(legal,output[:,:4],-np.inf).argmax(1)
    best=q.max(1);chosen=q[np.arange(len(q)),choices]
    mse=float(np.mean((output[:,4]-best)**2))
    ce=float(policy_value_loss(torch.from_numpy(output),torch.from_numpy(q),
        torch.from_numpy(legal),scale,temperature,0))
    z=q/temperature;z-=z.max(1,keepdims=True)
    probabilities=np.exp(z);probabilities/=probabilities.sum(1,keepdims=True)
    entropy=float(-(probabilities*np.log(np.clip(probabilities,1e-38,None))).sum(1).mean())
    return dict(value_mse=mse,normalized_value_mse=mse/scale**2,
        policy_cross_entropy=ce,teacher_entropy=entropy,policy_kl=ce-entropy,
        teacher_action_agreement=float(np.isclose(chosen,best,atol=1e-5,rtol=1e-6).mean()),
        mean_teacher_action_gap=float((best-chosen).mean()))


def evaluate(agent,seeds):
    @torch.no_grad()
    def decision(boards):
        logits=policy_logits(agent.policy,tensor_boards(boards,agent.device)).cpu().numpy()
        masks=agent.policy_mask(boards,legal_masks(boards,*row_tables()))
        return np.where(masks,logits,-np.inf)
    result=evaluate_batch(None,seeds,decision=decision)
    assert result['summary']['complete'] and not result['summary']['truncated_episodes']
    result['summary']['tile_reaching_rates']={str(t):float(np.mean([e['max_tile']>=t for e in result['episodes']]))
        for t in (128,256,512,1024,2048,4096,8192)}
    result['summary']['action_selection']='direct legal argmax of policy logits; no root slides or future search'
    if agent.spawn_safety:
        result['summary']['action_selection']='highest policy logit among legal moves with minimum exact next-spawn death risk'
        result['summary']['spawn_safety']=True
    return result


def train_one(c,seed,steps,out,device):
    if c.get('resume_distillation') and c.get('encoder_checkpoint'):
        raise ValueError('Choose full policy continuation or encoder transfer, not both')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    _,_,validation,manifest,_,_=prepare_data(c['teacher_dataset'],c.get('validation_boards',2048))
    with np.load(Path(c['teacher_dataset'])/'data.npz',allow_pickle=False) as archive:
        fit=~archive['validation'];data={k:archive[k][fit] for k in ('states','values','gains','legal')}
    q=np.where(data['legal'],data['values']+data['gains']/128,-np.inf)
    target=q.max(1)
    if not np.isfinite(target).all():raise ValueError('Fitting data contain terminal or nonfinite targets')
    mean=float(target.mean(dtype=np.float64));scale=max(float(target.std(dtype=np.float64)),1.)
    available_boards=len(target)
    subset_ids=None
    if c.get('fit_subset_size'):
        if not c.get('diagnostic_only'):raise ValueError('Fixed subset is a fitting diagnostic, not a competitive score run')
        count=c['fit_subset_size'];pools=stage_pools(data['states']);active=[p for p in pools if len(p)]
        if count<1 or count%len(active):raise ValueError('Subset size must be divisible by the number of nonempty stages')
        sample_rng=np.random.default_rng(c.get('fit_subset_seed',11904))
        subset_ids=np.concatenate([sample_rng.choice(p,min(len(p),count//len(active)),replace=False) for p in active])
        data={k:v[subset_ids] for k,v in data.items()};q=q[subset_ids];target=target[subset_ids]
    c=c|dict(seed=seed,device=device,teacher_only=True,teacher=True,
        value_offset=mean,value_scale=scale,fitting_boards=len(target),
        available_fitting_boards=available_boards,
        validation_boards_used=len(validation['states']),data_sha256=manifest['data_sha256'],
        teacher_sha256=manifest['teacher_sha256'],value_target='maximum legal teacher Q: merge points/128 + full n-tuple value/128',
        initialization=('policy_continuation' if c.get('resume_distillation') else
                        'value_encoder_transfer' if c.get('encoder_checkpoint') else 'scratch'),
        online_training=False,root_slide_enumeration=False,search_at_training=False,search_at_evaluation=False)
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=TeacherPolicyTransformer(c['width'],c['input_encoding'],c['depth'],c['heads'],mean,scale).to(device)
    if c.get('encoder_checkpoint'):
        source=Path(c['encoder_checkpoint']);meta=json.loads((source/'metadata.json').read_text())
        if meta['architecture']!='afterstate_sym_transformer':raise ValueError('Expected a supervised afterstate Transformer')
        c['encoder_sha256']=digest(source/'policy.pt')
        c['encoder_pretraining_updates']=meta['experiment']['supervised_updates']
        c['encoder_pretraining_seconds']=meta['experiment']['training_seconds']
        copy_value_encoder(model,torch.load(source/'policy.pt',map_location='cpu',weights_only=True))
    agent=NeuralAgent(model,'teacher_policy_value',device,c['width'])
    agent.display_name='Teacher-supervised Transformer policy and value'
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    if c.get('resume_distillation'):
        parent=Path(c['resume_distillation']);old=json.loads((parent/'config.json').read_text())
        for key in ('width','depth','heads','input_encoding','value_offset','value_scale','data_sha256','teacher_sha256'):
            if old[key]!=c[key]:raise ValueError(f'Policy continuation must preserve {key}')
        state=torch.load(parent/'distillation_training.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(state['policy']);optimizer.load_state_dict(state['optimizer'])
        for group in optimizer.param_groups:group['lr']=c['lr']
        history=json.loads((parent/'last/metadata.json').read_text())['experiment']
        c['prior_supervised_updates']=history['total_supervised_updates']
        c['parent_training_seconds']=history['training_seconds']
    if subset_ids is not None:write(out/'fitting_subset.json',dict(indices_in_full_fitting_array=subset_ids.tolist(),
        note='Only original fitting boards; stage-stratified diagnostic subset. Value normalization still uses the original full fitting partition.'))
    write(out/'config.json',c)
    start=time.perf_counter();evaluation_seconds=0.;updates=0;best=-np.inf;stop_reason=None
    curve=[];teacher_curve=[];fit_curve=[];last_policy=-1;fit_goal_met=False
    fit_sample=None
    if c.get('fit_monitor_boards') or c.get('fit_subset_size'):
        n=min(c.get('fit_monitor_boards',512),len(target))
        ids=np.sort(np.random.default_rng(11905).choice(len(target),n,replace=False))
        fit_sample={k:v[ids] for k,v in data.items()}
    def elapsed():return time.perf_counter()-start-evaluation_seconds
    def metadata():return c|dict(supervised_updates=updates,
        total_supervised_updates=c.get('prior_supervised_updates',c.get('encoder_pretraining_updates',0))+updates,
        supervised_examples_seen=updates*c['batch'],training_seconds=elapsed(),
        training_transitions=0,transitions=0,parameter_count=sum(p.numel() for p in model.parameters()))
    def stopped():return c.get('stop_file') and Path(c['stop_file']).exists()
    def check(policy=False):
        nonlocal evaluation_seconds,best,last_policy,fit_goal_met
        before=time.perf_counter();seconds=elapsed();model.eval()
        metrics=validate(model,validation,device,c['batch'],c['action_temperature'],scale)
        teacher_curve.append(dict(updates=updates,seconds=seconds,**metrics));write(out/'teacher_curve.json',teacher_curve)
        if fit_sample is not None:
            fitted=validate(model,fit_sample,device,c['batch'],c['action_temperature'],scale)
            fit_curve.append(dict(updates=updates,seconds=seconds,**fitted));write(out/'fit_curve.json',fit_curve)
            if c.get('fit_kl_stop') is not None:
                fit_goal_met=(fitted['policy_kl']<=c['fit_kl_stop'] and
                    fitted['teacher_action_agreement']>=c.get('fit_agreement_stop',.95))
        if policy:
            result=evaluate(agent,range(c['monitor_seed_start'],c['monitor_seed_start']+c['monitor_games']))
            curve.append(result['summary']|dict(updates=updates,transitions=0,training_seconds=seconds,
                evaluation_transitions=result['summary']['transitions']));write(out/'curve.json',curve);last_policy=updates
            if result['summary']['mean_score']>best:
                best=result['summary']['mean_score'];agent.save(out/'best',metadata());write(out/'best_monitor.json',result)
        agent.save(out/'last',metadata())
        torch.save(dict(policy=model.state_dict(),optimizer=optimizer.state_dict()),out/'distillation_training.pt')
        print('TEACHER_POLICY',json.dumps(teacher_curve[-1]|dict(policy_mean=curve[-1]['mean_score'] if policy else None)),flush=True)
        if c.get('monitor_report'):subprocess.run([sys.executable,'-m',c['monitor_report']],check=False)
        evaluation_seconds+=time.perf_counter()-before;model.train()
    check(policy=True)
    for update in range(steps):
        if stopped():stop_reason='stop_file';break
        if fit_goal_met:stop_reason='diagnostic_fit_goal';break
        if elapsed()>=c['max_training_seconds']:stop_reason='training_time_budget';break
        ids=rng.integers(len(target),size=c['batch'])
        output=model(tensor_boards(data['states'][ids],device))
        loss=policy_value_loss(output,torch.as_tensor(q[ids],device=device),
            torch.as_tensor(data['legal'][ids],device=device),scale,c['action_temperature'],c['value_loss_weight'])
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite policy distillation loss')
        optimize(optimizer,loss,model.parameters());updates=update+1
        if updates%c['validation_interval']==0:check(updates%c['policy_monitor_interval']==0)
    if not stopped() and last_policy!=updates:check(policy=True)
    agent.save(out/'last',metadata())
    torch.save(dict(policy=model.state_dict(),optimizer=optimizer.state_dict()),out/'distillation_training.pt')
    result=metadata()|dict(completed_budget=updates==steps or fit_goal_met,
        completed_fit_goal=fit_goal_met,stop_reason=stop_reason,complete=False,
        mean_score=None,best_monitor_score=best,final_validation=teacher_curve[-1],monitoring_seconds=evaluation_seconds)
    if stopped():return result
    model.eval();seeds=range(c['final_seed_start'],c['final_seed_start']+100)
    final=evaluate(agent,seeds);write(out/'evaluation.json',final)
    selected=NeuralAgent.load(out/'best',device);chosen=evaluate(selected,seeds)
    write(out/'best_selection.json',chosen);save_replay(selected,out/'replay_best.html',seed=8930100)
    return result|final['summary']|dict(training_transitions=0,transitions=0,
        evaluation_transitions=final['summary']['transitions'],depth=c['depth'])
