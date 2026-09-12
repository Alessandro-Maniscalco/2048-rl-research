"""Let a teacher-initialized Transformer learn from its own games with PPO.

Only actor weights transfer. The teacher's value estimates describe strong
teacher play, so the critic is reset to learn returns of this student's policy.
Each rollout supplies new on-policy data; no teacher labels or replay buffer
are used. GAE and the clipped policy objective live in agents/ppo.py.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from research.afterstate_teacher import write,digest
from research.teacher_policy_experiment import evaluate
from rl2048.agents.neural import (NeuralAgent,tensor_boards,masked_log_probs,
                                 sample_actions,chosen,optimize)
from rl2048.agents.ppo import advantages,clipped_policy_loss
from rl2048.agents.teacher_policy_transformer import TeacherPolicyTransformer
from rl2048.vector_game import VectorGame
from rl2048.view import save_replay


@torch.no_grad()
def reset_critic(model):
    """Exactly preserve all actor logits and shared encoder tensors."""
    model.readout.weight[4].zero_();model.readout.bias[4].zero_()
    model.value_offset.zero_();model.value_scale.fill_(1.)


def train_one(c,seed,steps,out,device):
    if steps%c['envs'] or c['gamma']!=1. or c['reward_mode']!='score':
        raise ValueError('This screen uses complete game batches and undiscounted raw score/128')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    c=c|dict(seed=seed,device=device,teacher_only=False,online_training=True,
        teacher_labels_used_online=False,replay_buffer=False,root_slide_enumeration=False,
        search_at_training=False,search_at_evaluation=False,
        initialization='teacher_actor_and_reset_critic' if c.get('initial_actor_checkpoint') else 'scratch',
        critic_units='future own-policy raw score / 128',value_offset=0.,value_scale=1.)
    if c.get('initial_actor_checkpoint'):
        source=Path(c['initial_actor_checkpoint']);agent=NeuralAgent.load(source,device)
        model=agent.policy;meta=json.loads((source/'metadata.json').read_text())
        for key in ('architecture','width','depth','heads','input_encoding'):
            if meta[key]!=c[key]:raise ValueError(f'PPO actor initialization must preserve {key}')
        c['initial_actor_sha256']=digest(source/'policy.pt')
        c['teacher_initialization_history']=meta['experiment']
    else:
        model=TeacherPolicyTransformer(c['width'],c['input_encoding'],c['depth'],c['heads']).to(device)
    reset_critic(model);agent=NeuralAgent(model,'ppo',device,c['width'])
    optimizer=torch.optim.Adam(model.parameters(),lr=c['lr'],eps=1e-5)
    env=VectorGame(c['envs'],c['collection_seed'])
    # Warm compiled rules, then restore the advertised collection seed.
    env.step(np.array([np.flatnonzero(m)[0] for m in env.masks()]));env=VectorGame(c['envs'],c['collection_seed'])
    write(out/'config.json',c)
    start=time.perf_counter();evaluation_seconds=0.;collection_seconds=0.;update_seconds=0.
    transitions=updates=rollouts=gradient_examples=0
    curve=[];progress=[];episodes=[];best=-np.inf;stop_reason=None;last_monitor=-1
    def elapsed():return time.perf_counter()-start-evaluation_seconds
    def stopped():return c.get('stop_file') and Path(c['stop_file']).exists()
    def metadata():return c|dict(training_transitions=transitions,transitions=transitions,
        total_transitions=transitions,updates=updates,rollouts=rollouts,
        gradient_examples_seen=gradient_examples,training_seconds=elapsed(),
        collection_seconds=collection_seconds,update_seconds=update_seconds,
        parameter_count=sum(p.numel() for p in model.parameters()))
    def save():
        agent.save(out/'last',metadata())
        torch.save(dict(policy=model.state_dict(),optimizer=optimizer.state_dict(),
            transitions=transitions,updates=updates,torch_rng=torch.get_rng_state(),numpy_rng=rng.bit_generator.state,
            exact_environment_rng_resume=False),out/'training.pt')
        write(out/'progress.json',progress);write(out/'episodes.json',episodes)
    def monitor():
        nonlocal evaluation_seconds,best,last_monitor
        before=time.perf_counter();seconds=elapsed();model.eval()
        result=evaluate(agent,range(c['monitor_seed_start'],c['monitor_seed_start']+c['monitor_games']))
        curve.append(result['summary']|dict(transitions=transitions,updates=updates,
            training_seconds=seconds,evaluation_transitions=result['summary']['transitions']))
        write(out/'curve.json',curve)
        if result['summary']['mean_score']>best:
            best=result['summary']['mean_score'];agent.save(out/'best',metadata());write(out/'best_monitor.json',result)
        last_monitor=transitions;save()
        print('PPO_MONITOR',json.dumps(curve[-1]),flush=True)
        if c.get('monitor_report'):subprocess.run([sys.executable,'-m',c['monitor_report']],check=False)
        evaluation_seconds+=time.perf_counter()-before;model.train()
    monitor()
    while transitions<steps:
        if stopped():stop_reason='stop_file';break
        if elapsed()>=c['max_training_seconds']:stop_reason='training_time_budget';break
        before=time.perf_counter()
        data={k:[] for k in ('states','masks','actions','logp','rewards','values','next_values','terminated','truncated')}
        for _ in range(min(c['horizon'],(steps-transitions)//c['envs'])):
            if stopped():break
            states,masks=env.boards.copy(),env.masks()
            with torch.no_grad():
                output=model(tensor_boards(states,device))
                logs=masked_log_probs(output[:,:4],torch.as_tensor(masks,device=device))
                actions=sample_actions(logs.exp().cpu().numpy(),rng)
                old=logs.cpu().numpy()[np.arange(c['envs']),actions];values=output[:,4].cpu().numpy()
            reward,final_states,term,trunc,completed=env.step(actions);transitions+=c['envs']
            with torch.no_grad():next_values=model(tensor_boards(final_states,device))[:,4].cpu().numpy()
            for score,length,tile in completed[completed[:,1]>0]:
                episodes.append(dict(transitions=transitions,score=int(score),length=int(length),max_tile=int(tile)))
            items=(states,masks,actions,old,reward/128,values,next_values,term,trunc)
            for key,item in zip(data,items,strict=True):data[key].append(item)
        collection_seconds+=time.perf_counter()-before
        if stopped():stop_reason='stop_file';break
        arrays={k:np.asarray(v) for k,v in data.items()}
        adv,returns=advantages(arrays['rewards'],arrays['values'],arrays['next_values'],
            arrays['terminated'],arrays['truncated'],c['gamma'],c['lam'])
        adv=(adv-adv.mean())/(adv.std()+1e-8);size=adv.size
        x=tensor_boards(arrays['states'].reshape(size,16),device)
        masks=torch.as_tensor(arrays['masks'].reshape(size,4),device=device)
        actions=torch.as_tensor(arrays['actions'].reshape(size),device=device)
        old=torch.as_tensor(arrays['logp'].reshape(size),device=device)
        adv=torch.as_tensor(adv.reshape(size),device=device);targets=torch.as_tensor(returns.reshape(size),device=device)
        before=time.perf_counter();losses=[];kl_stop=False
        for epoch in range(c['epochs']):
            for ids in np.array_split(rng.permutation(size),max(1,int(np.ceil(size/c['batch'])))):
                if stopped():break
                index=torch.as_tensor(ids,device=device);output=model(x[index])
                logs=masked_log_probs(output[:,:4],masks[index]);new=chosen(logs,actions[index])
                ratio=(new-old[index]).exp()
                kl=float(((ratio-1)-(new-old[index])).mean().detach().cpu())
                if kl>c['target_kl']:kl_stop=True;break
                actor=clipped_policy_loss(new,old[index],adv[index],c['clip'])
                critic=(output[:,4]-targets[index]).square().mean()
                entropy=-(logs.exp()*logs).sum(1).mean()
                loss=actor+c['value_weight']*critic-c['entropy_weight']*entropy
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite PPO update')
                optimize(optimizer,loss,model.parameters(),c['max_grad_norm'])
                updates+=1;gradient_examples+=len(ids)
                losses.append([float(actor.detach().cpu()),float(critic.detach().cpu()),
                    float(entropy.detach().cpu()),kl,float(((ratio-1).abs()>c['clip']).float().mean().detach().cpu())])
            if kl_stop or stopped():break
        update_seconds+=time.perf_counter()-before;rollouts+=1
        means=np.mean(losses,axis=0).tolist() if losses else [None]*5
        row=dict(transitions=transitions,rollouts=rollouts,updates=updates,training_seconds=elapsed(),
            policy_loss=means[0],value_loss=means[1],entropy=means[2],approx_kl=means[3],clip_fraction=means[4],
            kl_early_stop=kl_stop,mean_training_score_100=float(np.mean([e['score'] for e in episodes[-100:]])) if episodes else None)
        progress.append(row);write(out/'progress.json',progress);print('PPO',json.dumps(row),flush=True)
        if not stopped() and transitions-last_monitor>=c['monitor_interval']:monitor()
    if not stopped() and last_monitor!=transitions:monitor()
    save()
    result=metadata()|dict(complete=False,mean_score=None,completed_budget=transitions==steps,
        stop_reason=stop_reason,best_monitor_score=best,monitoring_seconds=evaluation_seconds)
    if stopped():return result
    model.eval();seeds=range(c['final_seed_start'],c['final_seed_start']+100)
    final=evaluate(agent,seeds);write(out/'evaluation.json',final)
    selected=NeuralAgent.load(out/'best',device);write(out/'best_selection.json',evaluate(selected,seeds))
    save_replay(selected,out/'replay_best.html',seed=8930100)
    return result|final['summary']|dict(transitions=transitions,training_transitions=transitions,
        evaluation_transitions=final['summary']['transitions'],depth=c['depth'])
