"""Behavior cloning from the stronger native player's recorded decisions.

The objective is mean -log pi(expert action|current board), with illegal actions
masked out. No critic, score target, teacher value, bootstrapping or online data.
Complete games measure the policy, because teacher agreement can hide collapse.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from research.afterstate_teacher import write, digest
from research.teacher_policy_experiment import evaluate
from rl2048.agents.native_policy import NativePolicy
from rl2048.agents.neural import NeuralAgent, tensor_boards
from rl2048.view import save_replay


def action_loss(logits, actions, legal):
    return F.cross_entropy(logits.masked_fill(~legal, -torch.inf), actions.long())


def state_hash(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def initialize(seed, initialization, source, continuation=None):
    torch.manual_seed(seed)
    model=NativePolicy()
    if continuation:
        if initialization != 'continuation':raise ValueError('Continuation must be explicit')
        model.load_state_dict(torch.load(Path(continuation)/'policy.pt',map_location='cpu',weights_only=True))
    elif initialization == 'encoder':
        model.transfer_encoder(torch.load(Path(source)/'policy.pt',map_location='cpu',weights_only=True))
    elif initialization != 'scratch':
        raise ValueError('Expected scratch or encoder initialization')
    return model, state_hash(model._actor.state_dict())


def sample_indices(rng, count, batch, pools, fraction=0.):
    """Mix uniform board sampling with uniform stage, then uniform board in stage."""
    if not 0<=fraction<=1:raise ValueError('Stage fraction must be in[0,1]')
    indices=rng.integers(count,size=batch)
    active=[p for p in pools if len(p)]
    n=round(batch*fraction)
    if n and not active:raise ValueError('Stage mixture needs nonempty pools')
    stages=rng.integers(len(active),size=n) if n else []
    for stage,pool in enumerate(active):
        positions=np.flatnonzero(np.asarray(stages)==stage)
        indices[positions]=pool[rng.integers(len(pool),size=len(positions))]
    rng.shuffle(indices)
    return indices


@torch.no_grad()
def validate(model, states, actions, masks, batch):
    total_loss=0.;correct=0
    for start in range(0,len(states),batch):
        s,a,m=(v[start:start+batch] for v in (states,actions,masks))
        logits=model(s)
        total_loss+=float(action_loss(logits,a,m))*len(s)
        correct+=int((logits.masked_fill(~m,-torch.inf).argmax(1)==a).sum())
    return dict(action_cross_entropy=total_loss/len(states),action_agreement=correct/len(states),boards=len(states))


def train_one(c, seed, steps, out, device):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    archive_path=Path(c['expert_dataset'])/'data.npz'
    manifest=json.loads((archive_path.parent/'manifest.json').read_text())
    if not manifest['complete'] or digest(archive_path)!=manifest['data_sha256']:
        raise ValueError('Incomplete or changed expert data')
    rng=np.random.default_rng(seed)
    with np.load(archive_path,allow_pickle=False) as data:
        fitting=~data['validation'];heldout=np.flatnonzero(data['validation'])
        fit_ranks=data['states'][fitting].max(1)
        pools=[np.flatnonzero(fit_ranks<=8),np.flatnonzero((fit_ranks>=9)&(fit_ranks<=12)),
               np.flatnonzero(fit_ranks>=13)]
        ids=np.random.default_rng(128191).choice(heldout,min(c['validation_boards'],len(heldout)),replace=False)
        fit=[torch.as_tensor(data[k][fitting],dtype=d,device=device) for k,d in
             [('states',torch.long),('actions',torch.long),('legal',torch.bool)]]
        validation=[torch.as_tensor(data[k][ids],dtype=d,device=device) for k,d in
                    [('states',torch.long),('actions',torch.long),('legal',torch.bool)]]
    model,actor_hash=initialize(seed,c['initialization'],c['source_checkpoint'],c.get('continuation_checkpoint'))
    agent=NeuralAgent(model,'native_behavior_cloning',device,width=1024)
    model=agent.policy
    optimizer=torch.optim.Adam(model.parameters(),lr=c['lr'],eps=1e-5)
    source_sha=digest(Path(c['source_checkpoint'])/'policy.pt')
    c=c|dict(device=device,seed=seed,source_sha256=source_sha,data_sha256=manifest['data_sha256'],
        initialization_actor_sha256=actor_hash,fitting_boards=len(fit[0]),
        initialization_policy_sha256=state_hash(model.state_dict()),
        continuation_sha256=digest(Path(c['continuation_checkpoint'])/'policy.pt') if c.get('continuation_checkpoint') else None,
        fitting_stage_counts=[len(p) for p in pools],
        parameter_count=sum(p.numel() for p in model.parameters()),input_max_rank=17,
        value_target=None,critic=False,reinforcement_learning=False,online_training=False,
        search_at_training=False,search_at_evaluation=False,teacher_only=True,
        expert_label='recorded action, not Q-value or subgoal probability',
        shared_fitting_tensors_resident_on_device=True)
    write(out/'config.json',c)
    updates=0;training_seconds=0.;monitoring_seconds=0.;curve=[];best=-1.;losses=[]
    first_ids=None;stop_reason='update_budget';last_monitor=-1
    def stopped():return Path(c['stop_file']).exists()
    def metadata():return c|dict(updates=updates,training_seconds=training_seconds,
        transitions=0,training_transitions=0,examples_seen=updates*c['batch'])
    def save():
        agent.save(out/'last',metadata())
        torch.save(dict(policy=model.state_dict(),optimizer=optimizer.state_dict(),
            updates=updates,numpy_rng=rng.bit_generator.state),out/'training.pt')
    def monitor():
        nonlocal best,monitoring_seconds,last_monitor
        before=time.perf_counter();model.eval()
        metrics=validate(model,*validation,c['batch'])
        evaluation=evaluate(agent,range(c['monitor_seed_start'],c['monitor_seed_start']+c['monitor_games']))
        row=metadata()|metrics|evaluation['summary']|dict(updates=updates,training_seconds=training_seconds,
            mean_recent_training_loss=float(np.mean(losses[-128:])) if losses else None)
        curve.append(row);write(out/'curve.json',curve)
        if row['mean_score']>best:
            best=row['mean_score'];agent.save(out/'best',metadata());write(out/'best_monitor.json',evaluation)
        save();last_monitor=updates
        print('NATIVE_POLICY',json.dumps({k:row[k] for k in ('updates','mean_score','action_agreement','action_cross_entropy','training_seconds')}),flush=True)
        if c.get('monitor_report'):
            subprocess.run([sys.executable,'-m',c['monitor_report']],check=False)
        monitoring_seconds+=time.perf_counter()-before;model.train()
    agent.save(out/'initial',metadata())
    monitor()
    since=time.perf_counter()
    for update in range(steps):
        if stopped():stop_reason='stop_file';break
        if training_seconds+time.perf_counter()-since>=c['max_training_seconds']:
            stop_reason='training_time_budget';break
        # Preserve the exact original sampling stream for the existing uniform
        # experiments; a changed mixture is an explicitly different data recipe.
        ids=(sample_indices(rng,len(fit[0]),c['batch'],pools,c['stage_fraction'])
             if c.get('stage_fraction',0) else rng.integers(len(fit[0]),size=c['batch']))
        if first_ids is None:first_ids=hashlib.sha256(ids.tobytes()).hexdigest()
        idx=torch.as_tensor(ids,device=device)
        s,a,m=(v[idx] for v in fit)
        loss=action_loss(model(s),a,m)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite imitation loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),c['max_grad_norm'],error_if_nonfinite=True)
        optimizer.step();updates=update+1;losses.append(float(loss.detach()))
        if updates%c['monitor_interval']==0:
            training_seconds+=time.perf_counter()-since;monitor();since=time.perf_counter()
    training_seconds+=time.perf_counter()-since
    if not stopped() and last_monitor!=updates:monitor()
    save()
    result=metadata()|dict(complete=False,completed_budget=updates==steps,stop_reason=stop_reason,
        monitoring_seconds=monitoring_seconds,first_minibatch_indices_sha256=first_ids,
        best_monitor_score=best,source_unchanged=digest(Path(c['source_checkpoint'])/'policy.pt')==source_sha,
        dataset_unchanged=digest(archive_path)==manifest['data_sha256'])
    if stopped():return result
    model.eval()
    final=evaluate(agent,range(c['final_seed_start'],c['final_seed_start']+c.get('final_games',100)))
    write(out/'evaluation.json',final)
    restored=NeuralAgent.load(out/'last',device)
    torch.testing.assert_close(restored.policy(validation[0][:8]),model(validation[0][:8]))
    save_replay(restored,out/'replay_last.html',seed=8930100)
    return result|final['summary']|dict(transitions=0,training_transitions=0,
        evaluation_transitions=final['summary']['transitions'],checkpoint_reload_verified=True,
        elapsed_seconds=training_seconds+monitoring_seconds+final['summary'].get('elapsed_seconds',0))
