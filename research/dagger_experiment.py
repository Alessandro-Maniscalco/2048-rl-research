"""DAgger versus a fixed-data action-imitation control.

Both start from the same actor and original fitting partition. In DAgger,
the greedy student plays complete games, the frozen n-tuple teacher labels
all visited boards, and the data join the cumulative fitting set. The fixed
control gets the same optimizer budget without the newly collected boards.

Ross, Gordon and Bagnell (2011), Algorithm 3.1, beta=0 after pretraining:
https://proceedings.mlr.press/v15/ross11a.html
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from research.afterstate_teacher import digest, write
from research.dagger_sampling import StageSampler
from research.reinforce_experiment import collect_complete_games
from research.teacher_policy_experiment import evaluate
from research.teacher_transformer_experiment import prepare_data
from rl2048.afterstate_compare import moves, tuple_values
from rl2048.agents.imitation import optimal_actions, imitation_loss, teacher_gap_penalty, reference_policy_kl
from rl2048.agents.neural import NeuralAgent, tensor_boards, policy_logits
from rl2048.agents.ntuple import NTupleAgent, row_tables
from rl2048.agents.reinforce import PolicyTransformer, copy_actor
from rl2048.view import save_replay


def teacher_targets(states, teacher):
    """Four exact root slides + all shared table entries; no future search."""
    after, gains, legal = moves(states, *row_tables())
    q = np.full(legal.shape, -np.inf, np.float32)
    q[legal] = (gains[legal] + tuple_values(after[legal], teacher.weights, teacher.patterns)) / 128.
    return dict(states=states, legal=legal, q=q, best=optimal_actions(q, legal))


def from_archive(data):
    q = np.where(data['legal'], data['values'] + data['gains']/128., -np.inf)
    return dict(states=data['states'], legal=data['legal'], q=q,
                best=optimal_actions(q, data['legal']))


def aggregate(old, new):
    """Keep every old and new board; no forgetting, truncation or subsampling."""
    return {k: np.concatenate((old[k], new[k])) for k in old}


def initialize_actor(config, source, device):
    """Preserve the chosen pretrained actor; CNN critic never enters the loss."""
    old = NeuralAgent.load(source, 'cpu')
    meta = json.loads((Path(source)/'metadata.json').read_text())
    if config['architecture'] == 'pretrained_ml2048':
        if meta['architecture'] != 'pretrained_ml2048':
            raise ValueError('CNN initialization requires pretrained_ml2048 source')
        for key in ('width', 'input_encoding'):
            if meta[key] != config[key]:
                raise ValueError(f'Actor initialization must preserve {key}')
        if config.get('depth') is not None or config.get('heads') is not None:
            raise ValueError('CNN has no configurable attention depth or heads')
        model = old.policy.to(device)
        model._critic.requires_grad_(False)
        return model, meta
    if config['architecture'] != 'transformer_policy':
        raise ValueError('Unsupported DAgger actor architecture')
    for key in ('width', 'depth', 'heads', 'input_encoding'):
        if meta[key] != config[key]:
            raise ValueError(f'Actor initialization must preserve {key}')
    model = PolicyTransformer(config['width'], config['input_encoding'],
                              config['depth'], config['heads']).to(device)
    copy_actor(model, old.policy)
    return model, meta


def restore_continuation(folder, out, config, data, model, optimizer, rng):
    """Restore a complete round, validating model, data, recipe and provenance.

    Copy the small compressed blocks to the new run so every experiment is
    auditable without mutating or depending on a writable parent checkpoint.
    The first saved format omitted cumulative counts; its local counts are
    the correct historical totals because that version could not resume.
    """
    folder = Path(folder)
    old = json.loads((folder/'last/metadata.json').read_text())['experiment']
    state = torch.load(folder/'training.pt', map_location='cpu', weights_only=True)
    unchanged = ('algorithm', 'data_source', 'width', 'depth', 'heads', 'input_encoding',
        'batch', 'lr', 'max_grad_norm', 'updates_per_round', 'collection_games',
        'collection_seed', 'seed', 'initial_actor_sha256', 'data_sha256', 'teacher_sha256')
    if old.get('architecture', 'transformer_policy') != config['architecture']:
        raise ValueError('Exact DAgger continuation must preserve architecture')
    for key in unchanged:
        if old[key] != config[key]:
            raise ValueError(f'Exact DAgger continuation must preserve {key}')
    objective_keys = ('teacher_gap_weight', 'teacher_gap_scale_points', 'reference_kl_weight', 'teacher_action_weight')
    old_objective = tuple(old.get(k, default) for k,default in zip(objective_keys, (0., 16384., 0., 1.)))
    new_objective = tuple(config[k] for k in objective_keys)
    changed = old_objective != new_objective
    if changed and not config.get('transfer_dagger_objective', False):
        raise ValueError('Changed loss requires explicit transfer_dagger_objective')
    if not changed and config.get('transfer_dagger_objective', False):
        raise ValueError('Objective transfer must actually change the loss')
    sampling_changed = old.get('stage_sampling_fraction', 0.) != config['stage_sampling_fraction']
    if sampling_changed and not config.get('transfer_dagger_sampling', False):
        raise ValueError('Changed sampling requires explicit transfer_dagger_sampling')
    if not sampling_changed and config.get('transfer_dagger_sampling', False):
        raise ValueError('Sampling transfer must actually change the sampling fraction')
    if (old['updates'] != state['updates'] or old['rounds'] != state['rounds']
            or old['updates'] % config['updates_per_round']
            or old['updates'] != old['rounds']*config['updates_per_round']):
        raise ValueError('DAgger continuation requires a saved complete round')
    if state['original_data_sha256'] != config['data_sha256']:
        raise ValueError('Continuation fitting data changed')
    last = torch.load(folder/'last/policy.pt', map_location='cpu', weights_only=True)
    if last.keys() != state['policy'].keys() or any(not torch.equal(last[k], state['policy'][k]) for k in last):
        raise ValueError('Actor checkpoint and optimizer checkpoint are from different saves')
    blocks = state['dataset_blocks']
    if blocks != json.loads((folder/'dataset_blocks.json').read_text()):
        raise ValueError('Continuation dataset manifest differs from saved training state')
    for block in blocks:
        name = block['file']
        if Path(name).name != name or digest(folder/name) != block['sha256']:
            raise ValueError('Continuation dataset block path or hash changed')
        with np.load(folder/name, allow_pickle=False) as archive:
            new = {k:archive[k] for k in data}
            if len(new['states']) != block['boards'] or archive['episode_lengths'].sum() != block['boards']:
                raise ValueError('Saved complete-game block counts differ')
            np.testing.assert_array_equal(optimal_actions(new['q'], new['legal']), new['best'])
        data = aggregate(data, new)
        shutil.copy2(folder/name, out/name)
    if len(data['states']) != old['fitting_boards']:
        raise ValueError('Restored fitting dataset size differs')
    model.load_state_dict(state['policy'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    rng.bit_generator.state = state['numpy_rng']
    torch.set_rng_state(state['torch_rng'])
    return data, blocks, old


@torch.no_grad()
def validate(model, data, device, batch):
    model.eval()
    losses, agreements, gaps = [], [], []
    for start in range(0, len(data['states']), batch):
        part = {k: v[start:start+batch] for k, v in data.items()}
        logits = policy_logits(model, tensor_boards(part['states'], device))
        loss = imitation_loss(logits, torch.as_tensor(part['legal'], device=device),
                              torch.as_tensor(part['best'], device=device))
        choices = np.where(part['legal'], logits.cpu().numpy(), -np.inf).argmax(1)
        ids = np.arange(len(choices))
        losses.append((float(loss.cpu()), len(choices)))
        agreements.extend(part['best'][ids, choices])
        gaps.extend(part['q'].max(1)-part['q'][ids, choices])
    return dict(hard_action_loss=sum(v*n for v,n in losses)/sum(n for _,n in losses),
        teacher_action_agreement=float(np.mean(agreements)), mean_teacher_action_gap=float(np.mean(gaps)))


def train_one(config, seed, steps, out, device):
    c = dict(config)
    c.setdefault('teacher_gap_weight', 0.)
    c.setdefault('teacher_gap_scale_points', 16384.)
    c.setdefault('stage_sampling_fraction', 0.)
    c.setdefault('reference_kl_weight', 0.)
    c.setdefault('teacher_action_weight', 1.)
    if c['architecture'] == 'pretrained_ml2048':
        c.setdefault('depth', None)
        c.setdefault('heads', None)
    if not np.isfinite(c['stage_sampling_fraction']) or not 0 <= c['stage_sampling_fraction'] <= 1:
        raise ValueError('Stage sampling fraction must be finite and between zero and one')
    if not np.isfinite(c['teacher_gap_weight']) or c['teacher_gap_weight'] < 0:
        raise ValueError('Teacher gap weight must be nonnegative and finite')
    if not np.isfinite(c['reference_kl_weight']) or c['reference_kl_weight'] < 0:
        raise ValueError('Reference KL weight must be nonnegative and finite')
    if not np.isfinite(c['teacher_action_weight']) or c['teacher_action_weight'] < 0:
        raise ValueError('Teacher action weight must be nonnegative and finite')
    if c['teacher_action_weight']+c['teacher_gap_weight'] <= 0:
        raise ValueError('At least one teacher objective must have positive weight')
    if not np.isfinite(c['teacher_gap_scale_points']) or c['teacher_gap_scale_points'] <= 0:
        raise ValueError('Teacher gap scale must be positive and finite')
    if c.get('transfer_dagger_objective') and not c.get('resume_dagger'):
        raise ValueError('Objective transfer requires a completed source run')
    if c.get('transfer_dagger_sampling') and not c.get('resume_dagger'):
        raise ValueError('Sampling transfer requires a completed source run')
    if c['algorithm'] != 'dagger' or c['data_source'] not in ('fixed', 'student'):
        raise ValueError('Expected a DAgger experiment with fixed or student data')
    if steps < 1 or c['updates_per_round'] < 1 or c['collection_games'] < 1:
        raise ValueError('Positive update and collection budgets required')
    if c.get('resume'):
        raise ValueError('Use explicit resume_dagger to restore optimizer and every dataset block')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    _, _, held, manifest, _, _ = prepare_data(c['teacher_dataset'], c.get('validation_boards', 2048))
    validation = from_archive(held)
    with np.load(Path(c['teacher_dataset'])/'data.npz', allow_pickle=False) as archive:
        fit = ~archive['validation']
        data = from_archive({k: archive[k][fit] for k in ('states', 'legal', 'values', 'gains')})
    original_count = len(data['states'])
    teacher = None
    if c['data_source'] == 'student':
        teacher_path = Path(c['teacher_checkpoint'])
        if digest(teacher_path/'weights.npy') != manifest['teacher_sha256']:
            raise ValueError('New and old fitting boards must use the identical frozen teacher')
        teacher = NTupleAgent.load(teacher_path, mmap_mode='r')
    source = Path(c['initial_actor_checkpoint'])
    model, meta = initialize_actor(c, source, device)
    reference = None
    if c['reference_kl_weight']:
        reference = NeuralAgent.load(source, device).policy.eval().requires_grad_(False)
    c.update(seed=seed, device=device, teacher=True, teacher_only=teacher is None,
        online_training=teacher is not None, reinforcement_learning=False, imitation_learning=True,
        teacher_execution_fraction=0., teacher_queries_for_labels_only=True,
        critic=False, bootstrap=False, search_at_evaluation=False,
        target='all legal teacher-best root actions; full table value plus immediate merge points',
        loss=(f'{c["teacher_action_weight"]:g} times negative log total probability of teacher-best action set'
            + (' plus weighted expected teacher gap' if c['teacher_gap_weight'] else '')
            + (' plus KL from frozen starting policy to student' if c['reference_kl_weight'] else '')),
        initialization=('external pretrained CNN actor' if c['architecture'] == 'pretrained_ml2048'
                        else 'original supervised actor'),
        initial_actor_sha256=digest(source/'policy.pt'),
        teacher_initialization_history=meta['experiment'], teacher_sha256=manifest['teacher_sha256'],
        data_sha256=manifest['data_sha256'], original_fitting_boards=original_count,
        validation_boards_used=len(validation['states']), requested_supervised_updates=steps,
        sampling=('mixture of uniform boards and uniform nonempty max-tile stages'
            if c['stage_sampling_fraction'] else
            'uniform boards from original fitting set plus every collected round'),
        label_units='teacher root Q in raw merge points / 128')
    if reference is not None:
        c.update(reference_policy_sha256=c['initial_actor_sha256'],
                 reference_policy='frozen original initialization, not the latest student or the n-tuple teacher')
    if c['architecture'] == 'pretrained_ml2048':
        c.update(unused_critic_parameters_frozen=True, input_max_rank=15,
                 input_rank_overflow='clip to rank 15, preserved pretrained input contract')
    agent = NeuralAgent(model, 'dagger', device, c['width'])
    optimizer = torch.optim.Adam(model.parameters(), lr=c['lr'], eps=1e-5)
    inherited, labelled_rounds = {}, []
    if c.get('resume_dagger'):
        data, labelled_rounds, inherited = restore_continuation(
            c['resume_dagger'], out, c, data, model, optimizer, rng)
        c.update(initialization='complete DAgger round continuation',
            resumed_policy_sha256=digest(Path(c['resume_dagger'])/'last/policy.pt'),
            inherited_fitting_boards=len(data['states']))
        if c.get('transfer_dagger_objective'):
            c.update(initialization='explicit DAgger objective transfer; actor, Adam, RNG and data retained',
                source_objective={k:inherited.get(k, default) for k,default in
                    zip(('teacher_gap_weight', 'teacher_gap_scale_points', 'reference_kl_weight', 'teacher_action_weight'), (0., 16384., 0., 1.))})
        if c.get('transfer_dagger_sampling'):
            c.update(initialization='explicit DAgger sampling transfer; actor, Adam, RNG and data retained',
                source_stage_sampling_fraction=inherited.get('stage_sampling_fraction', 0.))
    inherited_rounds = inherited.get('total_rounds', inherited.get('rounds', 0))
    inherited_updates = inherited.get('total_dagger_updates', inherited.get('updates', 0))
    inherited_transitions = inherited.get('total_training_transitions', inherited.get('training_transitions', 0))
    inherited_games = inherited.get('total_completed_training_games', inherited.get('completed_training_games', 0))
    inherited_block_count = len(labelled_rounds)
    write(out/'config.json', c)
    started = time.perf_counter()
    evaluation_seconds = collection_seconds = label_seconds = update_seconds = 0.
    transitions = labelled_transitions = updates = rounds = games = 0
    best_score, last_monitor, stop_reason = -np.inf, -1, None
    progress, curve = [], []

    def elapsed():
        return time.perf_counter()-started-evaluation_seconds

    def abort():
        if c.get('stop_file') and Path(c['stop_file']).exists():
            return 'stop_file'
        return 'training_time_budget' if elapsed() >= c['max_training_seconds'] else None

    def metadata():
        return c | dict(transitions=transitions, training_transitions=transitions,
            labelled_transitions=labelled_transitions, discarded_transitions=transitions-labelled_transitions,
            updates=updates, supervised_updates=updates,
            total_dagger_updates=inherited_updates+updates,
            total_supervised_updates=meta['experiment'].get('total_supervised_updates',0)+inherited_updates+updates,
            supervised_examples_seen=updates*c['batch'], rounds=rounds, completed_training_games=games,
            total_rounds=inherited_rounds+rounds, total_training_transitions=inherited_transitions+transitions,
            total_completed_training_games=inherited_games+games,
            total_training_seconds=inherited.get('total_training_seconds',inherited.get('training_seconds',0.))+elapsed(),
            fitting_boards=len(data['states']), teacher_board_queries=labelled_transitions,
            teacher_afterstate_queries=sum(r['afterstate_queries'] for r in labelled_rounds[inherited_block_count:]),
            total_teacher_afterstate_queries=sum(r['afterstate_queries'] for r in labelled_rounds),
            training_seconds=elapsed(), collection_seconds=collection_seconds,
            label_seconds=label_seconds, update_seconds=update_seconds,
            parameter_count=sum(p.numel() for p in model.parameters()),
            trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad))

    def save():
        agent.save(out/'last', metadata())
        torch.save(dict(format_version=2, policy=model.state_dict(), optimizer=optimizer.state_dict(),
            updates=updates, rounds=rounds, transitions=transitions,
            torch_rng=torch.get_rng_state(), numpy_rng=rng.bit_generator.state,
            dataset_blocks=labelled_rounds, original_data_sha256=c['data_sha256']), out/'training.pt')
        write(out/'progress.json', progress)
        write(out/'dataset_blocks.json', labelled_rounds)

    def monitor():
        nonlocal evaluation_seconds, best_score, last_monitor
        before, seconds = time.perf_counter(), elapsed()
        metrics = validate(model, validation, device, c['batch'])
        result = evaluate(agent, range(c['monitor_seed_start'], c['monitor_seed_start']+c['monitor_games']))
        curve.append(result['summary'] | metrics | dict(transitions=transitions, updates=updates,
            rounds=rounds, training_seconds=seconds, fitting_boards=len(data['states']),
            total_dagger_updates=inherited_updates+updates,
            total_training_transitions=inherited_transitions+transitions,
            total_training_seconds=inherited.get('total_training_seconds',inherited.get('training_seconds',0.))+seconds))
        write(out/'curve.json', curve)
        if result['summary']['mean_score'] > best_score:
            best_score = result['summary']['mean_score']
            agent.save(out/'best', metadata())
            write(out/'best_monitor.json', result)
        last_monitor = updates
        save()
        print('DAGGER_MONITOR', json.dumps(curve[-1]), flush=True)
        if c.get('monitor_report'):
            subprocess.run([sys.executable, '-m', c['monitor_report']], check=False)
        evaluation_seconds += time.perf_counter()-before

    monitor()
    while updates < steps:
        stop_reason = abort()
        if stop_reason:
            break
        current_metrics = {}
        if teacher is not None:
            before = time.perf_counter()
            collected, episodes, used, stop_reason = collect_complete_games(model,
                c['collection_games'], c['collection_seed']+inherited_rounds+rounds, rng, device, abort, greedy=True)
            transitions += used
            collection_seconds += time.perf_counter()-before
            if stop_reason:
                break
            before = time.perf_counter()
            new = teacher_targets(collected['states'], teacher)
            np.testing.assert_array_equal(new['legal'], collected['masks'])
            labelled_transitions += used
            games += len(episodes)
            label_seconds += time.perf_counter()-before
            # Audit teacher disagreements before fitting this newly visited batch.
            current_metrics = validate(model, new, device, c['batch'])
            actions = collected['actions']
            mistakes = np.flatnonzero(~new['best'][np.arange(used), actions])
            examples = [dict(index=int(i), board_exponents=new['states'][i].tolist(),
                played_action=int(actions[i]), teacher_best_actions=np.flatnonzero(new['best'][i]).tolist(),
                teacher_root_q_points=[float(v*128) if ok else None for v,ok in zip(new['q'][i],new['legal'][i])])
                for i in mistakes[:12]]
            block_name = f'round_{inherited_rounds+rounds+1:03d}.npz'
            np.savez_compressed(out/block_name, **new, played_actions=actions,
                episode_lengths=np.array([e['length'] for e in episodes]),
                episode_scores=np.array([e['score'] for e in episodes]))
            labelled_rounds.append(dict(file=block_name, sha256=digest(out/block_name),
                collection_seed=c['collection_seed']+inherited_rounds+rounds, boards=used, games=len(episodes),
                afterstate_queries=int(new['legal'].sum()), mean_collection_score=float(np.mean([e['score'] for e in episodes])),
                before_fit=current_metrics, examples=examples))
            data = aggregate(data, new)
            write(out/'dataset_blocks.json', labelled_rounds)
        end_update = min(updates+c['updates_per_round'], steps)
        before = time.perf_counter()
        sampler = StageSampler(data['states'], c['stage_sampling_fraction'])
        losses, ce_losses, gap_losses, kl_losses = [], [], [], []
        model.train()
        while updates < end_update:
            stop_reason = abort()
            if stop_reason:
                break
            ids = sampler.sample(rng, c['batch'])
            logits = policy_logits(model, tensor_boards(data['states'][ids], device))
            legal = torch.as_tensor(data['legal'][ids], device=device)
            best = torch.as_tensor(data['best'][ids], device=device)
            ce_loss = imitation_loss(logits, legal, best)
            loss = c['teacher_action_weight']*ce_loss
            if c['teacher_gap_weight']:
                gap_loss = teacher_gap_penalty(logits, legal, best,
                    torch.as_tensor(data['q'][ids], device=device), c['teacher_gap_scale_points'])
                loss = loss+c['teacher_gap_weight']*gap_loss
                gap_losses.append(float(gap_loss.detach().cpu()))
            if reference is not None:
                with torch.no_grad():
                    prior_logits = policy_logits(reference, tensor_boards(data['states'][ids], device))
                kl_loss = reference_policy_kl(logits, prior_logits, legal)
                loss = loss+c['reference_kl_weight']*kl_loss
                kl_losses.append(float(kl_loss.detach().cpu()))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite imitation loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c['max_grad_norm'], error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            ce_losses.append(float(ce_loss.detach().cpu()))
            updates += 1
        update_seconds += time.perf_counter()-before
        rounds += 1
        progress.append(dict(updates=updates, rounds=rounds, transitions=transitions,
            fitting_boards=len(data['states']), training_seconds=elapsed(),
            mean_minibatch_loss=float(np.mean(losses)) if losses else None,
            mean_cross_entropy_loss=float(np.mean(ce_losses)) if ce_losses else None,
            mean_teacher_gap_penalty=float(np.mean(gap_losses)) if gap_losses else None,
            mean_reference_kl=float(np.mean(kl_losses)) if kl_losses else None,
            sampling_stage_counts=sampler.counts,
            latest_collection_validation=current_metrics))
        save()
        print('DAGGER', json.dumps(progress[-1]), flush=True)
        if stop_reason:
            break
        if rounds % c['monitor_every_rounds'] == 0:
            monitor()
    if stop_reason != 'stop_file' and last_monitor != updates:
        monitor()
    save()
    result = metadata() | dict(complete=False, mean_score=None, completed_budget=updates == steps,
        stop_reason=stop_reason, best_monitor_score=best_score, monitoring_seconds=evaluation_seconds)
    if stop_reason == 'stop_file':
        return result
    model.eval()
    seeds = range(c['final_seed_start'], c['final_seed_start']+100)
    final = evaluate(agent, seeds)
    write(out/'evaluation.json', final)
    selected = NeuralAgent.load(out/'best', device)
    write(out/'best_selection.json', evaluate(selected, seeds))
    save_replay(agent, out/'replay_last.html', seed=8930100)
    save_replay(selected, out/'replay_best.html', seed=8930100)
    return result | final['summary'] | dict(transitions=transitions, training_transitions=transitions,
        evaluation_transitions=final['summary']['transitions'], depth=c['depth'])
