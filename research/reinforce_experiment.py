"""Readable, critic-free REINFORCE training on complete 2048 games.

The collection policy cannot change until ALL games in the batch terminate.
Finished games wait; no new episode starts under a partially updated policy.
CPU simulation and GPU inference alternate. Gradient microbatches limit memory
without taking extra optimizer steps or reusing stale data.
"""
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from research.afterstate_teacher import write, digest
from research.teacher_policy_experiment import evaluate
from rl2048.agents.neural import NeuralAgent, tensor_boards, sample_actions, policy_logits
from rl2048.agents.reinforce import (PolicyTransformer, copy_actor, reward_to_go, policy_loss,
                                    leave_one_out_time_baseline, action_log_probs)
from rl2048.vector_game import VectorGame, legal_masks, step_boards
from rl2048.view import save_replay


def collect_complete_games(model, count, seed, rng, device, abort=lambda: None, temperature=1., greedy=False, spawn_safety=False):
    """Return full episodes or an explicitly discarded partial batch on abort.

    step_boards auto-resets a completed row, but that row is removed from the
    active set immediately; its reset board is never played in this batch.
    DAgger can reuse collection with greedy=True; REINFORCE always samples.
    """
    env = VectorGame(count, seed)
    active = np.ones(count, dtype=bool)
    tracks = [{k: [] for k in ('states', 'masks', 'actions', 'rewards', 'logp', 'legal_masks')}
              for _ in range(count)]
    episodes = [None] * count
    transitions = 0
    model.eval()
    while active.any():
        reason = abort()
        if reason:
            return None, [], transitions, reason
        ids = np.flatnonzero(active)
        boards = env.boards[ids].copy()
        masks = legal_masks(boards, env.rows, env.row_rewards)
        legal = masks.copy()
        if spawn_safety:
            from rl2048.agents.spawn_safety import minimum_risk_mask
            masks = minimum_risk_mask(boards, legal)
        with torch.no_grad():
            logits = policy_logits(model, tensor_boards(boards, device))
            logs = action_log_probs(logits,
                                    torch.as_tensor(masks, device=device), temperature).cpu().numpy()
        actions = (np.where(masks, logits.cpu().numpy(), -np.inf).argmax(1)
                   if greedy else sample_actions(np.exp(logs), rng))
        scores, lengths = env.scores[ids].copy(), env.lengths[ids].copy()
        moved = boards.copy()
        rewards, final, term, trunc, completed = step_boards(
            moved, actions, scores, lengths, env.rows, env.row_rewards, env.max_steps)
        if trunc.any():
            raise RuntimeError('REINFORCE requires complete games; a game hit the explicit safety limit')
        env.boards[ids], env.scores[ids], env.lengths[ids] = moved, scores, lengths
        transitions += len(ids)
        for j, i in enumerate(ids):
            items = (boards[j], masks[j], actions[j], rewards[j] / 128., logs[j, actions[j]], legal[j])
            for key, value in zip(tracks[i], items, strict=True):
                tracks[i][key].append(value)
            if term[j]:
                active[i] = False
                score, length, tile = completed[j]
                episodes[i] = dict(score=int(score), length=int(length), max_tile=int(tile),
                                   terminated=True, truncated=False)
    arrays = {key: np.concatenate([np.asarray(t[key]) for t in tracks]) for key in tracks[0]}
    arrays['returns'] = np.concatenate([reward_to_go(t['rewards']) for t in tracks])
    return arrays, episodes, transitions, None


def train_one(config, seed, steps, out, device):
    if config['gamma'] != 1. or config['reward_mode'] != 'score':
        raise ValueError('Plain score experiment uses undiscounted actual merge points/128')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    temperature = config.get('action_temperature', 1.)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('A positive finite action temperature is required')
    baseline_mode = config.get('baseline_mode', 'none')
    complete_batches = config.get('complete_batches')
    if complete_batches is not None and (type(complete_batches) is not int or complete_batches < 1):
        raise ValueError('Complete-batch budget must be a positive integer')
    if baseline_mode not in ('none', 'leave_one_out_time'):
        raise ValueError('Unknown REINFORCE baseline mode')
    if (config['algorithm'] == 'reinforce') != (baseline_mode == 'none'):
        raise ValueError('Keep plain REINFORCE and the baseline experiment distinctly named')
    if baseline_mode != 'none' and config['algorithm'] != 'reinforce_loo':
        raise ValueError('The leave-one-out baseline experiment is named reinforce_loo')
    c = config | dict(seed=seed, device=device, action_temperature=temperature,
        teacher_only=False, online_training=True,
        teacher_labels_used_online=False, replay_buffer=False, critic=False,
        baseline=False if baseline_mode == 'none' else baseline_mode,
        bootstrap=False, return_normalization='fixed points / 128 only', entropy_bonus=0.,
        optimizer_steps_per_complete_batch=1,
        requested_transitions=steps if complete_batches is None else None,
        requested_complete_batches=complete_batches,
        search_at_training=False, search_at_evaluation=False,
        initialization='teacher_actor' if config.get('initial_actor_checkpoint') else 'scratch')
    c['spawn_safety'] = config.get('spawn_safety', False)
    if c['spawn_safety']:
        c.update(one_spawn_dynamics_for_action_mask=True,
            policy_support='legal moves with minimum exact next-spawn death risk',
            search_at_training=True, search_at_evaluation=True,
            search_description='Exact one-spawn risk filter only; no future reward search or value estimates')
    if c.get('architecture') == 'pretrained_ml2048':
        if not c.get('initial_actor_checkpoint') or c.get('depth') is not None or c.get('heads') is not None:
            raise ValueError('Pretrained CNN requires a source checkpoint and no attention depth/heads')
        source = Path(c['initial_actor_checkpoint'])
        old = NeuralAgent.load(source, 'cpu')
        meta = json.loads((source / 'metadata.json').read_text())
        for key in ('architecture', 'width', 'input_encoding'):
            if meta[key] != c[key]:
                raise ValueError(f'Actor initialization must preserve {key}')
        model = old.policy.to(device)
        model._critic.requires_grad_(False)
        c.update(initial_actor_sha256=digest(source/'policy.pt'),
            initialization='external pretrained CNN actor', pretraining_history=meta['experiment'],
            unused_critic_parameters_frozen=True,input_max_rank=15)
        del old
    else:
        model = PolicyTransformer(c['width'], c['input_encoding'], c['depth'], c['heads']).to(device)
        if c.get('initial_actor_checkpoint'):
            source = Path(c['initial_actor_checkpoint'])
            old = NeuralAgent.load(source, 'cpu')
            meta = json.loads((source / 'metadata.json').read_text())
            for key in ('width', 'depth', 'heads', 'input_encoding'):
                if meta[key] != c[key]:
                    raise ValueError(f'Actor initialization must preserve {key}')
            copy_actor(model, old.policy)
            c['initial_actor_sha256'] = digest(source / 'policy.pt')
            c['teacher_initialization_history'] = meta['experiment']
            del old
    agent = NeuralAgent(model, c['algorithm'], device, c['width'], c['spawn_safety'])
    optimizer = torch.optim.Adam(model.parameters(), lr=c['lr'], eps=1e-5)
    warm = VectorGame(2, c['collection_seed'])
    warm.step(np.array([np.flatnonzero(m)[0] for m in warm.masks()]))
    write(out / 'config.json', c)
    started = time.perf_counter()
    evaluation_seconds = collection_seconds = update_seconds = 0.
    transitions = trained_transitions = updates = 0
    curve, progress, episodes = [], [], []
    best, last_monitor, stop_reason = -np.inf, -1, None

    def elapsed():
        return time.perf_counter() - started - evaluation_seconds

    def abort():
        if c.get('stop_file') and Path(c['stop_file']).exists():
            return 'stop_file'
        if elapsed() >= c['max_training_seconds']:
            return 'training_time_budget'
        return None

    def budget_completed():
        return updates >= complete_batches if complete_batches is not None else trained_transitions >= steps

    def metadata():
        return c | dict(transitions=transitions, training_transitions=transitions,
            trained_transitions=trained_transitions, discarded_transitions=transitions-trained_transitions,
            updates=updates, completed_training_games=len(episodes), training_seconds=elapsed(),
            collection_seconds=collection_seconds, update_seconds=update_seconds,
            parameter_count=sum(p.numel() for p in model.parameters()))

    def save():
        agent.save(out / 'last', metadata())
        torch.save(dict(policy=model.state_dict(), optimizer=optimizer.state_dict(),
            transitions=transitions, trained_transitions=trained_transitions, updates=updates,
            next_batch_seed=c['collection_seed']+updates, torch_rng=torch.get_rng_state(),
            numpy_rng=rng.bit_generator.state, resume_at_complete_batch_boundary=True), out / 'training.pt')
        write(out / 'progress.json', progress)
        write(out / 'episodes.json', episodes)

    def monitor():
        nonlocal evaluation_seconds, best, last_monitor
        before = time.perf_counter()
        model.eval()
        seconds = elapsed()
        result = evaluate(agent, range(c['monitor_seed_start'], c['monitor_seed_start']+c['monitor_games']))
        curve.append(result['summary'] | dict(transitions=transitions, updates=updates,
            training_seconds=seconds, evaluation_transitions=result['summary']['transitions']))
        if c.get('monitor_sampled'):
            from research.policy_sampling_experiment import evaluate_sampled
            sampled = evaluate_sampled(agent,
                range(c['monitor_seed_start'], c['monitor_seed_start']+c['monitor_games']),
                c['sampling_seed'], temperature=temperature)
            if not sampled['summary']['complete'] or sampled['summary']['truncated_episodes']:
                raise RuntimeError('A sampled monitoring game did not complete')
            curve[-1].update(sampled_mean_score=sampled['summary']['mean_score'],
                sampled_evaluation_transitions=sampled['summary']['transitions'],
                action_temperature=temperature, sampling_seed=c['sampling_seed'])
            write(out/'last_sampled_monitor.json', sampled)
        if updates == 0 and c.get('evaluate_initial_policy'):
            final_seeds = range(c['final_seed_start'], c['final_seed_start']+100)
            write(out/'initial_evaluation.json', evaluate(agent, final_seeds))
            if c.get('monitor_sampled'):
                initial_sampled = evaluate_sampled(agent, final_seeds, c['sampling_seed'], temperature=temperature)
                if not initial_sampled['summary']['complete'] or initial_sampled['summary']['truncated_episodes']:
                    raise RuntimeError('Initial sampled control did not complete')
                write(out/'initial_sampled_evaluation.json', initial_sampled)
            agent.save(out/'initial', metadata())
        write(out / 'curve.json', curve)
        if result['summary']['mean_score'] > best:
            best = result['summary']['mean_score']
            agent.save(out / 'best', metadata())
            write(out / 'best_monitor.json', result)
        last_monitor = transitions
        save()
        print('REINFORCE_MONITOR', json.dumps(curve[-1]), flush=True)
        if c.get('monitor_report'):
            subprocess.run([sys.executable, '-m', c['monitor_report']], check=False)
        evaluation_seconds += time.perf_counter() - before

    monitor()
    while not budget_completed():
        stop_reason = abort()
        if stop_reason:
            break
        before = time.perf_counter()
        data, completed, used, stop_reason = collect_complete_games(
            model, c['episode_batch'], c['collection_seed']+updates, rng, device, abort, temperature,
            spawn_safety=c['spawn_safety'])
        transitions += used
        collection_seconds += time.perf_counter() - before
        if stop_reason:
            break  # Never invent Monte Carlo returns for interrupted episodes.
        baselines = (leave_one_out_time_baseline(data['returns'], [e['length'] for e in completed])
                     if baseline_mode == 'leave_one_out_time' else np.zeros_like(data['returns']))
        weights = data['returns'] - baselines
        if updates == 0:
            identity = hashlib.sha256()
            for key in ('states', 'masks', 'actions', 'rewards', 'returns'):
                identity.update(np.ascontiguousarray(data[key]).tobytes())
            c['first_complete_batch_sha256'] = identity.hexdigest()
            length = completed[0]['length']
            write(out / 'first_training_game.json', dict(
                description='First complete stochastic game, before the first optimizer step. Tiles are stored as exponents; empty=0. Returns are actual remaining merge points/128.',
                episode=completed[0], collection_seed=c['collection_seed'], action_temperature=temperature,
                transitions=[dict(move=t, board_exponents=data['states'][t].tolist(),
                    legal_mask=data['legal_masks'][t].tolist(), policy_mask=data['masks'][t].tolist(), action=int(data['actions'][t]),
                    reward_points=float(data['rewards'][t]*128),
                    actual_return_points=float(data['returns'][t]*128),
                    baseline_points=float(baselines[t]*128),
                    learning_weight_points=float(weights[t]*128),
                    chosen_probability=float(np.exp(data['logp'][t]))) for t in range(length)]))
        before = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total_loss = entropy_sum = logp_error = 0.
        # eval mode still permits gradients and matches the behavior policy.
        # All chunk gradients use the SAME weights. Exactly one step follows.
        for start in range(0, len(data['actions']), c['batch']):
            if c.get('stop_file') and Path(c['stop_file']).exists():
                stop_reason = 'stop_file'
                break
            end = start + c['batch']
            logits = policy_logits(model, tensor_boards(data['states'][start:end], device))
            masks = torch.as_tensor(data['masks'][start:end], device=device)
            actions = torch.as_tensor(data['actions'][start:end], device=device)
            returns = torch.as_tensor(weights[start:end], device=device)
            loss = policy_loss(logits, masks, actions, returns, len(completed), temperature)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite REINFORCE loss')
            loss.backward()
            total_loss += float(loss.detach().cpu())
            with torch.no_grad():
                logs = action_log_probs(logits, masks, temperature)
                entropy_sum += float((-(logs.exp()*logs).sum(1)).sum().cpu())
                logp = logs.gather(1, actions[:, None]).squeeze(1).cpu().numpy()
                logp_error = max(logp_error, float(np.max(np.abs(logp-data['logp'][start:end]))))
        if stop_reason:
            optimizer.zero_grad(set_to_none=True)
            break
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c['max_grad_norm'], error_if_nonfinite=True)
        optimizer.step()
        updates += 1
        trained_transitions += used
        update_seconds += time.perf_counter() - before
        episodes.extend(e | dict(transitions=transitions, batch=updates) for e in completed)
        row = dict(transitions=transitions, updates=updates, training_seconds=elapsed(),
            action_temperature=temperature,
            mean_training_score=float(np.mean([e['score'] for e in completed])),
            completed_games=len(completed), batch_transitions=used, policy_loss=total_loss,
            return_mean=float(data['returns'].mean()), return_std=float(data['returns'].std()),
            baseline_mode=baseline_mode, baseline_mean=float(baselines.mean()),
            rows_with_filtered_legal_actions=int((data['legal_masks'] != data['masks']).any(1).sum()),
            learning_weight_mean=float(weights.mean()), learning_weight_std=float(weights.std()),
            entropy=entropy_sum/used, gradient_norm_before_clip=float(norm.cpu()),
            max_behavior_logp_difference=logp_error)
        progress.append(row)
        save()
        print('REINFORCE', json.dumps(row), flush=True)
        if transitions-last_monitor >= c['monitor_interval']:
            monitor()
    if stop_reason != 'stop_file' and last_monitor != transitions:
        monitor()
    save()
    result = metadata() | dict(complete=False, mean_score=None, completed_budget=budget_completed(),
        stop_reason=stop_reason, best_monitor_score=best, monitoring_seconds=evaluation_seconds)
    if stop_reason == 'stop_file':
        return result
    model.eval()
    final_seeds = range(c['final_seed_start'], c['final_seed_start']+100)
    final = evaluate(agent, final_seeds)
    write(out / 'evaluation.json', final)
    sampled_mean = None
    if c.get('monitor_sampled'):
        from research.policy_sampling_experiment import evaluate_sampled
        sampled = evaluate_sampled(agent, final_seeds, c['sampling_seed'], temperature=temperature)
        if not sampled['summary']['complete'] or sampled['summary']['truncated_episodes']:
            raise RuntimeError('A sampled endpoint game did not complete')
        write(out/'sampled_evaluation.json', sampled)
        sampled_mean = sampled['summary']['mean_score']
    save_replay(agent, out / 'replay_last.html', seed=8930100)
    selected = NeuralAgent.load(out / 'best', device)
    write(out / 'best_selection.json', evaluate(selected, final_seeds))
    save_replay(selected, out / 'replay_best.html', seed=8930100)
    return result | final['summary'] | dict(transitions=transitions, training_transitions=transitions,
        evaluation_transitions=final['summary']['transitions'], depth=c['depth'],
        sampled_mean_score=sampled_mean)
