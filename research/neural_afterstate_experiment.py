"""Online neural-only afterstate learning: a distinct formulation screen.

Data are new games played by this learner. Uniform replay stores
(afterstate x_t, actual spawned state s_(t+1), terminal). Training enumerates
the next state's legal slides. Optional expected targets average all possible
spawns during learning. Collection never copies a teacher or searches future
random spawn branches. Separate training/monitor/selection RNGs.
"""
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from rl2048.agents.afterstate_mlp import AfterstateLearner, AfterstateMLPAgent
from rl2048.afterstate_compare import evaluate_batch
from rl2048.offline_train import training_state
from rl2048.vector_game import VectorGame
from rl2048.view import save_replay
from research.transformer_td_experiment import write_json


class AfterstateReplay:
    def __init__(self, capacity):
        self.capacity, self.size, self.position = capacity, 0, 0
        self.afterstates = np.empty((capacity, 16), np.uint8)
        self.next_states = np.empty((capacity, 16), np.uint8)
        self.terminated = np.empty(capacity, bool)

    def add(self, afterstates, next_states, terminated):
        if len(afterstates) > self.capacity:
            raise ValueError('Insertion larger than replay capacity')
        indices = (np.arange(len(afterstates)) + self.position) % self.capacity
        self.afterstates[indices] = afterstates
        self.next_states[indices] = next_states
        self.terminated[indices] = terminated
        self.position = (self.position + len(afterstates)) % self.capacity
        self.size = min(self.capacity, self.size + len(afterstates))

    def sample(self, count, rng):
        indices = rng.integers(self.size, size=count)
        return self.afterstates[indices], self.next_states[indices], self.terminated[indices]


def evaluate(learner, seeds):
    result = evaluate_batch(None, seeds, decision=lambda b: learner.decision(b)[0])
    assert result['summary']['complete'] and not result['summary']['truncated_episodes']
    result['summary']['tile_reaching_rates'] = {
        str(t): float(np.mean([e['max_tile'] >= t for e in result['episodes']]))
        for t in (128, 256, 512, 1024, 2048, 4096, 8192)}
    result['summary']['action_selection'] = 'four exact root slides + scalar afterstate value; no future spawn search'
    return result


def train_one(c, seed, steps, out, device):
    if steps % c['envs'] or c.get('n', 1) != 1 or c['reward_mode'] != 'score':
        raise ValueError('Afterstate screen requires one-step raw-score targets and whole collection batches')
    update_ratio=c.get('updates_per_collection',1)
    if not isinstance(update_ratio,int) or isinstance(update_ratio,bool) or update_ratio<1:
        raise ValueError('updates_per_collection must be a positive integer')
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    has_teacher = bool(c.get('teacher_pretrain'))
    c = c | dict(seed=seed, steps=steps, device=device, teacher=has_teacher,
                 initialization='teacher_value_distillation' if has_teacher else 'resume' if c.get('resume') else 'scratch',
                 search_at_training=False, search_at_evaluation=False,
                 root_slide_enumeration=True,
                 target='U(x_t) <- r_(t+1)/128 + gamma U_target(x_(t+1)); online greedy next slide')
    if c.get('spawn_target') == 'expected':
        c.update(target='U(x_t) <- sum_spawn p(spawn|x_t) [r_next(a*)/128 + gamma U_target(x_next(a*))]; a* chosen online after that spawn',
                 model_based_target=True, search_at_training=True, search_for_action_collection=False,
                 training_search_purpose='Exact chance backup for learning targets; collection still uses only root slides',
                 spawn_expectation='all2/4 outcomes at every empty cell; no probability cutoff')
    write_json(out/'config.json', c)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    learner = AfterstateLearner(**c); prior = 0
    if c.get('resume'):
        previous = Path(c['resume']); old = json.loads((previous/'config.json').read_text())
        add_symmetry = (c.get('symmetry_warm_start_from_plain',False)
                        and old['architecture']=='afterstate_mlp'
                        and c['architecture']=='afterstate_sym_mlp')
        if c.get('symmetry_warm_start_from_plain') and not add_symmetry:
            raise ValueError('Symmetry conversion requires a plain MLP source and symmetric MLP destination')
        for key in ('algorithm', 'architecture', 'input_encoding', 'width', 'depth', 'embedding_dim', 'gamma'):
            if key=='architecture' and add_symmetry: continue
            if old[key] != c[key]:
                raise ValueError(f'Afterstate resume must preserve {key}')
        state = torch.load(previous/'training.pt', map_location='cpu', weights_only=True)
        for key, weights in state.items():
            if add_symmetry and key in ('policy','target'):
                # Same learned tensors and parameter order; only a fixed,
                # non-learned permutation buffer is new. Reject any other drift.
                mismatch=getattr(learner,key).load_state_dict(weights,strict=False)
                if mismatch.missing_keys!=['symmetry_permutations'] or mismatch.unexpected_keys:
                    raise ValueError('Unexpected tensor change in symmetry warm start')
            else:
                getattr(learner, key).load_state_dict(weights)
        for group in learner.optimizer.param_groups: group['lr'] = c['lr']
        prior = json.loads((previous/'last/metadata.json').read_text())['experiment']['total_transitions']
    teacher_result = None
    if has_teacher:
        from research.afterstate_teacher import warm_start
        teacher_result = warm_start(learner, c, out)
    agent = AfterstateMLPAgent(learner)
    env = VectorGame(c['envs'], c.get('collection_seed', seed))
    replay = AfterstateReplay(c['capacity'])
    curve, progress, episodes = [], [], []
    updates = processed = 0; best = -float('inf'); stop_reason = None
    start = time.perf_counter(); eval_seconds = update_seconds = game_seconds = 0.

    def elapsed(): return time.perf_counter()-start-eval_seconds

    def metadata():
        return c | dict(transitions=processed, training_transitions=processed,
                        total_transitions=prior+processed, prior_transitions=prior,
                        updates=updates, replay_examples_sampled=updates*c['batch'],
                        training_seconds=elapsed(), teacher_pretraining=teacher_result,
                        parameter_count=sum(p.numel() for p in learner.policy.parameters()))

    def save_logs():
        for filename, rows in [('progress', progress), ('episodes', episodes)]:
            if rows:
                with (out/f'{filename}.csv').open('w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)

    def monitor():
        nonlocal eval_seconds, best
        before = time.perf_counter(); training_seconds = elapsed()
        first = c['monitor_seed_start']
        check = evaluate(learner, range(first, first+c['monitor_games']))
        curve.append(check['summary'] | dict(transitions=processed, updates=updates,
                      training_seconds=training_seconds,
                      evaluation_transitions=check['summary']['transitions']))
        write_json(out/'curve.json', curve)
        if check['summary']['mean_score'] > best:
            best = check['summary']['mean_score']
            agent.save(out/'best', metadata())
            write_json(out/'best_monitor.json', check | dict(transitions=processed))
        agent.save(out/'last', metadata())
        torch.save(training_state(learner), out/'training.pt')
        save_logs()
        if c.get('monitor_report'):
            import subprocess, sys
            subprocess.run([sys.executable, '-m', c['monitor_report']], check=False)
        eval_seconds += time.perf_counter()-before

    if c.get('monitor_initial', True) and not (c.get('stop_file') and Path(c['stop_file']).exists()): monitor()
    for step in range(c['envs'], steps+1, c['envs']):
        if c.get('stop_file') and Path(c['stop_file']).exists():
            stop_reason = 'stop_file'; break
        if c.get('max_training_seconds') and elapsed() >= c['max_training_seconds']:
            stop_reason = 'training_time_budget'; break
        q, after, legal = learner.decision(env.boards)
        epsilon = 1-(1-c['epsilon_end'])*min(1, (prior+step)/c['epsilon_steps'])
        actions = q.argmax(1)
        random_actions = np.where(legal, rng.random(legal.shape), -np.inf).argmax(1)
        actions = np.where(rng.random(c['envs']) < epsilon, random_actions, actions)
        chosen_afterstates = after[np.arange(c['envs']), actions].copy()
        before = time.perf_counter()
        _, next_states, term, trunc, completed = env.step(actions)
        game_seconds += time.perf_counter()-before
        # VectorGame returns the pre-reset next state, so neither a termination
        # nor truncation ever stores a different game's initial board here.
        replay.add(chosen_afterstates, next_states, term)
        processed = step
        for score, length, tile in completed[completed[:, 1] > 0]:
            episodes.append(dict(transitions=step, score=int(score), length=int(length), max_tile=int(tile)))
        if step >= c['warmup'] and replay.size >= c['batch']:
            for _ in range(update_ratio):
                before = time.perf_counter()
                metrics = learner.update(*replay.sample(c['batch'], rng)); updates += 1
                update_seconds += time.perf_counter()-before
                if not all(np.isfinite(v) for v in metrics.values()):
                    stop_reason = 'nonfinite_update'; break
                progress.append(dict(transitions=step, updates=updates, training_seconds=elapsed(),
                                     epsilon=epsilon, **metrics))
                if updates % 256 == 0: print('TRAIN', json.dumps(progress[-1]), flush=True)
            if stop_reason == 'nonfinite_update':break
        if step % c['monitor_interval'] == 0 or step == steps: monitor()
    # A wall-time budget can end between scheduled checks. Include that final
    # frozen policy in the same monitoring protocol before choosing the best.
    if stop_reason == 'training_time_budget' and (not curve or curve[-1]['transitions'] != processed):
        monitor()
    agent.save(out/'last', metadata()); torch.save(training_state(learner), out/'training.pt'); save_logs()
    result = metadata() | dict(completed_budget=processed==steps, stop_reason=stop_reason,
             monitoring_seconds=eval_seconds, update_seconds=update_seconds,
             cpu_game_step_seconds=game_seconds, best_monitor_score=best if np.isfinite(best) else None,
             name='neural_' + c['architecture'], config=c, complete=False, mean_score=None)
    if c.get('spawn_target') == 'expected':
        result.update(simulated_spawn_outcomes=sum(row['simulated_spawn_outcomes'] for row in progress),
                      simulated_next_slides=sum(row['simulated_next_slides'] for row in progress))
    if stop_reason in ('stop_file', 'nonfinite_update'): return result
    first = c['final_seed_start']
    saved = AfterstateMLPAgent.load(out/'last', device)
    final = evaluate(saved.learner, range(first, first+100)); write_json(out/'evaluation.json', final)
    selected = AfterstateMLPAgent.load(out/'best', device)
    best_evaluation = evaluate(selected.learner, range(first, first+100))
    best_evaluation.update(checkpoint=str((out/'best').resolve()),
                           selection='Highest 128-game monitor mean; 100-game selection suite reused for tuning')
    write_json(out/'best_selection.json', best_evaluation)
    # A fixed complete diagnostic game, not a game cherry-picked for score.
    cpu_selected = AfterstateMLPAgent.load(out/'best', 'cpu')
    replay_result = save_replay(cpu_selected, out/'replay_best.html', seed=8930100)
    write_json(out/'replay_selection.json', dict(seed=8930100, result=replay_result))
    return result | final['summary'] | dict(training_transitions=processed,
                     evaluation_transitions=final['summary']['transitions'],
                     stop_reason=stop_reason,
                     depth=c['depth'], final_selection_mean_score=final['summary']['mean_score'])
