"""Evaluate the policy actually optimized by policy gradients, versus argmax.

No training, parameter updates, or tree search occurs. Any non-unit temperature
is explicit in the configuration and results. Each game has independent
recorded RNGs for tile spawns and action sampling.
"""
from pathlib import Path
import time
import numpy as np
import torch

from research.afterstate_teacher import write, digest
from rl2048.afterstate_compare import evaluate_batch
from rl2048.agents.neural import NeuralAgent, tensor_boards
from rl2048.agents.ntuple import row_tables
from rl2048.vector_game import legal_masks


def sample_preferences(logits, masks, ids, rngs):
    """One finite preferred move per live row; terminal rows stay all -inf."""
    result = np.full_like(logits, -np.inf, dtype=np.float64)
    for j, i in enumerate(ids):
        legal = np.flatnonzero(masks[j])
        if not len(legal):
            continue
        values = logits[j, legal].astype(np.float64)
        probabilities = np.exp(values-values.max())
        cumulative = probabilities.cumsum()
        draw = rngs[i].random()*cumulative[-1]
        action = legal[np.searchsorted(cumulative, draw, side='right')]
        result[j, action] = 1.
    return result


def evaluate_sampled(agent, seeds, sampling_seed, deadline=float('inf'), temperature=1.):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('A positive finite action temperature is required')
    seeds = list(seeds)
    rngs = [np.random.default_rng([sampling_seed, seed]) for seed in seeds]
    agent.policy.eval()

    @torch.no_grad()
    def decision(boards, ids):
        from rl2048.agents.neural import policy_logits
        logits = policy_logits(agent.policy, tensor_boards(boards, agent.device)).cpu().numpy()/temperature
        masks = agent.policy_mask(boards, legal_masks(boards, *row_tables()))
        return sample_preferences(logits, masks, ids, rngs)

    result = evaluate_batch(None, seeds, decision_with_ids=decision, deadline=deadline)
    result['summary'].update(evaluation_policy='categorical from legal softmax(logits / temperature)',
        sampling_seed=sampling_seed, sampling_rng='NumPy default_rng([sampling_seed, game_seed]) per game',
        temperature=temperature, future_search=False)
    if agent.spawn_safety:
        result['summary'].update(evaluation_policy='categorical softmax among legal minimum next-spawn death-risk moves',
            spawn_safety=True, one_spawn_dynamics_for_action_mask=True)
    return result


def train_one(c, seed, steps, out, device):
    """Queue-compatible frozen evaluation worker; the name does not imply learning."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write(out/'config.json', c | dict(device=device, training=False))
    records = []
    start = time.time()
    deadline = start+c['max_evaluation_seconds']
    seeds = range(c['game_seed_start'], c['game_seed_start']+c['games'])
    for item in c['checkpoints']:
        if Path(c['stop_file']).exists():
            return dict(complete=False, completed_budget=False, stop_reason='stop_file', records=records)
        path = Path(item['path'])
        checksum = digest(path/'policy.pt')
        agent = NeuralAgent.load(path, device)
        result = evaluate_sampled(agent, seeds, c['sampling_seed'], deadline, item.get('temperature',1.))
        assert digest(path/'policy.pt') == checksum
        write(out/(item['id']+'.json'), result)
        record = dict(id=item['id'], checkpoint=str(path), sha256=checksum,
                      **result['summary'])
        records.append(record)
        write(out/'summary.json', records)
        print('SAMPLING_EVALUATION', record, flush=True)
        del agent
        if not result['summary']['complete'] or result['summary']['truncated_episodes']:
            return dict(complete=False, completed_budget=False, stop_reason='evaluation_incomplete', records=records)
    return dict(complete=True, completed_budget=True, records=records, training_transitions=0,
                transitions=0, updates=0, evaluation_seconds=time.time()-start,
                diagnostic_only=True, mean_score=None)
