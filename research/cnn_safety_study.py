"""Frozen CNN ablation: does avoiding immediate spawn death improve raw score?"""
import json
from pathlib import Path
import time

import numpy as np
import torch

from research.afterstate_teacher import digest, write
from rl2048.agents.neural import NeuralAgent, policy_logits, tensor_boards
from rl2048.agents.ntuple import encode
from rl2048.agents.spawn_safety import safest_policy_scores
from rl2048.afterstate_compare import evaluate_batch
from rl2048.view import save_replay


class SpawnSafetyAgent:
    name = 'neural_current_state_with_spawn_safety'
    display_name = 'Frozen CNN plus exact immediate game-over risk check'

    def __init__(self, agent):
        self.agent = agent
        self.rng = np.random.default_rng(0)  # Common replay interface; greedy policy does not sample it.

    @torch.no_grad()
    def act(self, board, action_mask):
        boards = encode(board)[None]
        logits = policy_logits(self.agent.policy, tensor_boards(boards, self.agent.device)).cpu().numpy()
        scores, _, legal = safest_policy_scores(logits, boards)
        np.testing.assert_array_equal(legal[0], action_mask)
        return int(scores.argmax(1)[0])


def main():
    torch.set_num_threads(1)
    root = Path('runs/research/scaled_transformer')
    out = root/'cnn_spawn_safety'
    out.mkdir(exist_ok=False)
    source = Path('runs/research/pretrained_cnn/original')
    checksum = digest(source/'policy.pt')
    seeds = list(range(8958300, 8958400))
    write(out/'protocol.json', dict(source=str(source.resolve()), source_sha256=checksum,
        device='cpu', cpu_threads=1, seeds=seeds, max_steps=100000, per_arm_seconds=1200,
        reason='The latest CNN replay audit and genuine GPT game both contained avoidable immediate '
        'game-over risk. Compare the same frozen original CNN with and without choosing its highest '
        'logit among minimum-risk legal actions. Compute exact one-spawn risk from known rules. '
        'Do not add a corner rule, change weights or alter rewards. Immediate survival can sacrifice '
        'long-term points, so use100preselected fresh paired games and retain both full endpoints.',
        limitation='This changes inference and uses one-move game dynamics; it is not learned neural generalization. '
        'Both arms run on CPU to avoid confounding device changes and leave MPS to the adaptation study.'))
    agent = NeuralAgent.load(source, 'cpu')
    results = {}
    for label,shield in [('baseline', False), ('safety', True)]:
        counts = dict(active_decisions=0, changed_decisions=0, sum_risk_reduction=0.)
        @torch.no_grad()
        def decision(boards):
            if (root/'STOP').exists():
                raise InterruptedError('user_stop')
            logits = policy_logits(agent.policy, tensor_boards(boards, 'cpu')).numpy()
            safe, risks, legal = safest_policy_scores(logits, boards)
            plain = np.where(legal, logits, -np.inf)
            if shield:
                valid = np.flatnonzero(legal.any(1))
                old, new = plain.argmax(1), safe.argmax(1)
                counts['active_decisions'] += len(valid)
                counts['changed_decisions'] += int((old[valid] != new[valid]).sum())
                counts['sum_risk_reduction'] += float((risks[valid,old[valid]]-risks[valid,new[valid]]).sum())
            return safe if shield else plain
        write(out/'status.json', dict(phase='evaluating', arm=label))
        result = evaluate_batch(None, seeds, decision=decision, deadline=time.time()+1200, max_steps=100000)
        result['intervention_counts'] = counts
        result['timing_note'] = 'Both arms compute risks for a matched diagnostic path; this is not unmodified baseline inference speed.'
        write(out/f'{label}.json', result)
        if not result['summary']['complete'] or result['summary']['truncated_episodes']:
            write(out/'status.json', dict(phase='incomplete', arm=label))
            return
        results[label] = result
        print(label, json.dumps(result['summary'] | counts), flush=True)
    by_seed = {k:{e['seed']:e['score'] for e in r['episodes']} for k,r in results.items()}
    difference = np.array([by_seed['safety'][s]-by_seed['baseline'][s] for s in seeds])
    boot = difference[np.random.default_rng(8958400).integers(100,size=(20000,100))].mean(1)
    summary = dict(complete=True, baseline_mean=results['baseline']['summary']['mean_score'],
        safety_mean=results['safety']['summary']['mean_score'], gain=float(difference.mean()),
        wins=int((difference>0).sum()), ties=int((difference==0).sum()),
        paired_game_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
        intervention_counts=results['safety']['intervention_counts'],
        note='Frozen original CNN,100fresh preselected paired games, weights unchanged. One-spawn risk minimization need not optimize long-term points.')
    write(out/'comparison.json', summary)
    for label,player in [('baseline', agent), ('safety', SpawnSafetyAgent(agent))]:
        save_replay(player, out/f'{label}_replay.html', seed=8958300, max_steps=100000)
    assert digest(source/'policy.pt') == checksum
    write(out/'verified.json', dict(source_weights_unchanged=True, complete=True))
    write(out/'status.json', dict(phase='complete'))
    print('COMPARISON',json.dumps(summary),flush=True)


if __name__ == '__main__':
    try:
        main()
    except InterruptedError:
        write(Path('runs/research/scaled_transformer/cnn_spawn_safety/status.json'),dict(phase='stopped',reason='user_stop'))
