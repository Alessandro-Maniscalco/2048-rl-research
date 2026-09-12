"""A frozen-policy screen: use different parts of the learned return distribution.

This is an inference experiment, not IQN or risk-sensitive retraining. QR was
trained using mean-greedy continuation; its quantiles are not calibrated returns
for the new policies. Raw game score, not a risk-adjusted metric, judges success.

Motivation: Section 4 of https://proceedings.mlr.press/v80/dabney18a.html
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from rl2048.agents.neural import NeuralAgent, tensor_boards
from rl2048.agents.ntuple import encode, row_tables
from rl2048.afterstate_compare import evaluate_batch
from rl2048.vector_game import legal_masks
from rl2048.view import save_replay


MODES = ('mean', 'lower_half', 'median', 'upper_half')


def aggregate_quantiles(values, mode):
    """Integrate quantile bins; do not sort away quantile-crossing diagnostics.

    Head i represents the bin [i/N, (i+1)/N]. When N=51, the middle
    head contributes half of its bin to each half-distribution average.
    """
    n = values.shape[-1]
    if mode == 'mean':
        return values.mean(-1)
    if mode == 'median':
        return (values[..., (n - 1) // 2] + values[..., n // 2]) / 2
    if mode not in ('lower_half', 'upper_half'):
        raise ValueError(f'Unknown quantile aggregation: {mode}')
    left = torch.arange(n, device=values.device, dtype=values.dtype) / n
    right = (torch.arange(n, device=values.device, dtype=values.dtype) + 1) / n
    if mode == 'lower_half':
        weights = (right.clamp(max=.5) - left).clamp(min=0) / .5
    else:
        weights = (right - left.clamp(min=.5)).clamp(min=0) / .5
    return (values * weights).sum(-1)


class StopRequested(Exception):
    pass


class QuantileRiskAgent:
    def __init__(self, agent, mode, stop_file=None):
        self.agent, self.mode, self.stop_file = agent, mode, stop_file
        self.rng = np.random.default_rng(0)
        self.algorithm = f'qr_frozen_{mode}'
        self.display_name = f'Frozen QR Transformer: {mode.replace("_", " ")}'
        self.crossed = self.pairs = 0

    @torch.no_grad()
    def action_values(self, boards):
        if self.stop_file and self.stop_file.exists():
            raise StopRequested('User stop requested')
        values = self.agent.policy.quantile_values(tensor_boards(boards, 'cpu'))
        masks = legal_masks(boards, *row_tables())
        legal_values = values[torch.from_numpy(masks)]
        self.crossed += int((legal_values[:, 1:] < legal_values[:, :-1]).sum())
        self.pairs += len(legal_values) * (values.shape[-1] - 1)
        q = aggregate_quantiles(values, self.mode).numpy()
        return np.where(masks, q, -np.inf)

    def act(self, board, action_mask):
        values = self.action_values(encode(board)[None])[0]
        return int(np.where(action_mask, values, -np.inf).argmax())


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def run(args):
    if args.stop_file and args.stop_file.exists():
        raise StopRequested('User stop requested')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    agent = NeuralAgent.load(args.checkpoint, 'cpu')
    assert agent.algorithm == 'qr_dqn'
    agent.policy.eval()
    metadata = json.loads((args.checkpoint / 'metadata.json').read_text())
    agent.save(args.out / 'frozen', metadata['experiment'])
    protocol = dict(
        hypothesis='Averaging the lower return half may avoid catastrophic moves; the upper half tests whether more ambitious choices help large-tile growth. Median tests sensitivity to the tails. All may be worse than the mean, which matches our score objective.',
        control='Same frozen checkpoint, 100 selection seeds, legal masks and CPU execution. Only aggregation changes; no training or search.',
        modes=list(MODES), source=str(args.checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256((args.out / 'frozen/policy.pt').read_bytes()).hexdigest(),
        seeds=list(range(args.seed_start, args.seed_start + args.games)),
        total_budget_seconds=args.seconds, per_mode_budget_seconds=180,
        source_paper='https://proceedings.mlr.press/v80/dabney18a.html',
        limitation='The paper changes the acting and training policies. This cheaper frozen-QR screen is not a replication. Quantiles describe return variability, not confidence about the model. Crossing and off-policy mismatch limit a literal CVaR interpretation.',
    )
    write_json(args.out / 'protocol.json', protocol)
    deadline = time.time() + args.seconds
    results = {}
    try:
        for mode in MODES:
            if time.time() >= deadline:
                break
            actor = QuantileRiskAgent(agent, mode, args.stop_file)
            result = evaluate_batch(None, protocol['seeds'], decision=actor.action_values,
                                    deadline=min(deadline, time.time() + 180))
            summary = result['summary']
            summary.pop('depth', None)  # The shared evaluator defaults to 1; this policy has NO search.
            summary['planning_depth'] = 0
            if summary['complete']:
                assert not summary['truncated_episodes']
                summary['tile_reaching_rates'] = {
                    str(t): float(np.mean([e['max_tile'] >= t for e in result['episodes']]))
                    for t in (1024, 2048, 4096, 8192)}
                summary['maximum_game_score'] = max(e['score'] for e in result['episodes'])
            result['adjacent_quantile_crossing_rate'] = actor.crossed / max(1, actor.pairs)
            results[mode] = result
            write_json(args.out / f'{mode}.json', result)
            write_json(args.out / 'results.json', results)
            print(mode, json.dumps(summary), flush=True)
            if not summary['complete']:
                break
        # One diagnostic seed across policies, separate from the 100-game score.
        for mode, result in results.items():
            if time.time() >= deadline or not result['summary']['complete']:
                break
            actor = QuantileRiskAgent(agent, mode, args.stop_file)
            game = save_replay(actor, args.out / f'replay_{mode}.html', seed=8930100)
            write_json(args.out / f'replay_{mode}_summary.json', game)
        write_json(args.out / 'status.json', {'status': 'complete' if len(results) == len(MODES)
            and all(r['summary']['complete'] for r in results.values()) else 'budget_reached'})
    except StopRequested:
        write_json(args.out / 'status.json', {'status': 'stopped', 'completed_modes': list(results)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed-start', type=int, default=8910000)
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--seconds', type=float, default=600)
    parser.add_argument('--stop-file', type=Path)
    run(parser.parse_args())
