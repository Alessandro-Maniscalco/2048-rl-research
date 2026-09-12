"""Compare frozen pretrained and fine-tuned policies on identical complete games.

Run: uv run python research/evaluate_pretrained.py --split validation
The test split must only be used after choosing training settings.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from rl2048.agents.neural import NeuralAgent
from rl2048.afterstate_compare import evaluate_batch
from rl2048.agents.ntuple import row_tables
from rl2048.vector_game import legal_masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / 'runs/research/pretrained_cnn'
    torch.set_num_threads(2)
    start = 8400000 if args.split == 'validation' else 8500000
    results = {}
    paths = {'original': root / 'original', 'finetuned': root / 'finetune/last'}
    manifest = {name: hashlib.sha256((path / 'policy.pt').read_bytes()).hexdigest()
                for name, path in paths.items()}
    (root / f'{args.split}_frozen_checkpoints.json').write_text(json.dumps(manifest, indent=2))
    for name, path in paths.items():
        agent = NeuralAgent.load(path, 'mps')

        @torch.no_grad()
        def decision(boards):
            logits = agent.policy(torch.as_tensor(boards, dtype=torch.long, device='mps'))[:, :4]
            return np.where(legal_masks(boards, *row_tables()), logits.cpu().numpy(), -np.inf)

        result = evaluate_batch(None, range(start, start + 100),
                                deadline=time.time() + 120, decision=decision)
        result['summary'].pop('depth')
        result['summary']['decision'] = 'greedy legal policy logits, no planning'
        result['checkpoint_sha256'] = manifest[name]
        (root / f'{name}_{args.split}.json').write_text(json.dumps(result, indent=2))
        results[name] = result
        print(name, args.split, result['summary'], flush=True)
    if all(r['summary']['complete'] for r in results.values()):
        delta = np.array([b['score'] - a['score'] for a, b in zip(
            results['original']['episodes'], results['finetuned']['episodes'], strict=True)])
        radius = 1.96 * delta.std(ddof=1) / np.sqrt(len(delta))
        comparison = {'paired_mean_score_difference': float(delta.mean()),
                      'approximate_95_percent_ci': [float(delta.mean() - radius), float(delta.mean() + radius)],
                      'note': 'Variation across game seeds; one fine-tuning seed, not variation across training seeds.'}
        (root / f'comparison_{args.split}.json').write_text(json.dumps(comparison, indent=2))
        print(comparison, flush=True)


if __name__ == '__main__':
    main()
