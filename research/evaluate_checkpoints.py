"""Validation only; published weights are explicitly separate from local learning."""
import json
from pathlib import Path
from rl2048.agents.ntuple import NTupleAgent
from rl2048.ntuple_train import evaluate_fast

if __name__ == '__main__':
    for name in ['native_otd', 'published/checkpoint']:
        agent = NTupleAgent.load(Path('runs/research') / name, mmap_mode='r')
        results = {}
        for depth in (1, 2):
            summary, records = evaluate_fast(agent, range(6_000_000, 6_000_100), depth=depth, workers=6)
            results[str(depth)] = {'summary': summary, 'episodes': records}
            target = Path('runs/research') / name / 'validation.json'
            target.write_text(json.dumps(results, indent=2))
            print(name, depth, json.dumps(summary), flush=True)
