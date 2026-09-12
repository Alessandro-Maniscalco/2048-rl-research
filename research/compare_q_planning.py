"""Freeze each Q-network and compare direct argmax against one-step planning."""
import json
from pathlib import Path
import torch
from rl2048.agents.neural import NeuralAgent
from rl2048.agents.q_planning import PlanningQAgent
from rl2048.evaluate import evaluate


if __name__ == '__main__':
    torch.set_num_threads(2)
    root = Path('runs/research')
    results = {}
    for name in ('dqn16_exponents', 'dqn16_corner_snake', 'dqn16_long', 'dqn16_gamma1'):
        folder = root / name
        config = json.loads((folder / 'config.json').read_text())
        agent = NeuralAgent.load(folder / 'last')
        planner = PlanningQAgent(agent, config['gamma'], config['reward_mode'], config['shaping_scale'])
        summary, episodes = evaluate(planner, seeds=range(6_300_000, 6_300_100), max_steps=40_000)
        greedy = json.loads((folder / 'validation.json').read_text())['summary']
        results[name] = {'greedy': greedy, 'planning': summary, 'episodes': episodes,
                         'checkpoint': str(folder / 'last'), 'split': 'validation'}
        (root / 'q_planning_comparison.json').write_text(json.dumps(results, indent=2))
        print(name, 'greedy', greedy['mean_score'], 'planning', summary['mean_score'], flush=True)
