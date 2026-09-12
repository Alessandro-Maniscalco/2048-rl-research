import numpy as np
import torch

from research.policy_sampling_experiment import sample_preferences, evaluate_sampled
from rl2048.agents.neural import NeuralAgent
from rl2048.agents.reinforce import PolicyTransformer


def test_sampling_masks_and_terminal_rows():
    rngs = [np.random.default_rng(i) for i in range(3)]
    logits = np.array([[1000., 0., -1., 0.], [2., 2., 2., 2.], [0., 0., 0., 0.]])
    masks = np.array([[False, True, False, False], [False]*4, [True, False, True, False]])
    preferences = sample_preferences(logits, masks, np.arange(3), rngs)
    assert preferences[0].argmax() == 1
    assert not np.isfinite(preferences[1]).any()
    assert preferences[2].argmax() in (0, 2)


def test_sampled_complete_games_are_reproducible_and_order_independent():
    torch.set_num_threads(1)
    torch.manual_seed(3)
    agent = NeuralAgent(PolicyTransformer(8, 'embedding', 1, 2), 'reinforce', width=8)
    before = {k: v.clone() for k, v in agent.policy.state_dict().items()}
    seeds = [12930101, 12930102, 12930103]
    first = evaluate_sampled(agent, seeds, 52, temperature=.25)
    second = evaluate_sampled(agent, seeds[::-1], 52, temperature=.25)
    assert first['summary']['complete'] and not first['summary']['truncated_episodes']
    assert first['summary']['temperature'] == .25
    assert first['episodes'] == second['episodes'][::-1]
    for k, v in before.items():
        torch.testing.assert_close(v, agent.policy.state_dict()[k], rtol=0, atol=0)
