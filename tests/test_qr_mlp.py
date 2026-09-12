import pytest
import torch

from rl2048.agents.mlp_q import MLPQNetwork
from rl2048.agents.qr_mlp import QuantileMLP
from rl2048.agents.qr_dqn import QRDQN
from rl2048.agents.neural import NeuralAgent


@pytest.mark.parametrize('encoding', ['exponents', 'embedding'])
def test_mlp_quantiles_match_scalar_initialization_and_roundtrip(tmp_path, encoding):
    torch.manual_seed(7)
    control = MLPQNetwork(16, encoding, 2)
    torch.manual_seed(7)
    model = QuantileMLP(16, encoding, 2, num_quantiles=5)
    boards = torch.randint(0, 12, (3, 16))
    assert model.quantile_values(boards).shape == (3, 4, 5)
    torch.testing.assert_close(model(boards), control(boards))
    agent = NeuralAgent(model, 'qr_dqn', width=16)
    agent.save(tmp_path)
    restored = NeuralAgent.load(tmp_path)
    assert restored.algorithm == 'qr_dqn'
    assert restored.policy.architecture == 'mlp_qr'
    torch.testing.assert_close(restored.policy.quantile_values(boards), model.quantile_values(boards))


def test_mlp_update_reaches_embedding_and_keeps_target_frozen():
    torch.manual_seed(4)
    learner = QRDQN(architecture='mlp_q', width=16, input_encoding='embedding', num_quantiles=5)
    boards = torch.randint(0, 12, (8, 16))
    batch = dict(states=boards, next_states=boards, actions=torch.arange(8) % 4,
        rewards=torch.ones(8), terminated=torch.ones(8, dtype=torch.bool),
        next_masks=torch.zeros(8, 4, dtype=torch.bool))
    before = learner.policy.quantile_values(boards).detach().clone()
    metrics = learner.update(batch)
    assert all(torch.isfinite(value) for value in metrics.values())
    assert not torch.equal(before, learner.policy.quantile_values(boards))
    assert learner.policy.embedding.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in learner.target.parameters())


def test_reject_engineered_features():
    with pytest.raises(ValueError):
        QuantileMLP(input_encoding='relational')
