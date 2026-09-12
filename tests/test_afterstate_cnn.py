import numpy as np
import pytest
import torch

from rl2048.agents.afterstate_cnn import AfterstateCNN
from rl2048.agents.afterstate_mlp import AfterstateLearner, AfterstateMLPAgent
from rl2048.vector_game import VectorGame


def test_tile_channel_layout_and_whole_board_receptive_field():
    torch.set_num_threads(1)
    ranks = torch.arange(16).reshape(1, 16)
    model = AfterstateCNN(width=1, input_encoding='exponents')
    torch.testing.assert_close(model.encode_inputs(ranks)[0, 0], ranks.reshape(4, 4)/16)
    # With positive filters, a perturbation at ANY tile must reach the output.
    # This catches dropped rows/columns or accidental spatial pooling/truncation.
    with torch.no_grad():
        for parameter in model.parameters(): parameter.fill_(1.)
    empty = torch.zeros(1, 16)
    baseline = model(empty)
    for cell in range(16):
        board = empty.clone(); board[0, cell] = 1
        assert model(board).item() > baseline.item()
    embedded = AfterstateCNN(width=2, embedding_dim=3)
    encoded = embedded.encode_inputs(ranks)
    torch.testing.assert_close(encoded[0, :, 2, 3], embedded.embedding.weight[11])
    with pytest.raises(ValueError): AfterstateCNN(depth=2)
    with pytest.raises(ValueError): AfterstateCNN(input_encoding='relational')


def test_cnn_learning_and_saved_architecture_roundtrip(tmp_path):
    torch.set_num_threads(1); torch.manual_seed(17)
    learner = AfterstateLearner(architecture='afterstate_cnn', width=8, depth=3,
                               input_encoding='embedding', embedding_dim=4)
    env = VectorGame(16, 4321)
    q, after, _ = learner.decision(env.boards)
    actions = q.argmax(1)
    chosen = after[np.arange(16), actions]
    _, nxt, term, _, _ = env.step(actions)
    initial = {k: v.clone() for k, v in learner.policy.state_dict().items()}
    for _ in range(3):
        assert all(np.isfinite(v) for v in learner.update(chosen, nxt, term).values())
    assert not torch.equal(initial['embedding.weight'], learner.policy.embedding.weight)
    AfterstateMLPAgent(learner).save(tmp_path, dict(width=8))
    restored = AfterstateMLPAgent.load(tmp_path)
    assert isinstance(restored.learner.policy, AfterstateCNN)
    np.testing.assert_array_equal(learner.decision(nxt)[0], restored.learner.decision(nxt)[0])
