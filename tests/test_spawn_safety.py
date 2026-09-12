import numpy as np
import pytest

from rl2048.agents.spawn_safety import immediate_death_risks, safest_policy_scores
from rl2048.agents.q_planning import outcomes
from rl2048.agents.ntuple import encode, row_tables
from rl2048.vector_game import legal_masks


def test_gpt_final_decision_has_avoidable_ten_percent_risk():
    board = np.array([[2,2,32,8], [0,8,128,16], [32,256,64,2], [1024,32,16,4]])
    boards = encode(board)[None]
    risk, legal = immediate_death_risks(boards)
    np.testing.assert_array_equal(legal, np.ones((1,4), bool))
    np.testing.assert_allclose(risk, [[0., 0., .1, 0.]])
    scores, _, _ = safest_policy_scores(np.array([[0.,1.,10.,2.]]), boards)
    assert scores.argmax(1).item() == 3  # left, not the preferred risky down


def test_optimized_risk_matches_exhaustive_spawn_enumeration():
    rng = np.random.default_rng(8958250)
    boards = rng.integers(1, 10, size=(160,16), dtype=np.uint8)
    for i,board in enumerate(boards):
        board[rng.choice(16, i%7, replace=False)] = 0
    before = boards.copy()
    risk, legal = immediate_death_risks(boards)
    expected = np.where(legal, 0., np.inf)
    for i,board in enumerate(boards):
        spawned, actions, probabilities, _ = outcomes(board, *row_tables())
        terminal = ~legal_masks(spawned, *row_tables()).any(1)
        for action in np.flatnonzero(legal[i]):
            pick = actions == action
            assert probabilities[pick].sum() == pytest.approx(1.)
            expected[i,action] = probabilities[pick]@terminal[pick]
    np.testing.assert_allclose(risk, expected, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(boards, before)
    logits = rng.normal(size=(160,4))
    scores, _, _ = safest_policy_scores(logits, boards)
    for i in np.flatnonzero(legal.any(1)):
        action = scores[i].argmax()
        assert legal[i, action] and risk[i,action] == risk[i].min()
        if np.all(risk[i,legal[i]] == risk[i].min()):
            assert action == np.where(legal[i],logits[i],-np.inf).argmax()
    assert np.isneginf(scores[~legal.any(1)]).all()


def test_safety_wrapper_supports_shared_replay_and_restores_rng(tmp_path):
    import json
    import torch
    from research.cnn_safety_study import SpawnSafetyAgent
    from rl2048.agents.neural import BoardNet, NeuralAgent
    from rl2048.view import save_replay
    torch.set_num_threads(1)
    player = SpawnSafetyAgent(NeuralAgent(BoardNet(4, 8), 'test'))
    previous = player.rng
    result = save_replay(player, tmp_path/'replay.html', seed=8958251, max_steps=16)
    assert player.rng is previous
    assert result['length'] == 16 and result['truncated']
    saved = json.loads((tmp_path/'replay.json').read_text())
    assert len(saved['frames']) == 17 and (tmp_path/'replay.html').exists()


def test_serialized_policy_filter_and_masked_gradient(tmp_path):
    import torch
    from rl2048.agents.neural import BoardNet, NeuralAgent
    from rl2048.agents.reinforce import policy_loss
    from rl2048.agents.spawn_safety import minimum_risk_mask
    board=np.array([[2,2,32,8],[0,8,128,16],[32,256,64,2],[1024,32,16,4]])
    boards=encode(board)[None]
    legal=legal_masks(boards,*row_tables())
    mask=minimum_risk_mask(boards,legal)
    np.testing.assert_array_equal(mask,[[True,True,False,True]])
    logits=torch.tensor([[0.,1.,10.,2.]],requires_grad=True)
    policy_loss(logits,torch.as_tensor(mask),torch.tensor([3]),torch.tensor([5.]),1).backward()
    assert logits.grad[0,2]==0  # The risky-but-legal action is excluded from this policy.
    net=BoardNet(4,8)
    with torch.no_grad():
        for p in net.parameters():p.zero_()
        net.layers[-1].bias.copy_(logits.detach()[0])
    plain=NeuralAgent(net,'reinforce',width=8)
    safe=NeuralAgent(net,'reinforce',width=8,spawn_safety=True)
    assert plain.act(board,legal[0])==2 and safe.act(board,legal[0])==3
    safe.save(tmp_path/'safe')
    restored=NeuralAgent.load(tmp_path/'safe')
    assert restored.spawn_safety and restored.act(board,legal[0])==3
    with pytest.raises(ValueError):minimum_risk_mask(boards,~legal)
