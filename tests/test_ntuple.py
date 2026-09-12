import numpy as np
import pytest

from rl2048.agents.ntuple import NTupleAgent, decode, encode, make_patterns, row_tables
from rl2048.fast2048 import chance_value, evaluate_game, greedy, merge_exponents, slide, train_batch, tuple_index, update_value, value
from rl2048.game import Game2048, legal_actions, move


def test_compiled_rules_match_reference_over_diverse_boards():
    rng = np.random.default_rng(123)
    moves, rewards = row_tables()
    for _ in range(500):
        exponents = rng.integers(0, 18, 16, dtype=np.uint8)
        board = decode(exponents)
        for action in range(4):
            expected, reward, changed = move(board, action)
            actual, gain, moved = slide(exponents, action, moves, rewards)
            np.testing.assert_array_equal(decode(actual), expected)
            assert gain == reward and moved == changed


def test_high_tiles_do_not_wrap_and_row_merges_once():
    after, reward = merge_exponents(np.array([15, 15, 16, 16], np.uint8))
    np.testing.assert_array_equal(after, [16, 17, 0, 0])
    assert reward == 65536 + 131072
    after, reward = merge_exponents(np.array([1, 1, 2, 0], np.uint8))
    np.testing.assert_array_equal(after, [2, 2, 0, 0])
    assert reward == 4


def test_tuple_indices_symmetry_and_optimism():
    agent = NTupleAgent("4x4", initial_value=32000)
    board = np.array([[2, 4, 8, 0], [16, 32, 0, 2], [4, 0, 0, 0], [0, 2, 0, 0]])
    assert agent.afterstate_value(board) == 32000
    assert tuple_index(np.array([1, 2, 3, 4], np.uint8), np.arange(4)) == 0x4321
    rng = np.random.default_rng(3)
    agent.weights[:] = rng.normal(size=agent.weights.shape)
    expected = agent.afterstate_value(board)
    for k in range(4):
        assert agent.afterstate_value(np.rot90(board, k)) == pytest.approx(expected)
        assert agent.afterstate_value(np.rot90(np.fliplr(board), k)) == pytest.approx(expected)


def test_normalized_update_and_temporal_coherence():
    # Distinct single-cell indices: no repeated symmetric features in this fixture.
    patterns = np.arange(8).reshape(1, 8, 1)
    board = np.arange(16, dtype=np.uint8)
    weights = np.zeros((1, 16), np.float32)
    sums, absolute = np.zeros_like(weights), np.zeros_like(weights)
    error = update_value(board, 80, weights, patterns, .1, sums, absolute, False)
    assert error == 80 and value(board, weights, patterns) == 8
    np.testing.assert_allclose(weights[0, :8], 1)
    update_value(board, 88, weights, patterns, 1., sums, absolute, True)
    assert value(board, weights, patterns) == 88
    assert sums[0, 0] == absolute[0, 0] == 80
    update_value(board, 8, weights, patterns, 1., sums, absolute, True)
    assert sums[0, 0] == 0 and absolute[0, 0] == 160
    before = weights.copy()
    update_value(board, 88, weights, patterns, 1., sums, absolute, True)
    np.testing.assert_array_equal(weights, before)  # old coherence is now zero


@pytest.mark.parametrize('use_tc', [False, True])
def test_repeated_features_have_controlled_value_step(use_tc):
    from rl2048.fast2048 import feature_squared_norm
    # Eight references to one weight: V=8w and ||features||^2=64.
    patterns=np.zeros((1,8,1),np.int64)
    board=np.zeros(16,np.uint8)
    assert feature_squared_norm(board,patterns)==64
    for normalize,expected in [(False,64.),(True,8.)]:
        weights=np.zeros((1,16),np.float32)
        sums=np.zeros_like(weights);absolute=np.zeros_like(weights)
        update_value(board,80,weights,patterns,.1,sums,absolute,use_tc,normalize)
        assert value(board,weights,patterns)==pytest.approx(expected)


def test_collision_normalization_matches_unique_feature_update():
    from rl2048.fast2048 import feature_squared_norm
    patterns=np.arange(8).reshape(1,8,1)
    board=np.arange(16,dtype=np.uint8)
    assert feature_squared_norm(board,patterns)==8
    weights=np.zeros((1,16),np.float32)
    update_value(board,80,weights,patterns,.1,weights.copy(),weights.copy(),False,True)
    assert value(board,weights,patterns)==pytest.approx(8.)


def test_expectimax_probability_and_reward_timing():
    agent = NTupleAgent("4x4", 0)
    rows, rewards = row_tables()
    after = encode(np.array([[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 2048, 4096], [8192, 16384, 0, 0]]))
    expected = 0.
    for pos in [14, 15]:
        for exponent, probability in [(1, .45), (2, .05)]:
            board = after.copy()
            board[pos] = exponent
            best = 0
            for action in range(4):
                _, gain, moved = slide(board, action, rows, rewards)
                if moved:
                    best = max(best, gain)
            expected += probability * best
    actual = chance_value(after, agent.weights, agent.patterns, rows, rewards, 1, 1., 0.)
    assert actual == pytest.approx(expected)
    assert chance_value(after, agent.weights, agent.patterns, rows, rewards, 0, 1., 0.) == 0


def test_training_reproducibility_checkpoint_and_reference_game(tmp_path):
    first, second = NTupleAgent("4x4"), NTupleAgent("4x4")
    dummy = np.zeros((1, 1), np.float32)
    rows, rewards = row_tables()
    a = train_batch(first.weights, first.patterns, rows, rewards, 10, 19, .1, dummy, dummy, False)
    b = train_batch(second.weights, second.patterns, rows, rewards, 10, 19, .1, dummy, dummy, False)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(first.weights, second.weights)
    first.save(tmp_path / "checkpoint")
    loaded = NTupleAgent.load(tmp_path / "checkpoint")
    np.testing.assert_array_equal(first.weights, loaded.weights)
    env = Game2048()
    board, info = env.reset(seed=5000000)
    total = 0
    for step in range(40000):
        action = loaded.act(board, info["action_mask"])
        board, reward, terminated, _, info = env.step(action)
        total += reward
        if terminated:
            break
    assert terminated and total == info["score"] and not legal_actions(board).any()


def test_tile_downgrade_preserves_legal_moves_and_original_board():
    from rl2048.fast2048 import downgrade_root
    from rl2048.agents.ntuple import decode
    from rl2048.game import legal_actions
    board=np.array([15,14,12,11,9,8,7,6,5,4,3,2,1,0,1,2],dtype=np.uint8)
    original=board.copy()
    result=downgrade_root(board,32768)
    # Missing rank13 is highest gap; only ranks14 and15 fall by one.
    expected=board.copy(); expected[:2]-=1
    np.testing.assert_array_equal(result,expected)
    np.testing.assert_array_equal(board,original)
    np.testing.assert_array_equal(legal_actions(decode(board)),legal_actions(decode(result)))
    np.testing.assert_array_equal(downgrade_root(board,65536),board)
