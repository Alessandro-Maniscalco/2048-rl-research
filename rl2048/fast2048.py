"""Compiled game rules and n-tuple kernels; game.py remains the readable oracle.

Boards here contain exponents (0=empty, 1=2, 2=4), not actual tile values.
Numba compiles these ordinary Python functions with LLVM and releases the GIL.
Rows use a 65,536-entry move table. Tiles >=65536 use an exact fallback, so
the simulator never wraps a large tile back to an empty cell.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def merge_exponents(row):
    packed = np.zeros(4, dtype=np.uint8)
    count = 0
    for j in range(4):
        if row[j]:
            packed[count] = row[j]
            count += 1
    result = np.zeros(4, dtype=np.uint8)
    j = 0
    out = 0
    reward = 0
    while j < count:
        if j + 1 < count and packed[j] == packed[j + 1]:
            result[out] = packed[j] + 1
            reward += 1 << int(result[out])
            j += 2
        else:
            result[out] = packed[j]
            j += 1
        out += 1
    return result, reward


@njit(cache=True)
def make_row_tables():
    rows = np.zeros((65536, 4), dtype=np.uint8)
    rewards = np.zeros(65536, dtype=np.int64)
    for index in range(65536):
        row = np.empty(4, dtype=np.uint8)
        for j in range(4):
            row[j] = (index >> (4 * j)) & 15
        rows[index], rewards[index] = merge_exponents(row)
    return rows, rewards


@njit(cache=True, inline="always")
def location(action, line, j):
    if action == 0:
        return j * 4 + line
    if action == 1:
        return line * 4 + 3 - j
    if action == 2:
        return (3 - j) * 4 + line
    return line * 4 + j


@njit(cache=True)
def slide(board, action, row_moves, row_rewards):
    result = np.empty(16, dtype=np.uint8)
    reward = 0
    changed = False
    for line in range(4):
        index = 0
        high = False
        for j in range(4):
            exponent = int(board[location(action, line, j)])
            index |= exponent << (4 * j)
            high = high or exponent > 15
        if high:
            row = np.empty(4, dtype=np.uint8)
            for j in range(4):
                row[j] = board[location(action, line, j)]
            merged, gain = merge_exponents(row)
        else:
            merged, gain = row_moves[index], row_rewards[index]
        reward += gain
        for j in range(4):
            pos = location(action, line, j)
            result[pos] = merged[j]
            changed = changed or result[pos] != board[pos]
    return result, reward, changed


@njit(cache=True)
def spawn(board):
    empty = np.empty(16, dtype=np.int64)
    count = 0
    for j in range(16):
        if board[j] == 0:
            empty[count] = j
            count += 1
    if count:
        board[empty[np.random.randint(count)]] = 1 if np.random.random() < .9 else 2


@njit(cache=True, inline="always")
def tuple_index(board, positions):
    index = 0
    for j in range(positions.shape[0]):
        # The game remains exact; only extremely rare >=65536 patterns alias
        # the largest learned category, matching a 16-category table budget.
        exponent = min(int(board[positions[j]]), 15)
        index |= exponent << (4 * j)
    return index


@njit(cache=True)
def value(board, weights, patterns):
    total = 0.0
    for p in range(patterns.shape[0]):
        for symmetry in range(8):
            total += weights[p, tuple_index(board, patterns[p, symmetry])]
    return total


@njit(cache=True)
def feature_squared_norm(board, patterns):
    """Squared norm of the sparse feature vector, including repeated symmetries.

    A shared weight used c times has feature value c, so it contributes c*c.
    Counting matching ordered pairs avoids a dictionary inside the hot loop.
    """
    squared_norm = 0
    indices = np.empty(8, np.int64)
    for p in range(patterns.shape[0]):
        for i in range(8):
            indices[i] = tuple_index(board, patterns[p, i])
        for i in range(8):
            for j in range(8):
                squared_norm += indices[i] == indices[j]
    return squared_norm


@njit(cache=True)
def update_value(board, target, weights, patterns, alpha, tc_sum, tc_abs, use_tc,
                 normalize_collisions=False):
    error = target - value(board, weights, patterns)
    denominator = feature_squared_norm(board, patterns) if normalize_collisions else patterns.shape[0] * 8
    increment = alpha * error / denominator
    for p in range(patterns.shape[0]):
        for symmetry in range(8):
            idx = tuple_index(board, patterns[p, symmetry])
            coherence = 1.0
            if use_tc:
                if tc_abs[p, idx] > 0:
                    coherence = abs(tc_sum[p, idx]) / tc_abs[p, idx]
                tc_sum[p, idx] += error
                tc_abs[p, idx] += abs(error)
            weights[p, idx] += increment * coherence
    return error


@njit(cache=True)
def greedy(board, weights, patterns, row_moves, row_rewards):
    best = -1.0e30
    best_action = -1
    best_after = board.copy()
    best_reward = 0
    for action in range(4):
        after, reward, changed = slide(board, action, row_moves, row_rewards)
        if changed:
            estimate = reward + max(0.0, value(after, weights, patterns))
            if estimate > best:
                best = estimate
                best_action = action
                best_after = after
                best_reward = reward
    return best_action, best_after, best_reward, best


@njit(cache=True, nogil=True)
def train_batch(weights, patterns, row_moves, row_rewards, games, seed, alpha,
                tc_sum, tc_abs, use_tc, max_steps=40000, normalize_collisions=False):
    """Greedy self-play followed by backward afterstate TD(0), gamma=1.

    The target for afterstate t uses reward[t+1], not reward[t]. At the final
    afterstate target=0. At a move cap we bootstrap rather than invent a loss.
    RNG is thread-local and seeded once per independent batch.
    """
    np.random.seed(seed)
    history = np.empty((max_steps, 16), dtype=np.uint8)
    rewards = np.empty(max_steps, dtype=np.int64)
    rows = np.empty((games, 4), dtype=np.int64)
    for game in range(games):
        board = np.zeros(16, dtype=np.uint8)
        spawn(board)
        spawn(board)
        length = 0
        score = 0
        terminal = False
        while length < max_steps:
            action, after, reward, estimate = greedy(board, weights, patterns, row_moves, row_rewards)
            if action < 0:
                terminal = True
                break
            history[length] = after
            rewards[length] = reward
            score += reward
            length += 1
            board = after.copy()
            spawn(board)
        if terminal:
            target = 0.0
        else:
            _, _, _, target = greedy(board, weights, patterns, row_moves, row_rewards)
            target = max(0.0, target)
        for t in range(length - 1, -1, -1):
            update_value(history[t], target, weights, patterns, alpha, tc_sum, tc_abs, use_tc,
                         normalize_collisions)
            target = rewards[t] + max(0.0, value(history[t], weights, patterns))
        rows[game, 0] = score
        rows[game, 1] = length
        rows[game, 2] = 1 << int(board.max())
        rows[game, 3] = 0 if terminal else 1
    return rows


@njit(cache=True)
def chance_value(after, weights, patterns, row_moves, row_rewards, depth, probability, cutoff):
    """Expected future rewards from an afterstate; depth counts future actions.

    Exact 0.9/empty and 0.1/empty chance probabilities. Optional probability
    cutoff replaces a low-probability subtree with learned V(after).
    """
    if depth == 0 or probability < cutoff:
        return max(0.0, value(after, weights, patterns))
    empty = np.empty(16, dtype=np.int64)
    count = 0
    for j in range(16):
        if after[j] == 0:
            empty[count] = j
            count += 1
    if count == 0:
        return 0.0
    expected = 0.0
    board = after.copy()
    for k in range(count):
        pos = empty[k]
        for exponent in range(1, 3):
            prob = (0.9 if exponent == 1 else 0.1) / count
            board[pos] = exponent
            best = -1.0e30
            for action in range(4):
                successor, reward, changed = slide(board, action, row_moves, row_rewards)
                if changed:
                    candidate = reward + chance_value(successor, weights, patterns, row_moves,
                                                      row_rewards, depth - 1, probability * prob, cutoff)
                    best = max(best, candidate)
            expected += prob * max(0.0, best)
        board[pos] = 0
    return expected


@njit(cache=True)
def search(board, weights, patterns, row_moves, row_rewards, depth=1, cutoff=0.0001):
    best = -1.0e30
    best_action = -1
    for action in range(4):
        after, reward, changed = slide(board, action, row_moves, row_rewards)
        if changed:
            estimate = reward + chance_value(after, weights, patterns, row_moves, row_rewards,
                                             depth - 1, 1.0, cutoff)
            if estimate > best:
                best_action, best = action, estimate
    return best_action


@njit(cache=True)
def downgrade_root(board, threshold=32768):
    """Paper equation 33: halve tiles above the highest absent tile rank.

    Used only for choosing an action at the root. The real board and score
    stay unchanged. This implementation supports power-of-two thresholds.
    """
    result = board.copy()
    largest = int(board.max())
    if threshold <= 0 or (1 << largest) < threshold:
        return result
    for missing in range(largest - 1, 0, -1):
        if not np.any(board == missing):
            for i in range(16):
                if result[i] > missing:
                    result[i] -= 1
            break
    return result


@njit(cache=True, nogil=True)
def evaluate_game(weights, patterns, row_moves, row_rewards, seed, depth, cutoff, max_steps=40000, downgrade_threshold=0):
    np.random.seed(seed)
    board = np.zeros(16, dtype=np.uint8)
    spawn(board)
    spawn(board)
    score = 0
    steps = 0
    terminal = False
    while steps < max_steps:
        root = downgrade_root(board, downgrade_threshold) if downgrade_threshold else board
        action = search(root, weights, patterns, row_moves, row_rewards, depth, cutoff)
        if action < 0:
            terminal = True
            break
        board, reward, _ = slide(board, action, row_moves, row_rewards)
        score += reward
        steps += 1
        spawn(board)
    return score, steps, 1 << int(board.max()), not terminal
