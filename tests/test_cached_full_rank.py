import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from research.cached_full_rank import CachedFullRankSearch
from research.full_rank_search import FullRankSearch


def test_cache_requires_exact_board_depth_probability_and_root_epoch(tmp_path):
    binary = tmp_path / 'cache_check'
    subprocess.run(['clang++', '-std=c++17', '-O2', '-Iresearch/native',
        'research/native/exact_wide_cache_test.cc', '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_cached_values_equal_uncached_frozen_search():
    original, prototype = FullRankSearch(), CachedFullRankSearch()
    rng = np.random.default_rng(71933)
    boards = []
    for _ in range(16):
        ranks = rng.integers(0, 12, (4, 4))
        board = np.where(ranks > 0, 1 << ranks, 0)
        board[0, 0] = 65536
        boards.append((board, 3))
    protocol = json.loads(Path('runs/research/endgame_tablebase/late_game_comparison/protocol.json').read_text())
    boards.extend((np.array(c['board']), 5) for c in protocol['cases'])
    hits = 0
    for board, depth in boards:
        before = board.copy()
        expected = original.query(board, depth)
        uncached = prototype.query(board, depth, enabled=False)
        cached = prototype.query(board, depth, enabled=True)
        assert uncached['answer'] == cached['answer'] == expected
        assert cached['stats']['tile_calls'] <= uncached['stats']['tile_calls']
        hits += cached['stats']['hits']
        np.testing.assert_array_equal(before, board)
    assert hits > 0
    with pytest.raises(ValueError):
        prototype.query(board, depth=11)
