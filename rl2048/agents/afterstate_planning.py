"""Frozen afterstate values with one additional exact chance/decision layer.

For each root slide, enumerate every possible spawn, then maximize the next
slide's known reward plus learned continuation. This expands two actual moves
and one intervening random spawn, unlike the root-only afterstate actor.
"""
import numpy as np

from rl2048.afterstate_compare import moves
from rl2048.afterstate_expectation import spawn_outcomes
from rl2048.agents.ntuple import encode, row_tables


class AfterstateLookahead:
    name = 'neural_afterstate_lookahead'

    def __init__(self, agent, spawn_batch=256, depth=2, nonnegative_leaf=False):
        if spawn_batch < 1 or depth not in (1,2,3):
            raise ValueError('Inference batch must be positive; exact move depth must be1,2or3')
        self.agent, self.spawn_batch, self.depth = agent, spawn_batch, depth
        if nonnegative_leaf and agent.learner.gamma < 0:
            raise ValueError('A nonnegative continuation floor requires nonnegative gamma')
        self.nonnegative_leaf = nonnegative_leaf
        self.display_name = f'Neural afterstate · {depth} exact moves with all intervening spawns'
        if nonnegative_leaf:
            self.display_name += ' · future value floored at zero'
        self.rng = np.random.default_rng(0)

    def planned_values(self, boards):
        return self._values(boards,self.depth)

    def _values(self, boards, depth):
        if depth==1:
            q = self.agent.learner.decision(boards)[0]
            if self.nonnegative_leaf:
                _, gains, legal = moves(boards, *row_tables())
                # Q = known merge points / 128 + gamma * U. Only U is
                # floored: already-known merge rewards must be preserved.
                q = np.where(legal, np.maximum(q, gains / 128), -np.inf)
            return q
        after, gains, legal = moves(boards, *row_tables())
        scores = np.full(legal.shape, -np.inf, np.float64)
        if not legal.any(): return scores
        # Illegal root slides need not leave an empty cell; never spawn on them.
        afterstates = after[legal]
        spawned, owners, probabilities = spawn_outcomes(afterstates)
        continuation = []
        for start in range(0, len(spawned), self.spawn_batch):
            q = self._values(spawned[start:start+self.spawn_batch],depth-1)
            if np.isnan(q).any() or np.isposinf(q).any():
                raise FloatingPointError('Nonfinite learned value in exact afterstate search')
            best = q.max(1)
            continuation.append(np.where(np.isfinite(q).any(1), best, 0.))
        expected = np.bincount(owners, weights=probabilities*np.concatenate(continuation),
                               minlength=len(afterstates))
        scores[legal] = gains[legal]/128 + self.agent.learner.gamma*expected
        return scores

    def act(self, board, action_mask):
        q = self.planned_values(encode(board)[None])[0]
        if not np.array_equal(np.isfinite(q), action_mask):
            raise ValueError('Game and planner legal masks disagree')
        return int(q.argmax())
