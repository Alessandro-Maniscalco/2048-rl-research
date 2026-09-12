"""Exact probability that the next random tile ends a standard 2048 game."""
import numpy as np

from rl2048.afterstate_compare import moves
from rl2048.agents.ntuple import row_tables
from rl2048.vector_game import legal_masks


def immediate_death_risks(boards):
    """Return B×4 risks and legal masks; illegal actions have infinite risk.

    A spawn can end the game only when the afterstate has exactly one empty
    cell. With two or more empties, at least one remains after the spawn, so
    some slide remains legal. Only the single-empty cases need enumeration.
    Boards contain tile exponents, not raw tile values.
    """
    boards = np.asarray(boards, dtype=np.uint8)
    if boards.ndim != 2 or boards.shape[1] != 16:
        raise ValueError('Expected a batch of16-cell exponent boards')
    after, _, legal = moves(boards, *row_tables())
    risks = np.where(legal, 0., np.inf)
    owners, actions = np.where(legal & ((after == 0).sum(2) == 1))
    if len(owners):
        selected = after[owners, actions]
        empty = (selected == 0).argmax(1)
        for rank, probability in [(1, .9), (2, .1)]:
            spawned = selected.copy()
            spawned[np.arange(len(spawned)), empty] = rank
            terminal = ~legal_masks(spawned, *row_tables()).any(1)
            risks[owners, actions] += probability*terminal
    return risks, legal


def safest_policy_scores(logits, boards):
    """Keep the network's preferences only among moves with minimum risk.

    This is an explicit inference intervention. Minimizing immediate death
    need not maximize eventual score; evaluate the complete-game tradeoff.
    Terminal rows return all -inf, as required by the batch evaluator.
    """
    risks, legal = immediate_death_risks(boards)
    if np.shape(logits) != legal.shape:
        raise ValueError('Expected four policy logits per board')
    allowed = legal & (risks == risks.min(1, keepdims=True))
    return np.where(allowed, logits, -np.inf), risks, legal


def minimum_risk_mask(boards, legal=None):
    """Restrict action probabilities using exact risk, without changing game legality."""
    risks, actual_legal = immediate_death_risks(boards)
    if legal is not None and not np.array_equal(legal, actual_legal):
        raise ValueError('Supplied legal mask differs from game rules')
    return actual_legal & (risks == risks.min(1, keepdims=True))
