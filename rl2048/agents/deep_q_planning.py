"""1–5 move/spawn lookahead with a frozen Q-network at the leaves.

Identical boards at each depth share a network/search computation. This is
exact state merging, not beam search. With probability_cutoff=0, all chance
branches are expanded to the requested depth. An optional nonzero cutoff uses
the Q estimate earlier on unlikely branches, without dropping their probability
mass. A resource limit aborts the decision
explicitly; it never silently substitutes a shallower answer.
"""
import numpy as np
import torch
import json
from pathlib import Path
from numba import njit, types
from numba.typed import Dict

from .q_planning import PlanningQAgent, batch_outcomes
from .neural import tensor_boards
from .ntuple import row_tables
from rl2048.rewards import learning_rewards


class SearchBudgetExceeded(RuntimeError):
    pass


@njit(cache=True)
def fast_legal_masks(boards):
    """A tile can move toward an adjacent empty cell or merge with an equal tile.

Checking those neighbors avoids constructing four successor boards merely to
ask whether moves exist. Action order is up/right/down/left.
"""
    result = np.zeros((len(boards), 4), np.bool_)
    for i in range(len(boards)):
        for cell in range(16):
            tile = boards[i, cell]
            if tile == 0: continue
            row, col = cell // 4, cell % 4
            if row > 0 and (boards[i, cell-4] == 0 or boards[i, cell-4] == tile): result[i, 0] = True
            if col < 3 and (boards[i, cell+1] == 0 or boards[i, cell+1] == tile): result[i, 1] = True
            if row < 3 and (boards[i, cell+4] == 0 or boards[i, cell+4] == tile): result[i, 2] = True
            if col > 0 and (boards[i, cell-1] == 0 or boards[i, cell-1] == tile): result[i, 3] = True
    return result


BOARD_KEY = types.UniTuple(types.uint64, 2)


@njit(cache=True)
def unique_boards_hash(states):
    """Exact 128-bit board keys; linear-time alternative to sorting all edges."""
    packed = states.view(np.uint64).reshape(-1, 2)
    seen = Dict.empty(key_type=BOARD_KEY, value_type=types.int64)
    first = np.empty(len(states), np.int64)
    inverse = np.empty(len(states), np.int64)
    count = 0
    for i in range(len(states)):
        key = (packed[i, 0], packed[i, 1])
        index = seen.get(key, -1)
        if index == -1:
            index = count
            seen[key] = index
            first[count] = i
            count += 1
        inverse[i] = index
    return first[:count], inverse


class DeepQPlanner(PlanningQAgent):
    name = 'deep_q_network_search'
    def __init__(self, agent, gamma=.99, reward_mode='score', shaping_scale=2.,
                 depth=3, max_edges=2_000_000, network_batch=8192, dedup='sort', probability_cutoff=0.):
        if not 1 <= depth <= 5:
            raise ValueError('Depth must be between 1 and 5.')
        super().__init__(agent, gamma, reward_mode, shaping_scale, depth=1)
        self.depth = depth
        self.max_edges = max_edges
        self.network_batch = network_batch
        if dedup not in ('sort', 'hash'): raise ValueError('Unknown exact deduplication method.')
        self.dedup = dedup
        if not 0 <= probability_cutoff <= 1: raise ValueError('Probability cutoff must be in [0, 1].')
        self.probability_cutoff = probability_cutoff
        self.algorithm = agent.algorithm + f'_exact_depth{depth}'
        self.display_name = f'Q-network + exact {depth}-move lookahead'
        self.last_stats = {}
        if probability_cutoff:
            self.algorithm = agent.algorithm + f'_pruned_depth{depth}'
            self.display_name = f'Q-network + up to {depth} moves, probability cutoff {probability_cutoff}'

    def _leaf_q(self, boards):
        pieces = []
        for start in range(0, len(boards), self.network_batch):
            pieces.append(self.agent.policy(tensor_boards(boards[start:start+self.network_batch], self.agent.device)).cpu().numpy())
        return np.concatenate(pieces) if pieces else np.empty((0, 4))

    @torch.no_grad()
    def planned_values(self, boards, depth=None):
        depth = self.depth if depth is None else depth
        if not 1 <= depth <= 5:
            raise ValueError('Depth must be between 1 and 5.')
        current = np.ascontiguousarray(boards, dtype=np.uint8).reshape(-1, 16)
        layers = []
        used_edges = 0
        counts = [len(current)]
        path_probability = np.ones(len(current))
        fallback_count = 0
        for _ in range(depth):
            active = np.flatnonzero(path_probability >= self.probability_cutoff)
            if not len(active): break
            fallback = np.zeros((len(current), 4))
            skipped = np.flatnonzero(path_probability < self.probability_cutoff)
            if len(skipped):
                fallback[skipped] = self._leaf_q(current[skipped])
                fallback_count += len(skipped)
            chunks = []
            for start in range(0, len(active), 128):
                selected = active[start:start+128]
                states, owners, actions, probabilities, raw = batch_outcomes(current[selected], *row_tables())
                used_edges += len(states)
                if used_edges > self.max_edges:
                    raise SearchBudgetExceeded(f'Search at depth {depth} exceeded {self.max_edges:,} expanded outcomes.')
                chunks.append((states, selected[owners], actions, probabilities, raw))
            masks_here = fast_legal_masks(current)
            if not chunks:
                break
            states, owners, actions, probabilities, raw = [np.concatenate([c[i] for c in chunks]) for i in range(5)]
            # Treat each 16-byte board as one sortable key; merge only exact equals.
            if self.dedup == 'hash':
                first, inverse = unique_boards_hash(np.ascontiguousarray(states))
            else:
                keys = np.ascontiguousarray(states).view('V16').ravel()
                _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
            next_states = states[first]
            next_masks = fast_legal_masks(next_states)
            terminated = ~next_masks.any(1)
            reward = learning_rewards(raw, current[owners], states, terminated[inverse],
                                      self.gamma, self.reward_mode, self.shaping_scale)
            layers.append((masks_here, owners, actions, probabilities, reward, inverse, terminated, fallback))
            next_probability = np.ones(len(next_states))
            if self.probability_cutoff:
                next_probability.fill(0.)
                # Merging never prunes a node that any incoming path would
                # have expanded on its own.
                np.maximum.at(next_probability, inverse, path_probability[owners]*probabilities)
            path_probability = next_probability
            current = next_states
            counts.append(len(current))
        q = self._leaf_q(current)
        masks = fast_legal_masks(current)
        continuation = np.where(masks.any(1), np.where(masks, q, -np.inf).max(1), 0.)
        for masks, owners, actions, probabilities, reward, inverse, terminated, fallback in reversed(layers):
            targets = reward + self.gamma * np.where(terminated[inverse], 0., continuation[inverse])
            q = fallback.copy()
            np.add.at(q, (owners, actions), probabilities * targets)
            q = np.where(masks, q, -np.inf)
            continuation = np.where(masks.any(1), q.max(1), 0.)
        self.last_stats = dict(depth=depth, unique_boards_by_depth=counts,
                               expanded_outcomes=used_edges, leaf_network_boards=len(current),
                               fallback_network_boards=fallback_count, probability_cutoff=self.probability_cutoff)
        return q

    def save(self, path, metadata=None):
        super().save(path, metadata)
        file = Path(path) / 'metadata.json'
        saved = json.loads(file.read_text())
        saved.update(max_edges=self.max_edges, network_batch=self.network_batch, dedup=self.dedup,
                     probability_cutoff=self.probability_cutoff)
        file.write_text(json.dumps(saved, indent=2))

    @classmethod
    def load(cls, path, device='cpu'):
        from .neural import NeuralAgent
        path = Path(path)
        meta = json.loads((path / 'metadata.json').read_text())
        return cls(NeuralAgent.load(path / 'network', device), meta['gamma'], meta['reward_mode'],
                   meta['shaping_scale'], meta['depth'], meta['max_edges'], meta['network_batch'], meta.get('dedup','sort'), meta.get('probability_cutoff',0.))
