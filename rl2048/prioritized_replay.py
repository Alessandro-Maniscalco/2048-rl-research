"""Proportional experience replay with global importance-weight normalization.

P(i) = (abs(mean TD error_i) + epsilon)^alpha / sum_j priority_j^alpha
w_i = (N P(i))^-beta / max_j (N P(j))^-beta.

Sum/min trees keep sampling and updates logarithmic in replay capacity. A
million-record buffer is never rescanned for a single minibatch. This follows
Schaul et al. https://arxiv.org/abs/1511.05952, adapted to mean QR TD errors.
"""
import numpy as np
from numba import njit

from rl2048.offline_data import ReplayBuffer


@njit(cache=True)
def set_masses(sums, minimums, leaves, indices, masses):
    for i in range(len(indices)):
        node = leaves + indices[i]
        sums[node] = masses[i]
        minimums[node] = masses[i]
        while node > 1:
            node //= 2
            sums[node] = sums[node * 2] + sums[node * 2 + 1]
            minimums[node] = min(minimums[node * 2], minimums[node * 2 + 1])


@njit(cache=True)
def find_prefix(sums, leaves, masses):
    indices = np.empty(len(masses), np.int64)
    for i in range(len(masses)):
        node = 1
        mass = masses[i]
        while node < leaves:
            left = node * 2
            if mass < sums[left]:
                node = left
            else:
                mass -= sums[left]
                node = left + 1
        indices[i] = node - leaves
    return indices


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, alpha=.6, beta=.4, epsilon=1e-6):
        if capacity < 1 or not 0 <= alpha <= 1 or not 0 <= beta <= 1:
            raise ValueError('Positive capacity and alpha/beta in [0,1] required')
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('Priority epsilon must be positive and finite')
        super().__init__(capacity)
        self.alpha, self.beta, self.epsilon = alpha, beta, epsilon
        self.leaves = 1 << (capacity - 1).bit_length()
        self.sums = np.zeros(2 * self.leaves, np.float64)
        self.minimums = np.full(2 * self.leaves, np.inf, np.float64)
        self.maximum_priority = 1.

    def add(self, **batch):
        indices = (np.arange(len(batch['actions'])) + self.position) % self.capacity
        super().add(**batch)
        # Newly observed records enter with the highest priority seen so far.
        masses = np.full(len(indices), self.maximum_priority ** self.alpha)
        set_masses(self.sums, self.minimums, self.leaves, indices, masses)

    def sample(self, size, rng):
        if not self.size or size < 1:
            raise ValueError('Cannot sample an empty buffer or nonpositive batch')
        total = self.sums[1]
        if not np.isfinite(total) or total <= 0:
            raise ValueError('Replay priority mass must be positive and finite')
        draws = np.minimum(rng.random(size) * total, np.nextafter(total, 0.))
        indices = find_prefix(self.sums, self.leaves, draws)
        masses = self.sums[self.leaves + indices]
        # N and the total cancel under global max-weight normalization.
        weights = (self.minimums[1] / masses) ** self.beta
        result = {key: value[indices] for key, value in self.data.items()}
        result.update(indices=indices, weights=weights.astype(np.float32),
                      sampling_probabilities=(masses / total).astype(np.float32))
        return result

    def update_priorities(self, indices, errors):
        indices = np.asarray(indices, dtype=np.int64)
        errors = np.asarray(errors, dtype=np.float64)
        if indices.ndim != 1 or indices.shape != errors.shape or not np.isfinite(errors).all():
            raise ValueError('One finite priority error per sampled index is required')
        if np.any(indices < 0) or np.any(indices >= self.size):
            raise ValueError('Priority index is outside stored records')
        priorities = np.abs(errors) + self.epsilon
        # Sampling is with replacement. Resolve duplicate updates by max error,
        # so their order cannot affect the next minibatch distribution.
        unique, inverse = np.unique(indices, return_inverse=True)
        combined = np.zeros(len(unique))
        np.maximum.at(combined, inverse, priorities)
        if len(combined):
            self.maximum_priority = max(self.maximum_priority, float(combined.max()))
        set_masses(self.sums, self.minimums, self.leaves, unique, combined ** self.alpha)
