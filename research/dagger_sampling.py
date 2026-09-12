"""A controlled sampling change: expose the learner to every game stage.

No board features or teacher labels change. With fraction f, a board has
probability (1-f)/N + f/(K*N_stage), where K counts nonempty stages.
"""
import numpy as np


STAGE_UPPER_RANKS = (8, 10, 12)  # <=256, 512--1024, 2048--4096, >=8192.


class StageSampler:
    def __init__(self, states, fraction=0.):
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError('Stage sampling fraction must be finite and between zero and one')
        if len(states) == 0:
            raise ValueError('Cannot sample an empty fitting dataset')
        self.size, self.fraction = len(states), fraction
        self.groups, self.counts = [], None
        if fraction:
            ranks = np.asarray(states).reshape(self.size, -1).max(axis=1)
            stage = np.searchsorted(STAGE_UPPER_RANKS, ranks, side='left')
            groups = [np.flatnonzero(stage == i) for i in range(4)]
            self.counts = [len(g) for g in groups]
            self.groups = [g for g in groups if len(g)]

    def sample(self, rng, batch):
        # Preserve the legacy sampler's exact RNG path when fraction is zero.
        ids = rng.integers(self.size, size=batch)
        if self.fraction:
            balanced = np.flatnonzero(rng.random(batch) < self.fraction)
            stages = rng.integers(len(self.groups), size=len(balanced))
            for i, group in enumerate(self.groups):
                slots = balanced[stages == i]
                ids[slots] = group[rng.integers(len(group), size=len(slots))]
        return ids
