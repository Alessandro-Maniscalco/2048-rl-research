"""Build n-step replay records without crossing a game's reset boundary.

Each environment has its own queue. Truncations flush the queue but retain a
bootstrap from the final pre-reset state; natural termination zeroes it in DQN.
This is the ordinary uncorrected off-policy n-step DQN target, not Retrace.
"""
from collections import deque
import numpy as np
from rl2048.offline_data import ReplayBuffer


class NStepReplay:
    def __init__(self, capacity, envs, n=3, gamma=.99, priority_alpha=None,
                 priority_beta=.4, priority_epsilon=1e-6):
        if n < 1 or envs < 1 or not 0 <= gamma <= 1:
            raise ValueError('Positive n/envs and gamma in [0,1] required.')
        self.n, self.gamma = n, gamma
        self.queues = [deque() for _ in range(envs)]
        if priority_alpha is None:
            self.replay = ReplayBuffer(capacity)
        else:
            from rl2048.prioritized_replay import PrioritizedReplayBuffer
            self.replay = PrioritizedReplayBuffer(capacity, priority_alpha, priority_beta, priority_epsilon)
        self.replay.data['discounts'] = np.empty(capacity, np.float32)

    @property
    def size(self):
        return self.replay.size

    def _pop(self, queue):
        first, last = queue[0], queue[-1]
        item = {k: first[k] for k in ('states', 'actions', 'masks')}
        item.update({k: last[k] for k in ('next_states', 'next_masks', 'terminated', 'truncated')})
        item['rewards'] = sum(self.gamma**j * t['rewards'] for j, t in enumerate(queue))
        item['discounts'] = self.gamma**len(queue)
        queue.popleft()
        return item

    def _store(self, items):
        if items:
            batch = {k: np.asarray([t[k] for t in items], dtype=v.dtype)
                     for k, v in self.replay.data.items()}
            # A terminal batch can emit n records per environment.
            for start in range(0, len(items), self.replay.capacity):
                self.replay.add(**{k: v[start:start+self.replay.capacity] for k, v in batch.items()})

    def add(self, **batch):
        if len(batch['actions']) != len(self.queues):
            raise ValueError('One transition per environment is required.')
        items = []
        for i, queue in enumerate(self.queues):
            queue.append({k: np.array(v[i], copy=True) for k, v in batch.items()})
            if batch['terminated'][i] or batch['truncated'][i]:
                while queue:
                    items.append(self._pop(queue))
            elif len(queue) == self.n:
                items.append(self._pop(queue))
        self._store(items)

    def flush(self):
        """Preserve shorter pending returns at a collection-budget boundary."""
        items = []
        for queue in self.queues:
            while queue:
                items.append(self._pop(queue))
        self._store(items)

    def sample(self, size, rng):
        return self.replay.sample(size, rng)

    def update_priorities(self, indices, errors):
        self.replay.update_priorities(indices, errors)
