"""Enumerate the known random tile event for expected afterstate TD targets."""
import numpy as np
from numba import njit
import torch


@njit(cache=True)
def spawn_outcomes(afterstates):
    """Every empty cell is uniform; tile2 has .9 mass and tile4 has .1 mass.

    Boards contain exponents, so spawned entries are1 or2. Unlike a game step,
    this function does not slide, earn points, sample an RNG or reset a game.
    """
    states = np.empty((len(afterstates)*32,16),np.uint8)
    owners = np.empty(len(states),np.int64)
    probabilities = np.empty(len(states),np.float32)
    used = 0
    for i in range(len(afterstates)):
        empty = np.flatnonzero(afterstates[i]==0)
        if not len(empty):
            raise ValueError('A valid slide afterstate must leave an empty cell')
        for cell in empty:
            for rank, mass in ((1,.9),(2,.1)):
                states[used] = afterstates[i]
                states[used,cell] = rank
                owners[used] = i
                probabilities[used] = mass/len(empty)
                used += 1
    return states[:used],owners[:used],probabilities[:used]


def weighted_spawn_values(values, owners, probabilities, count):
    """Sum probabilities per original afterstate; never average branches equally."""
    result = torch.zeros(count, device=values.device, dtype=values.dtype)
    return result.index_add(0, owners, probabilities*values)
