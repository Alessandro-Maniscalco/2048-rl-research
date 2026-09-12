"""Exact move-and-spawn lookahead, with a frozen Q-network at the leaves.

This changes action selection only; it performs no learning. For each action:
score(s,a) = E_spawn[r_learning + gamma * max_legal Q(next_state, action)].
The spawn expectation enumerates every empty cell, with probabilities .9/.1.
"""
import numpy as np
import torch
from numba import njit
import json
from pathlib import Path

from rl2048.agents.neural import tensor_boards
from rl2048.agents.ntuple import encode, row_tables
from rl2048.fast2048 import slide
from rl2048.rewards import learning_rewards
from rl2048.vector_game import legal_masks


@njit(cache=True)
def outcomes(board, rows, row_rewards):
    # At most four moves * sixteen empty cells * two possible spawned tiles.
    states = np.empty((128, 16), np.uint8)
    actions = np.empty(128, np.int64)
    probabilities = np.empty(128, np.float64)
    rewards = np.empty(128, np.float32)
    count = 0
    for action in range(4):
        after, reward, moved = slide(board, action, rows, row_rewards)
        if not moved:
            continue
        empty = np.flatnonzero(after == 0)
        for cell in empty:
            for exponent, probability in ((1, .9), (2, .1)):
                states[count] = after
                states[count, cell] = exponent
                actions[count] = action
                probabilities[count] = probability / len(empty)
                rewards[count] = reward
                count += 1
    return states[:count], actions[:count], probabilities[:count], rewards[:count]


@njit(cache=True)
def batch_outcomes(boards, rows, row_rewards):
    states = np.empty((len(boards)*128,16),np.uint8)
    owners = np.empty(len(states),np.int64)
    actions = np.empty(len(states),np.int64)
    probabilities = np.empty(len(states),np.float64)
    rewards = np.empty(len(states),np.float32)
    count = 0
    for owner, board in enumerate(boards):
        next_states, moves, probs, raw = outcomes(board,rows,row_rewards)
        end = count + len(next_states)
        states[count:end] = next_states
        owners[count:end] = owner
        actions[count:end] = moves
        probabilities[count:end] = probs
        rewards[count:end] = raw
        count = end
    return states[:count],owners[:count],actions[:count],probabilities[:count],rewards[:count]


class PlanningQAgent:
    name = 'q_network_search'

    def __init__(self, agent, gamma=.99, reward_mode='score', shaping_scale=2., depth=1):
        if agent.algorithm not in ('dqn', 'double_dqn', 'dueling_dqn', 'dueling_double_dqn', 'cql', 'fitted_lookahead_q', 'qr_dqn'):
            raise ValueError('The leaf network must predict Q-values, not policy logits.')
        self.agent = agent
        if depth not in (1,2):
            raise ValueError('Supported exact search depths are 1 and 2.')
        self.depth = depth
        self.gamma, self.reward_mode, self.shaping_scale = gamma, reward_mode, shaping_scale
        self.algorithm = agent.algorithm + f'_search_depth{depth}'
        self.display_name = f'Q-network + exact {depth}-move spawn expectation'
        self.rng = np.random.default_rng(0)

    @torch.no_grad()
    def planned_values(self, boards, depth):
        states, owners, actions, probabilities, raw = batch_outcomes(boards, *row_tables())
        masks_here = legal_masks(boards, *row_tables())
        scores = np.zeros((len(boards),4))
        if not len(states):
            return np.full_like(scores,-np.inf)
        masks = legal_masks(states, *row_tables())
        terminated = ~masks.any(axis=1)
        if depth == 1:
            q = self.agent.policy(tensor_boards(states, self.agent.device)).cpu().numpy()
        else:
            q = self.planned_values(states, depth-1)
        continuation = np.where(terminated, 0., np.where(masks, q, -np.inf).max(axis=1))
        rewards = learning_rewards(raw, boards[owners], states,
                                   terminated, self.gamma, self.reward_mode, self.shaping_scale)
        targets = rewards + self.gamma * continuation
        np.add.at(scores, (owners, actions), probabilities*targets)
        return np.where(masks_here, scores, -np.inf)

    def action_values(self, board):
        return self.planned_values(encode(board)[None], self.depth)[0]

    def act(self, board, action_mask):
        if not np.any(action_mask):
            raise ValueError('No legal action.')
        return int(np.where(action_mask, self.action_values(board), -np.inf).argmax())

    def save(self, path, metadata=None):
        path = Path(path)
        path.mkdir(parents=True,exist_ok=True)
        self.agent.save(path/'network')
        (path/'metadata.json').write_text(json.dumps({
            'agent':self.name,'gamma':self.gamma,'reward_mode':self.reward_mode,
            'shaping_scale':self.shaping_scale,'depth':self.depth,
            'experiment':metadata or {}},indent=2))

    @classmethod
    def load(cls, path, device='cpu'):
        from rl2048.agents.neural import NeuralAgent
        path = Path(path)
        meta = json.loads((path/'metadata.json').read_text())
        return cls(NeuralAgent.load(path/'network',device),meta['gamma'],
                   meta['reward_mode'],meta['shaping_scale'],meta['depth'])
