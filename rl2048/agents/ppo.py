"""PPO's two mathematical operations, separated for hand-checking.

Schulman et al. (2017), https://arxiv.org/abs/1707.06347, equations 7, 11.
The training loop is in ppo_train.py; these functions do not run environments.
"""
import numpy as np
import torch


def advantages(rewards, values, next_values, terminated, truncated, gamma=.99, lam=.95):
    """[time, game] arrays. Bootstrap timeouts, but never cross episode resets."""
    result = np.zeros_like(rewards, dtype=np.float32)
    carry = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (~terminated[t]) - values[t]
        carry = delta + gamma * lam * (~(terminated[t] | truncated[t])) * carry
        result[t] = carry
    return result, result + values


def clipped_policy_loss(new_logp, old_logp, advantage, clip=.2):
    ratio = (new_logp - old_logp).exp()
    plain = ratio * advantage
    clipped = ratio.clamp(1 - clip, 1 + clip) * advantage
    return -torch.minimum(plain, clipped).mean()
