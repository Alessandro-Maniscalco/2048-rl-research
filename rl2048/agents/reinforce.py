"""Plain episodic REINFORCE: actual reward-to-go times action log probability.

The plain control has no critic, baseline, TD target, advantage normalization,
or replay. A separately named experiment uses the optional leave-one-out helper
below. Both freeze the collecting policy for a whole batch of complete games,
accumulate all gradients, and only then perform one optimizer step.
"""
import numpy as np
import torch
from torch import nn

from rl2048.agents.neural import masked_log_probs, chosen
from rl2048.agents.transformer_q import TransformerQNetwork
from rl2048.agents.teacher_policy_transformer import TeacherPolicyTransformer


class PolicyTransformer(TeacherPolicyTransformer):
    """The same sixteen tokens and eight shared views, with only four logits."""
    def __init__(self, width=128, input_encoding='embedding', depth=4, heads=4):
        super().__init__(width, input_encoding, depth, heads)
        self.architecture = 'transformer_policy'
        self.outputs = 4
        self.readout = nn.Linear(16 * width, 4)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        del self.value_offset, self.value_scale

    def forward(self, boards):
        views = boards.reshape(-1, 16)[:, self.symmetry_permutations].reshape(-1, 16)
        output = TransformerQNetwork.forward(self, views).reshape(-1, 8, 4)
        return output.gather(2, self.action_permutations[None].expand(len(output), -1, -1)).mean(1)


def copy_actor(model, source):
    """Transfer the complete actor, dropping the source's scalar critic row."""
    state = {k: v for k, v in source.state_dict().items()
             if k not in ('value_offset', 'value_scale')}
    state['readout.weight'] = state['readout.weight'][:4]
    state['readout.bias'] = state['readout.bias'][:4]
    model.load_state_dict(state, strict=True)


def reward_to_go(rewards, gamma=1.):
    """For ONE naturally completed episode, G[t] = r[t] + gamma * G[t+1]."""
    if not 0 <= gamma <= 1:
        raise ValueError('gamma must be between zero and one')
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 1 or not np.isfinite(rewards).all():
        raise ValueError('Expected one finite complete-episode reward sequence')
    result = np.zeros_like(rewards)
    carry = 0.
    for t in reversed(range(len(rewards))):
        carry = rewards[t] + gamma * carry
        result[t] = carry
    return result.astype(np.float32)


def leave_one_out_time_baseline(returns, lengths):
    """Mean return at move t in the OTHER independent completed games.

    A finished other game contributes zero future reward. This baseline can
    depend on move index t, but excludes every return from the current game.
    It therefore supplies no information about that game's sampled actions.
    This is a distinct REINFORCE baseline experiment, not the plain control.
    """
    returns = np.asarray(returns, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.int64)
    if lengths.ndim != 1 or len(lengths) < 2 or np.any(lengths < 1):
        raise ValueError('At least two complete nonempty episodes required')
    if returns.ndim != 1 or len(returns) != lengths.sum() or not np.isfinite(returns).all():
        raise ValueError('Episode lengths must partition finite flattened returns')
    matrix = np.zeros((len(lengths), lengths.max()), dtype=np.float64)
    offset = 0
    for i, length in enumerate(lengths):
        matrix[i, :length] = returns[offset:offset+length]
        offset += length
    totals = matrix.sum(0)
    return np.concatenate([(totals[:length]-matrix[i, :length])/(len(lengths)-1)
                           for i, length in enumerate(lengths)]).astype(np.float32)


def action_log_probs(logits, masks, temperature=1.):
    """One policy definition shared by collection and gradient evaluation."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('A positive finite action temperature is required')
    return masked_log_probs(logits / temperature, masks)


def policy_loss(logits, masks, actions, returns, episodes, temperature=1.):
    """A chunk's contribution to -sum_games sum_steps log(pi) * G / B.

    Divide by the FULL number of games, not this chunk's board count. Summing
    chunk gradients therefore equals differentiating the full episodic loss.
    The separately named baseline experiment passes G-minus-baseline here.
    """
    if episodes < 1:
        raise ValueError('A positive complete-game batch size is required')
    logp = chosen(action_log_probs(logits, masks, temperature), actions)
    return -(logp * returns.detach()).sum() / episodes
