"""Teacher action supervision, keeping equally good legal moves equivalent."""
import numpy as np
import torch


def optimal_actions(q, legal):
    """Accept all teacher-best actions, using the existing ranking tolerance."""
    q, legal = np.asarray(q), np.asarray(legal, dtype=bool)
    if q.shape != legal.shape or not legal.any(1).all() or not np.isfinite(q[legal]).all():
        raise ValueError('Need finite teacher Q on at least one legal action per board')
    best = np.where(legal, q, -np.inf).max(1, keepdims=True)
    return legal & np.isclose(q, best, rtol=1e-6, atol=1e-5)


def imitation_loss(logits, legal, best):
    """-log probability assigned to ANY teacher-best move (hard set target).

    With one best action this is ordinary cross entropy. For a true tie, do
    not force an arbitrary direction or a uniform distribution over the tie.
    Targets have no value regression, discount, reward shaping or TD update.
    """
    if logits.shape != legal.shape or best.shape != legal.shape:
        raise ValueError('Logits and action masks must have identical shapes')
    if torch.any(best & ~legal) or not torch.all(best.any(1)):
        raise ValueError('Each teacher target needs a nonempty subset of legal actions')
    logp = torch.log_softmax(logits.masked_fill(~legal, -torch.inf), 1)
    return -torch.logsumexp(logp.masked_fill(~best, -torch.inf), 1).mean()


def teacher_gap_penalty(logits, legal, best, teacher_q, scale_points=16384.):
    """Expected teacher-estimated mistake cost, in units of a fixed point scale.

    Saved teacher Q is in points/128. Keep numerical teacher ties equivalent,
    ignore illegal actions, and detach the teacher. This is an optional addition
    to hard-action cross entropy, not a measured own-policy RL return.
    """
    if not np.isfinite(scale_points) or scale_points <= 0:
        raise ValueError('Teacher gap scale must be a positive finite constant')
    if not (logits.shape == legal.shape == best.shape == teacher_q.shape):
        raise ValueError('Teacher values, logits and masks must have equal shapes')
    if torch.any(best & ~legal) or not torch.all(best.any(1)):
        raise ValueError('Each teacher target needs a nonempty legal action set')
    if not torch.isfinite(teacher_q[legal]).all():
        raise ValueError('Legal teacher Q values must be finite')
    q = teacher_q.detach()
    maximum = q.masked_fill(~legal, -torch.inf).amax(1, keepdim=True)
    costs = (maximum-q).masked_fill(~legal | best, 0.) * (128./scale_points)
    probabilities = torch.softmax(logits.masked_fill(~legal, -torch.inf), 1)
    return (probabilities*costs).sum(1).mean()


def reference_policy_kl(logits, reference_logits, legal):
    """KL(frozen starting policy || student), averaged over the sampled boards."""
    if logits.shape != reference_logits.shape or logits.shape != legal.shape:
        raise ValueError('Reference policy, student and masks must have equal shapes')
    if not legal.any(1).all():
        raise ValueError('Reference KL requires at least one legal action per board')
    if not torch.isfinite(logits[legal]).all() or not torch.isfinite(reference_logits[legal]).all():
        raise ValueError('Legal policy logits must be finite')
    old_logp = torch.log_softmax(reference_logits.detach().masked_fill(~legal, -torch.inf), 1)
    new_logp = torch.log_softmax(logits.masked_fill(~legal, -torch.inf), 1)
    difference = old_logp.masked_fill(~legal, 0.)-new_logp.masked_fill(~legal, 0.)
    return (old_logp.exp()*difference).sum(1).mean()
