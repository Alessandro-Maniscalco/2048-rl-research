"""Single-factor reward experiments; reporting always uses raw merge scores."""
import numpy as np


def snake_potential(boards,corner='bottom_left'):
    # Largest positional weight starts at bottom-left, snakes along each row.
    priority=np.array([[1,2,3,4],[8,7,6,5],[9,10,11,12],[16,15,14,13]],np.float32)
    if corner=='top_right':
        priority=np.rot90(priority,2)
    return (boards.reshape(-1,4,4)*priority).sum((1,2))/136


def corner_potential(boards):
    exponents=boards.reshape(-1,16).astype(np.float32)
    largest=exponents.max(axis=1)
    return np.where(exponents[:,12]==largest,largest/16,0.)


def positional_potential(boards,mode):
    if mode=='corner':
        return corner_potential(boards)
    if mode=='corner_snake':
        return corner_potential(boards)+snake_potential(boards,'bottom_left')
    return snake_potential(boards,'top_right' if mode=='top_right' else 'bottom_left')


def learning_rewards(raw,states,next_states,terminated,gamma,mode='score',scale=10.):
    base=raw.astype(np.float32)/128
    if mode in ('relative_score','log_relative_score'):
        # Storage uses ranks, not actual tile values. Use the pre-move maximum
        # so a newly created maximum does not change the normalization midway.
        max_rank=states.reshape(-1,16).max(axis=1).astype(np.float32)
        largest=np.exp2(max_rank) # empty-board fallback is 1, avoiding 0/0
        relative=raw.astype(np.float32)/largest
        # log1p preserves zero and scale invariance. Logging raw points before
        # division would assign different rewards to scaled copies of a merge.
        return (np.log1p(relative) if mode=='log_relative_score' else relative).astype(np.float32)
    if mode=='log_score':
        # 0 remains 0 and a 128-point merge remains 1. This changes the
        # objective by compressing big rewards; it is not merely normalization.
        return (np.log2(1+raw.astype(np.float32))/np.log2(129)).astype(np.float32)
    if mode=='constant':
        return np.ones_like(base) # explicitly optimizes survival, not score
    if mode=='dense_corner':
        return base+scale*corner_potential(next_states)
    if mode in ('bottom_left','top_right','corner','snake','corner_snake'):
        current=positional_potential(states,mode)
        future=positional_potential(next_states,mode)*(~terminated)
        # Potential shaping telescopes; terminal potential must be zero.
        return (base+scale*(gamma*future-current)).astype(np.float32)
    return base
