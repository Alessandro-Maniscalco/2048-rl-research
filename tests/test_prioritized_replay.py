import numpy as np
import pytest
import torch

from rl2048.prioritized_replay import PrioritizedReplayBuffer
from rl2048.n_step import NStepReplay
from rl2048.agents.qr_dqn import quantile_huber_loss


def transitions(count, terminal=False):
    return dict(states=np.zeros((count,16),np.uint8), next_states=np.ones((count,16),np.uint8),
        actions=np.arange(count,dtype=np.int64)%4, rewards=np.ones(count,np.float32),
        masks=np.ones((count,4),bool), next_masks=np.full((count,4),not terminal),
        terminated=np.full(count,terminal), truncated=np.zeros(count,bool))


class FixedDraws:
    def __init__(self, values): self.values=np.array(values)
    def random(self, size):
        assert size==len(self.values)
        return self.values.copy()


def test_known_probabilities_global_importance_weights_and_boundaries():
    replay=PrioritizedReplayBuffer(5,alpha=1,beta=1,epsilon=1)
    replay.add(**transitions(3))
    replay.update_priorities([0,1,2],[0,1,6])  # masses 1,2,7; total 10
    batch=replay.sample(5,FixedDraws([0,.1,.299,.3,.999999]))
    np.testing.assert_array_equal(batch['indices'],[0,1,1,2,2])
    np.testing.assert_allclose(batch['sampling_probabilities'],[.1,.2,.2,.7,.7])
    np.testing.assert_allclose(batch['weights'],[1,.5,.5,1/7,1/7])
    # Full IS correction recovers the uniform objective up to the same global
    # scale N*P_min, even when a minibatch contains no least-likely record.
    probabilities=np.array([.1,.2,.7]);weights=.1/probabilities;values=np.array([2.,4.,8.])
    assert np.dot(probabilities,weights*values)==pytest.approx(.3*values.mean())


def test_ring_overwrite_and_duplicate_priorities():
    replay=PrioritizedReplayBuffer(3,alpha=1,beta=.4,epsilon=1)
    replay.add(**transitions(3))
    replay.update_priorities([0,0,1,2],[2,8,0,3])
    np.testing.assert_allclose(replay.sums[replay.leaves:replay.leaves+3],[9,1,4])
    assert replay.sums[1]==14 and replay.minimums[1]==1
    replay.add(**transitions(2)) # indices 0 and 1 get historical maximum 9
    np.testing.assert_allclose(replay.sums[replay.leaves:replay.leaves+3],[9,9,4])
    assert replay.sums[1]==22 and replay.minimums[1]==4
    sampled=replay.sample(100,np.random.default_rng(1))
    assert sampled['indices'].max()<3
    assert np.isfinite(sampled['weights']).all()


def test_alpha_zero_is_uniform_even_with_extreme_errors():
    replay=PrioritizedReplayBuffer(7,alpha=0,beta=1)
    replay.add(**transitions(3));replay.update_priorities([0,1,2],[0,1,1e6])
    batch=replay.sample(3,FixedDraws([0,.34,.99]))
    np.testing.assert_array_equal(batch['indices'],[0,1,2])
    np.testing.assert_allclose(batch['sampling_probabilities'],np.full(3,1/3))
    np.testing.assert_array_equal(batch['weights'],np.ones(3))


def test_nstep_terminal_flush_preserves_discount_and_priorities():
    replay=NStepReplay(10,1,n=3,gamma=.5,priority_alpha=.6)
    replay.add(**transitions(1));replay.add(**transitions(1,terminal=True))
    assert replay.size==2
    np.testing.assert_allclose(replay.replay.data['rewards'][:2],[1.5,1.])
    np.testing.assert_allclose(replay.replay.data['discounts'][:2],[.25,.5])
    assert replay.replay.data['terminated'][:2].all()
    batch=replay.sample(2,FixedDraws([0,.9]))
    replay.update_priorities(batch['indices'],[1.,4.])
    assert replay.replay.sums[1]>0


def test_per_record_quantile_loss_and_weighted_gradient_by_hand():
    prediction=torch.tensor([[0.,2.],[1.,1.]],requires_grad=True)
    target=torch.tensor([[1.,3.],[1.,1.]],requires_grad=True)
    losses=quantile_huber_loss(prediction,target,reduction='none')
    torch.testing.assert_close(losses,torch.tensor([.3125,0.]))
    weighted=(losses*torch.tensor([.5,1.])).mean()
    assert weighted.item()==pytest.approx(.078125)
    weighted.backward()
    torch.testing.assert_close(prediction.grad[0],torch.tensor([-.03125,-.03125]))
    assert target.grad is None


def test_reject_invalid_indices_and_nonfinite_priorities():
    replay=PrioritizedReplayBuffer(4)
    with pytest.raises(ValueError):replay.sample(1,np.random.default_rng(0))
    replay.add(**transitions(1))
    with pytest.raises(ValueError):replay.update_priorities([1],[1.])
    with pytest.raises(ValueError):replay.update_priorities([0],[np.nan])
