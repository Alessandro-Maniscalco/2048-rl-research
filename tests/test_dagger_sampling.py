import numpy as np
import pytest

from research.dagger_sampling import StageSampler


def test_zero_fraction_preserves_exact_uniform_samples_and_rng():
    states = np.zeros((17, 16), np.uint8)
    a, b = np.random.default_rng(9), np.random.default_rng(9)
    sampler = StageSampler(states)
    for _ in range(3):
        np.testing.assert_array_equal(sampler.sample(a, 128), b.integers(17, size=128))
    assert a.bit_generator.state == b.bit_generator.state


def test_stage_boundaries_and_mixture_probabilities():
    ranks = np.array([8]*70+[9, 10]*10+[11, 12]*4+[13, 15], np.uint8)
    sampler = StageSampler(np.repeat(ranks[:, None], 16, axis=1), .5)
    assert sampler.counts == [70, 20, 8, 2]
    ids = sampler.sample(np.random.default_rng(51), 200000)
    # Half uniform + half balanced: [.475, .225, .165, .135].
    actual = np.bincount(np.searchsorted([8, 10, 12], ranks[ids]), minlength=4)/len(ids)
    np.testing.assert_allclose(actual, [.475, .225, .165, .135], atol=.004)
    # Within each stage, every board remains equally likely.
    counts = np.bincount(ids, minlength=len(ranks))
    assert abs(counts[-1]-counts[-2])/len(ids) < .004


def test_empty_stages_and_invalid_inputs():
    states = np.array([[8], [8], [13]], np.uint8)
    sampler = StageSampler(states, 1.)
    assert sampler.counts == [2, 0, 0, 1]
    ids = sampler.sample(np.random.default_rng(19), 100000)
    np.testing.assert_allclose(np.bincount(ids)/len(ids), [.25, .25, .5], atol=.006)
    for invalid in [-.1, 1.1, np.nan, np.inf]:
        with pytest.raises(ValueError, match='fraction'):
            StageSampler(states, invalid)
    with pytest.raises(ValueError, match='empty'):
        StageSampler(states[:0])
