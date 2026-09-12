import pytest
import torch
from research.quantile_risk_experiment import aggregate_quantiles


def test_known_distribution_integrals():
    values = torch.tensor([1., 3., 5., 7.])
    assert aggregate_quantiles(values, 'mean') == 4
    assert aggregate_quantiles(values, 'median') == 4
    assert aggregate_quantiles(values, 'lower_half') == 2
    assert aggregate_quantiles(values, 'upper_half') == 6


def test_odd_quantile_count_splits_middle_bin():
    # Three equal-probability bins. Each tail gets its outer bin plus half
    # the middle bin: lower=(2*0+3)/3, upper=(3+2*6)/3.
    values = torch.tensor([0., 3., 6.]).expand(2, 4, 3)
    torch.testing.assert_close(aggregate_quantiles(values, 'lower_half'), torch.ones(2, 4))
    torch.testing.assert_close(aggregate_quantiles(values, 'upper_half'), torch.full((2, 4), 5.))
    torch.testing.assert_close(aggregate_quantiles(values, 'median'), torch.full((2, 4), 3.))


def test_crossed_heads_keep_their_quantile_labels():
    # Sorting would silently change the policy; instead crossings are reported.
    values = torch.tensor([7., 5., 3., 1.])
    assert aggregate_quantiles(values, 'lower_half') == 6
    assert aggregate_quantiles(values, 'upper_half') == 2
    with pytest.raises(ValueError):
        aggregate_quantiles(values, 'unknown')
