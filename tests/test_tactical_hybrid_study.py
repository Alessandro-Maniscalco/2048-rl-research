from research.tactical_hybrid_study import summarize


def test_only_full_audited_pairs_produce_a_score_comparison():
    seeds = [101, 102]
    results = [dict(seed=s, tactical=t, complete=True, exact_seeded_replay_audited=True,
        source_and_binary_unchanged=True, score=100 + 20*t, elapsed_seconds=2 + t,
        max_tile=32, length=10, tactical_checks=int(t), tactical_overrides=int(t),
        tactical_budget_exhaustions=0) for s in seeds for t in (False, True)]
    assert summarize(results[:-1], seeds)['paired_difference'] is None
    assert summarize(results+[results[-1]], seeds)['paired_difference'] is None
    broken = [dict(r) for r in results]
    broken[-1]['exact_seeded_replay_audited'] = False
    assert not summarize(broken, seeds)['complete']
    broken[-1]['exact_seeded_replay_audited'] = True
    broken[-1]['source_and_binary_unchanged'] = False
    assert not summarize(broken, seeds)['complete']
    summary = summarize(results, seeds)
    assert summary['complete'] and summary['paired_difference']['mean'] == 20
    assert summary['paired_difference']['bootstrap95'] == [20, 20]
    assert summary['arms']['control']['mean_score'] == 100
    assert summary['arms']['tactical']['mean_score'] == 120
