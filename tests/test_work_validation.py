from research.work_validation import summarize


def result(seed, score):
    return dict(seed=seed, score=score, length=score//2, elapsed_seconds=2., max_tile=32768,
        complete=True, terminated=True, truncated=False, source_and_binary_unchanged=True,
        exact_seeded_replay_audited=True, saved_audit_checked=True)


def test_incomplete_or_unverified_evidence_never_gets_a_mean():
    good = [result(1, 20), result(2, 40)]
    variants = [good[:1], [good[0], good[0]],
                [good[0], dict(good[1], saved_audit_checked=False)],
                [good[0], dict(good[1], truncated=True)],
                [good[0], dict(good[1], source_and_binary_unchanged=False)]]
    for rows in variants:
        summary = summarize(rows, [1, 2], 3., True)
        assert not summary['complete'] and summary['mean_score'] is None
    assert summarize(good, [1, 2], 3.)['mean_score'] is None


def test_complete_aggregate_matches_hand_computed_scores_and_counts():
    summary = summarize([result(1, 20), result(2, 40)], [1, 2], 3., True)
    assert summary['complete'] and summary['mean_score'] == 30
    assert summary['mean_length'] == 15 and summary['total_transitions'] == 30
    assert summary['mean_score_bootstrap95'] == [20, 40]
    assert summary['tile_reaching_rates']['32768'] == 1
    assert summary['tile_reaching_rates']['65536'] == 0
