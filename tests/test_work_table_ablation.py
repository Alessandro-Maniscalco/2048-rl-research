import json

from research.work_table_ablation import compare_pair, decision_signature


def decision(action, mode='search', nodes=7):
    return dict(action=action, mode=mode, depth=3,
        search_board_hex='0', search_scores=[1,2,3,4], seconds=1.,
        work_calls=[dict(ignored_seconds=2., layers=[dict(depth=3,nodes=nodes,action=action,scores=[1,2,3,4])])])


def write_pair(tmp_path, before_difference=False):
    folders = [tmp_path/'with', tmp_path/'without']
    frames = [dict(board=[[i]*4]*4, score=i) for i in range(4)]
    for index, folder in enumerate(folders):
        folder.mkdir()
        ds = [decision(0), decision(1, 'frozen-10' if index==0 else 'search'), decision(index)]
        if before_difference and index==1:
            ds[0] = decision(2)
        (folder/'replay.json').write_text(json.dumps(dict(frames=frames)))
        (folder/'decisions.json').write_text(json.dumps(ds))
    return folders


def test_prefix_gate_allows_only_post_intervention_action_differences(tmp_path):
    with_folder, without_folder = write_pair(tmp_path)
    r = compare_pair(with_folder, without_folder)
    assert r['identical_before_first_table_intervention']
    assert r['first_table_move'] == 2 and r['first_action_difference_move'] == 3
    ds = json.loads((without_folder/'decisions.json').read_text())
    ds[0] = decision(2)
    (without_folder/'decisions.json').write_text(json.dumps(ds))
    assert not compare_pair(with_folder, without_folder)['identical_before_first_table_intervention']


def test_repeatability_ignores_clock_time_but_checks_executed_work():
    a, b = decision(0), decision(0)
    b['seconds'] = 987.
    b['work_calls'][0]['ignored_seconds'] = 654.
    assert decision_signature(a) == decision_signature(b)
    b['work_calls'][0]['layers'][0]['nodes'] += 1
    assert decision_signature(a) != decision_signature(b)
