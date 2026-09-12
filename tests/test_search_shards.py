import hashlib
import json
import time

import pytest

from research.aggregate_search_shards import aggregate


def fixture_shards(root):
    checkpoint=root/'frozen/network';checkpoint.mkdir(parents=True)
    (checkpoint/'policy.pt').write_bytes(b'fixed model')
    digest=hashlib.sha256(b'fixed model').hexdigest()
    protocol=dict(checkpoint=str(checkpoint.parent),depth=2,probability_cutoff=0.,
        started_epoch=time.time()-1,games=4,seed_start=1000,
        shards=[dict(directory='a',seed_start=1000,games=3),dict(directory='b',seed_start=1003,games=1)])
    (root/'protocol.json').write_text(json.dumps(protocol))
    for shard in protocol['shards']:
        directory=root/shard['directory'];directory.mkdir()
        config=dict(checkpoint_sha256=digest,seed_start=shard['seed_start'],
            games=shard['games'],depths=[2],probability_cutoff=0.)
        (directory/'config.json').write_text(json.dumps(config))
    return protocol


def episode(seed,score):
    return dict(seed=seed,score=score,complete=True,truncated=False,seconds=2.)


def test_combines_games_not_unweighted_shard_means_and_never_ranks_partial(tmp_path):
    fixture_shards(tmp_path)
    (tmp_path/'a/depth2.json').write_text(json.dumps({'episodes':[episode(i,0) for i in range(1000,1003)]}))
    partial=aggregate(tmp_path)
    assert not partial['complete'] and partial['mean_score'] is None
    assert partial['completed_games']==3
    (tmp_path/'b/depth2.json').write_text(json.dumps({'episodes':[episode(1003,400)]}))
    complete=aggregate(tmp_path)
    assert complete['complete'] and complete['mean_score']==100 # not 200, the mean of shard means
    assert complete['worker_game_seconds']==8
    assert complete['wall_seconds']>=0


def test_rejects_overlapping_games_and_checkpoint_mismatch(tmp_path):
    fixture_shards(tmp_path)
    path=tmp_path/'a/depth2.json';path.write_text(json.dumps({'episodes':[episode(1000,10),episode(1000,20)]}))
    with pytest.raises(ValueError,match='overlap'):aggregate(tmp_path)
    path.write_text(json.dumps({'episodes':[episode(1000,10)]}))
    config_path=tmp_path/'a/config.json';config=json.loads(config_path.read_text());config['checkpoint_sha256']='changed'
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='does not match'):aggregate(tmp_path)
