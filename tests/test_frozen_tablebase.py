"""Integration checks against recorded native decisions and immutable caches."""
from pathlib import Path
import json
import re

import numpy as np
import pytest

from rl2048.agents.frozen_tablebase import CACHE,LIBRARY,ROOT,FrozenTablebase
from rl2048.game import ACTION_NAMES,legal_actions

pytestmark=pytest.mark.skipif(not LIBRARY.exists(),reason='Build the local frozen table adapter first')


@pytest.fixture(scope='module')
def tables():
    table=FrozenTablebase()
    yield table
    table.close()


def test_missing_cache_is_an_error(tmp_path):
    with pytest.raises(ValueError,match='cache directory'):
        FrozenTablebase(tmp_path)


def test_invalid_input_and_unmatched_board(tables):
    assert tables.query(np.zeros((4,4),np.int64))==dict(kind=0,action=-1,probability=0.)
    for value in (-2,3,2**18):
        board=np.zeros((4,4),np.int64);board[0,0]=value
        with pytest.raises(ValueError):tables.query(board)


def test_cached_actions_match_recorded_native_player_including_high_tiles(tables):
    folder=ROOT/'runs/research/tablebase_search/validation_depth8_seed8949281'
    game=json.loads((folder/'replay.json').read_text())
    labels=[]
    for line in (folder/'native.log').read_text().splitlines():
        m=re.search(r'#(\d+):\s+(up|left|right|down).*from lookup-(10|11), prob ([0-9.]+)',line)
        if m:labels.append((int(m[1])-1,ACTION_NAMES.index(m[2]),int(m[3]),float(m[4])))
    selected=np.unique(np.linspace(0,len(labels)-1,512,dtype=int))
    hits=0;high_tiles=0
    for index in selected:
        t,action,kind,probability=labels[index]
        board=np.array(game['frames'][t]['board']);before=board.copy()
        result=tables.query(board)
        np.testing.assert_array_equal(board,before)
        if result['kind']==0:continue  # Missing cached chunks deliberately fall back.
        hits+=1;high_tiles+=bool(board.max()>=65536)
        assert result['action']==action and result['kind']==kind
        assert abs(result['probability']-probability)<=0.000501
        assert legal_actions(board)[result['action']]
    assert hits>=450 and high_tiles>0
