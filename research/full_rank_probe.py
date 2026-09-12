"""Check high-rank representation and measure the isolated search on saved boards."""
from concurrent.futures import ProcessPoolExecutor
import json
import time

import numpy as np

from research.afterstate_teacher import write
from research.full_rank_search import FullRankSearch,encode
from research.hybrid_endgame_study import BASE
from rl2048.agents.endgame import load_native,pack_ranks
from rl2048.game import legal_actions,move


def query(job):
    player=FullRankSearch()
    board=np.asarray(job['board'],dtype=np.int64)
    started=time.perf_counter()
    answer=player.query(board,job['depth'])
    assert answer['has_move'] and legal_actions(board)[answer['action']]
    return dict(**job,answer=answer,seconds=time.perf_counter()-started)


def main():
    load_native()
    from engine_core.BoardMover import s_move_board,decode_board
    player=FullRankSearch()
    examples=[]
    for row in ([65536,32768,32768,0],[65536,65536,0,0]):
        board=np.zeros((4,4),dtype=np.int64);board[3]=row
        real,gain,valid=move(board,3)
        full,full_gain,full_valid=player.slide(board,3)
        packed=np.uint64(pack_ranks(np.minimum(encode(board),15)))
        capped_after,capped_gain=s_move_board(packed,1)
        assert np.array_equal(full,real) and full_gain==gain and full_valid==valid
        assert capped_gain==0 and gain in (65536,131072)
        examples.append(dict(board=board.tolist(),action='left',real_after=real.tolist(),
            actual_reward=gain,full_rank_reward=full_gain,
            compact_capped_after=decode_board(capped_after).tolist(),compact_capped_reward=int(capped_gain),
            synthetic=True,note='Representation unit example, not a played game or score record.'))
    cases=[]
    for name in ('hybrid_validation100/seed8975052','work_budget_comparison/budget4000000_seed8978003'):
        folder=BASE/name
        replay=json.loads((folder/'replay.json').read_text())
        decisions=json.loads((folder/'decisions.json').read_text())
        first=next(i for i,f in enumerate(replay['frames']) if np.max(f['board'])>=65536)
        candidates=[i for i,d in enumerate(decisions) if i>=first and d['mode']=='search']
        for target in (first,first+100,(first+len(decisions))//2,len(decisions)-1):
            index=min(candidates,key=lambda i:abs(i-target))
            cases.append(dict(source=name,board_index=index,board=replay['frames'][index]['board'],
                              recorded_action=decisions[index]['action']))
    jobs=[dict(**c,depth=d,replicate=rep) for c in cases for d in (5,8) for rep in (0,1)]
    with ProcessPoolExecutor(max_workers=8) as pool:results=list(pool.map(query,jobs))
    signatures={};same=True
    for r in results:
        key=(r['source'],r['board_index'],r['depth'])
        if key in signatures:same=same and signatures[key]==r['answer']
        else:signatures[key]=r['answer']
    summary=dict(complete=True,synthetic_examples=examples,queries=len(results),
        repeated_queries_match=same,max_seconds=max(r['seconds'] for r in results),
        results=results,note='Full ranks remove a representational limitation. Search still uses '
            'probability pruning and programmed heuristic values, not exact full-game Q values. '
            'Saved-board probes do not establish improved final scores. Original engines remain unchanged.')
    write(BASE/'full_rank_probe.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('results','synthetic_examples')}),flush=True)


if __name__=='__main__':main()
