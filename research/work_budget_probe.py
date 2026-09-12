"""Bounded late-board cost checks before full deterministic-work experiments."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from research.afterstate_teacher import write
from research.hybrid_endgame_study import BASE
from research.work_budget_hybrid import WorkBudgetHybridAgent
from rl2048.game import legal_actions


def query(job):
    agent=WorkBudgetHybridAgent(job['work_target'])
    board=np.asarray(job['board'],dtype=np.int64)
    started=time.perf_counter()
    try:
        action=agent.act(board,legal_actions(board))
        return dict(**job,action=action,decision=agent.last_decision,
                    elapsed_seconds=time.perf_counter()-started)
    finally:agent.tables.close()


def main():
    cases=[]
    for name in ('hybrid_validation100/seed8975052', 'tactical_comparison/tactical_seed8976007'):
        folder=BASE/name
        replay=json.loads((folder/'replay.json').read_text())
        decisions=json.loads((folder/'decisions.json').read_text())
        # Recorded search boards at the difficult 65,536 transition, after it,
        # and near death. Selection is by stage and mode, not proposed score.
        first=next(i for i,f in enumerate(replay['frames']) if np.max(f['board'])>=65536)
        candidates=[i for i,d in enumerate(decisions) if d['mode']=='search']
        for target in (first-10,first-1,first,first+100,len(decisions)-5,len(decisions)-1):
            index=min(candidates,key=lambda i:abs(i-target))
            cases.append(dict(source=name,board_index=index,board=replay['frames'][index]['board']))
    jobs=[dict(**case,work_target=budget,replicate=rep)
          for case in cases for budget in (1000000,4000000) for rep in (0,1)]
    with ProcessPoolExecutor(max_workers=8) as pool:results=list(pool.map(query,jobs))
    signatures={}
    same=True
    for r in results:
        key=(r['source'],r['board_index'],r['work_target'])
        d=r['decision']
        signature={k:d.get(k) for k in ('action','mode','depth','work_calls','search_board_hex')}
        if key in signatures:same=same and signature==signatures[key]
        else:signatures[key]=signature
    summary=dict(complete=True,queries=len(results),duplicated_queries_agree=same,
        max_seconds=max(r['elapsed_seconds'] for r in results),
        p95_seconds=float(np.quantile([r['elapsed_seconds'] for r in results],.95)),
        results=results,note='Cold-controller diagnostic on recorded late boards; not full games, '
            'not original adaptive planner state, not a score comparison. Work target is soft; '
            'completed layers can overshoot. Each native process executes queries sequentially.')
    write(BASE/'work_budget_late_probe.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='results'}),flush=True)


if __name__=='__main__':main()
