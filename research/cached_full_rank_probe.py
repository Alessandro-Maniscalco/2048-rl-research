"""Matched saved-board timing and answer checks, not a game-score experiment."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from research.afterstate_teacher import digest, write
from research.cached_full_rank import CachedFullRankSearch, LIBRARY
from research.full_rank_search import FullRankSearch
from research.late_game_study import fingerprints as live_fingerprints

BASE = Path('runs/research/endgame_tablebase')
OUT = BASE / 'cached_full_rank_probe'


def query(job):
    original, prototype = FullRankSearch(), CachedFullRankSearch()
    board = np.array(job['board'], dtype=np.int64)
    # Initialize the libraries before timing; clear search caches on every call.
    original.query(board, 1)
    prototype.query(board, 1)
    order = ['frozen', 'off', 'on'] if job['repeat'] == 0 else ['on', 'off', 'frozen']
    records = {}
    for mode in order:
        if mode == 'frozen':
            started = time.perf_counter()
            answer = original.query(board, job['depth'])
            records[mode] = dict(answer=answer, seconds=time.perf_counter()-started)
        else:
            records[mode] = prototype.query(board, job['depth'], enabled=mode == 'on')
    return dict(**job, records=records,
        answers_match=records['frozen']['answer'] == records['off']['answer'] == records['on']['answer'])


def main():
    OUT.mkdir(exist_ok=False)
    cases = []
    existing = json.loads((BASE / 'full_rank_probe.json').read_text())
    for r in existing['results']:
        if r['depth'] == 5 and r['replicate'] == 0:
            cases.append({k: r[k] for k in ('source', 'board_index', 'board')})
    protocol = json.loads((BASE / 'late_game_comparison/protocol.json').read_text())
    for case in protocol['cases']:
        cases.append(dict(source=case['source'], board_index=case['source_move'], board=case['board']))
    for case_id, seed in enumerate((8981000, 8981004, 8981008)):
        folder = BASE / f'late_game_comparison/case{case_id}_seed{seed}_compact'
        replay = json.loads((folder / 'replay.json').read_text())
        decisions = json.loads((folder / 'decisions.json').read_text())
        candidates = [i for i, d in enumerate(decisions) if d['mode'] == 'search']
        for target in (256, len(decisions)//2, len(decisions)-1):
            index = min(candidates, key=lambda i: abs(i-target))
            cases.append(dict(source=str(folder.relative_to(BASE)), board_index=index,
                board=replay['frames'][index]['board']))
    jobs = [dict(**case, case_id=i, depth=depth, repeat=rep)
            for i, case in enumerate(cases) for depth in (5, 8) for rep in (0, 1)]
    fingerprint = live_fingerprints()
    prototype_files = [Path(__file__), Path('research/cached_full_rank.py'), LIBRARY,
                       *sorted((LIBRARY.parent / 'src').iterdir())]
    prototype_hashes = {str(p): digest(p) for p in prototype_files}
    write(OUT / 'protocol.json', dict(jobs=jobs, workers=4, live_fingerprints=fingerprint,
        prototype_fingerprints=prototype_hashes,
        hypothesis='Memoizing exact repeated subproblems may accelerate high-rank search '
            'without changing its values or actions.',
        design='20 recorded boards x depths5/8 x2 reversed-order repetitions. '
            'Frozen library versus instrumented cache-off versus cache-on in each job. '
            'Libraries initialized before timing and cache cleared for each query. '
            'Four CPU workers alongside the separate12-worker late-game experiment.',
        limits='Saved-query timing only, not full-game speed or score improvement. '
            'All answers must agree; any mismatch blocks use. No live engine files changed.'))
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for r in pool.map(query, jobs):
            results.append(r)
            write(OUT / 'status.json', dict(finished=len(results), planned=len(jobs)))
    summary = dict(complete=True, recorded_boards=len(cases), comparisons=len(results),
        native_queries=3*len(results), all_answers_match=all(r['answers_match'] for r in results),
        live_source_unchanged=live_fingerprints() == fingerprint,
        prototype_unchanged=all(digest(p) == sha for p, sha in prototype_hashes.items()),
        elapsed_wall_seconds=time.perf_counter()-started, by_depth={},
        note='Paired saved-board query benchmark under concurrent CPU load. '
            'Exact memoization within this heuristic probability-pruned search; '
            'not exact full-game Q, a game-score claim, or an end-to-end speed multiplier.')
    for depth in (5, 8):
        rows = [r for r in results if r['depth'] == depth]
        seconds = {m: sum(r['records'][m]['seconds'] for r in rows) for m in ('frozen','off','on')}
        calls = {m: sum(r['records'][m]['stats']['tile_calls'] for r in rows) for m in ('off','on')}
        summary['by_depth'][str(depth)] = dict(sum_seconds=seconds,
            frozen_to_cached_ratio=seconds['frozen']/seconds['on'],
            off_to_cached_ratio=seconds['off']/seconds['on'],
            median_paired_speed_ratio=float(np.median([r['records']['off']['seconds']/r['records']['on']['seconds'] for r in rows])),
            tile_calls=calls, tile_call_reduction=1-calls['on']/calls['off'],
            max_cached_seconds=max(r['records']['on']['seconds'] for r in rows),
            hits=sum(r['records']['on']['stats']['hits'] for r in rows))
    write(OUT / 'results.json', results)
    write(OUT / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
