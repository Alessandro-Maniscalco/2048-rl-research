"""Report cache correctness and measured query cost without score claims."""
import json
from pathlib import Path


def render():
    out = Path('runs/research/endgame_tablebase/cached_full_rank_probe')
    if not (out / 'summary.json').exists():
        return
    s = json.loads((out / 'summary.json').read_text())
    live = json.loads((out.parent / 'late_game_comparison/status.json').read_text())
    comparison_status = ('still running' if live['phase'] == 'running' else
                         'finished; see the conditional comparison for its results and completion checks')
    rows = []
    for depth, d in s['by_depth'].items():
        rows.append(f'<tr><td>{depth}</td><td>{d["sum_seconds"]["frozen"]:.3f} s</td>'
            f'<td>{d["sum_seconds"]["on"]:.3f} s</td>'
            f'<td>{d["frozen_to_cached_ratio"]:.2f}×</td>'
            f'<td>{100*d["tile_call_reduction"]:.1f}%</td></tr>')
    checks = []
    for filename, label in [('trajectory_check.json', 'Recorded smoke prefix'),
                             ('complete_continuation_check.json', 'Entire completed continuation')]:
        path = out / filename
        if path.exists():
            a = json.loads(path.read_text())
            checks.append(f'<li><a href="{filename}">{label}</a>: {a["moves"]:,} moves; '
                f'{a["full_rank_queries"]:,} search queries and {a["table_hits"]:,} table hits. '
                'Every action, search value, board, spawn and reward matched. '
                f'Cached verification took {a["prefix_seconds"]:.2f} seconds; '
                'this is not a controlled full-game timing comparison.</li>')
    (out / 'index.html').write_text(f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>Exact cache for large tiles</title>
    <style>body{{font:17px system-ui;line-height:1.55;max-width:1040px;margin:36px auto;padding:0 22px;color:#20313e}}
    h1{{line-height:1.2}}.note{{background:#fff0ca;padding:18px;border-radius:10px}}
    table{{border-collapse:collapse;width:100%;font-size:16px}}th,td{{padding:12px 8px;border-bottom:1px solid #ddd;text-align:left}}
    .scroll{{overflow-x:auto}}a{{color:#245ab5}}code{{background:#eff2f5;padding:2px 4px}}</style></head><body>
    <h1>Reusing identical search work after 65,536</h1>
    <p class="note"><strong>Speed improvement in recorded-board queries, with unchanged answers.</strong>
    All {s['comparisons']} comparisons matched the frozen search's move and heuristic value.
    This does not demonstrate a higher game score or a tenfold speedup of the whole player.</p>
    <h2>Why the old search repeats work</h2>
    <p>Different move and spawn sequences can lead to the same board. A cache remembers the search result so
    it does not have to calculate that continuation again. The older key used four bits per tile, which cannot
    encode 65,536. Our correctness fix disabled that cache for large tiles; this made late search expensive.</p>
    <p>The separate prototype uses five bits for each of the 16 cells: an exact 80-bit board key.
    A result is reused only when the board, remaining depth and accumulated probability's float bits all match.
    The cache is reset for each root query, so values from a different search configuration cannot carry over.</p>
    <p><code>cache key = (all 16 tile ranks, remaining depth, exact probability bits)</code></p>
    <p>The table uses 32 MiB. Hash collisions may discard an entry, but complete-key verification prevents
    them from returning a different board's answer. The search still uses its original heuristic and probability pruning.</p>
    <h2>Measured work</h2>
    <p>Twenty recorded test cases (18 distinct boards; two positions occur twice) × two depths × two reversed-order repetitions.
    The totals include those repeated cases. Each comparison runs the frozen library,
    the instrumented prototype with caching off, and the same prototype with caching on: 240 native queries.
    Four benchmark workers ran alongside the separate late-game comparison. Libraries were initialized before timing.</p>
    <div class="scroll"><table><thead><tr><th>Depth</th><th>Frozen total time</th><th>Cached total time</th>
    <th>Time ratio</th><th>Fewer search calls</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
    <p>Times sum 40 queries per method at each depth. At depth eight the median paired cache-off/cache-on ratio
    was {s['by_depth']['8']['median_paired_speed_ratio']:.2f}×. This measures these saved boards under concurrent CPU load.</p>
    <h2>Correctness checks</h2>
    <p>Two tests passed: exact board/depth/probability matching, cache clearing, full integer score preservation,
    and agreement with frozen search on 16 randomized boards plus three recorded positions.</p>
    <ul>{''.join(checks)}</ul>
    <p>Original policy files remained unchanged. The late-game experiment used its frozen implementation throughout;
    this prototype is for future experiments after verification.</p>
    <h2>What the completed control games show</h2>
    <p>All 12 compact-player continuations have ended naturally. Their 148,503 moves passed seeded replay checks;
    the last 20 decisions of each game were checked again against the frozen tables. Ten built another 16,384 tile;
    two reached only another 8,192. None built another 32,768 tile.</p>
    <p>Therefore the inability to merge two 32,768 tiles was not directly encountered on these actual boards.
    It remains a representation limitation, but is not yet an explanation for these losses. The full-rank arm is {comparison_status}.</p>
    <p><a href="protocol.json">Frozen probe protocol</a> · <a href="summary.json">Timing and correctness summary</a> ·
    <a href="results.json">Every query</a> · <a href="../cached_full_rank_bridge/build.json">Build and source hashes</a> ·
    <a href="../late_game_comparison/control_transition_audit.json">Control transition audit</a> ·
    <a href="../late_game_comparison/index.html">Live conditional comparison</a> · <a href="../index.html">All research</a></p>
    </body></html>''')


if __name__ == '__main__':
    render()
