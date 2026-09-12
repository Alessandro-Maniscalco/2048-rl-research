"""Build an isolated exact-key cache prototype; never edit the live player."""
from pathlib import Path
import shutil
import subprocess

from research.afterstate_teacher import digest, write

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/research/endgame_tablebase'
BUILD = BASE / 'cached_full_rank_bridge'


def build():
    source = BUILD / 'src'
    source.mkdir(parents=True, exist_ok=False)
    for path in (BASE / 'full_rank_bridge/src').iterdir():
        shutil.copy2(path, source / path.name)
    shutil.copy2(ROOT / 'research/native/exact_wide_cache.h', source)
    node = source / 'node.h'
    original = node.read_text()
    replacements = {
        '#include "cache.h"': '#include "cache.h"\n#include "exact_wide_cache.h"',
        '  int TryAllTiles(int depth, float prob) {':
            '  int TryAllTiles(int depth, float prob) {\n    ++wide_cache.tile_calls;',
        '    unsigned long long compact_board = 0;':
            '    unsigned long long compact_board = 0;\n    uint64_t wide_low = 0, wide_high = 0;',
        '        compact_board = compact_board * 16 + board[x][y];':
            '        compact_board = compact_board * 16 + board[x][y];\n'
            '        if (y < 2) wide_low = (wide_low << 5) | board[x][y];\n'
            '        else wide_high = (wide_high << 5) | board[x][y];',
        '    float tile2_prob = prob / empty_tiles * 0.9;':
            '    if (!skip_cache && !exact_cache_key && wide_cache.enabled &&\n'
            '        wide_cache.Lookup(wide_low, wide_high, prob, depth, &score)) return score;\n\n'
            '    float tile2_prob = prob / empty_tiles * 0.9;',
        '    return total_score / empty_tiles;':
            '    if (!exact_cache_key && wide_cache.enabled)\n'
            '      wide_cache.Update(wide_low, wide_high, prob, depth, total_score / empty_tiles);\n'
            '    return total_score / empty_tiles;',
        '  static Cache cache;': '  static Cache cache;\n  static ExactWideCache wide_cache;',
        'Cache Node::cache;': 'Cache Node::cache;\nExactWideCache Node::wide_cache;',
    }
    for old, new in replacements.items():
        assert original.count(old) == 1, old
        original = original.replace(old, new)
    node.write_text(original)
    bridge = (ROOT / 'research/native/full_rank_search.cc').read_text()
    old = 'EXPORT int full_rank_query(const int* ranks, int depth, int* action, int* value) {'
    assert bridge.count(old) == 1
    bridge = bridge.replace(old,
        'EXPORT int cached_full_rank_query(const int* ranks, int depth, int enabled, int* action, int* value, unsigned long long* stats) {')
    bridge = bridge.replace('  Node::cache.Clear();',
        '  Node::cache.Clear();\n  Node::wide_cache.Clear();\n  Node::wide_cache.enabled = enabled != 0;')
    bridge = bridge.replace('  *value=node.Search(depth,&native_action);',
        '  *value=node.Search(depth,&native_action);\n'
        '  if (stats) { stats[0]=Node::wide_cache.lookups; stats[1]=Node::wide_cache.hits;\n'
        '    stats[2]=Node::wide_cache.updates; stats[3]=Node::wide_cache.tile_calls; }')
    bridge_path = source / 'cached_full_rank_search.cc'
    bridge_path.write_text(bridge)
    library = BUILD / 'libcached_full_rank_search.dylib'
    command = ['clang++', '-std=c++17', '-O3', '-fwrapv', '-fvisibility=hidden',
               '-fPIC', '-dynamiclib', '-pthread', '-I', str(source),
               str(bridge_path), str(source / 'board.cc'), '-o', str(library)]
    subprocess.run(command, check=True)
    write(BUILD / 'build.json', dict(command=command, library_sha256=digest(library),
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in
                      [Path(__file__), ROOT / 'research/native/exact_wide_cache.h', *sorted(source.iterdir())]},
        hypothesis='Exact 80-bit board, depth and probability memoization may remove '
            'redundant work above rank15 while preserving the uncached answer.',
        limits='Separate prototype; live late-game experiment and original engines unchanged. '
            '32MiB direct-mapped cache; full-key verification, per-query epochs. '
            'Probability pruning and heuristic evaluation are unchanged and remain approximate.'))
    print(library)


if __name__ == '__main__':
    build()
