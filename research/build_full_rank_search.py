"""Build a separate full-rank search library, preserving all frozen engines."""
from pathlib import Path
import shutil
import subprocess

from research.afterstate_teacher import digest, write

ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/'runs/research/endgame_tablebase/full_rank_bridge'
VENDOR=ROOT/'third_party/2048-ai-cache-fix'


def build():
    source=BUILD/'src';source.mkdir(parents=True,exist_ok=False)
    for name in ('board.h','board.cc','node.h','cache.h','LICENSE'):
        shutil.copy2(VENDOR/name,source/name)
    node=source/'node.h'
    original=node.read_text()
    # The initializer enumerates unreachable ranks too. Make its shifts
    # unsigned and use explicit wrap semantics for signed score-table sums.
    old='static int TileScore(int r) { return r << r; }'
    assert old in original
    node.write_text(original.replace(old,
        'static int TileScore(int r) { return static_cast<int>(static_cast<unsigned int>(r) << r); }'))
    bridge=ROOT/'research/native/full_rank_search.cc'
    library=BUILD/'libfull_rank_search.dylib'
    command=['clang++','-std=c++17','-O3','-fwrapv','-fvisibility=hidden','-fPIC',
             '-dynamiclib','-pthread','-I',str(source),str(bridge),str(source/'board.cc'),'-o',str(library)]
    subprocess.run(command,check=True)
    write(BUILD/'build.json',dict(command=command,library_sha256=digest(library),
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [bridge,*sorted(source.glob('*'))]},
        reason='Late-stage search preserves full ranks and can model32768+32768 and65536+65536. '
            'No table generation, original source mutation or game RNG access. Cache bypass '
            'for rank>=16 is retained; orientation-adaptive built-in evaluation is enabled.',
        limitation='Heuristic probability-pruned expectimax, not exact full-game score maximization. '
            'Input ranks0..17, search depth1..10, sequential calls per library handle.'))
    print(library)


if __name__=='__main__':build()
