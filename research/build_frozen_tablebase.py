"""Build an isolated, read-only view of existing MacroXue strategy tables.

Do not modify either established native engine or regenerate its cache files.
The generated header disables table computation and saving, and uses const
mapped entries directly instead of triggering upstream copy-on-write reads.
"""
import difflib
from pathlib import Path
import shutil
import subprocess

from research.afterstate_teacher import digest, write

ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/'runs/research/endgame_tablebase/frozen_bridge'
VENDOR=ROOT/'third_party/2048-ai-cache-fix'


def build():
    source=BUILD/'src';source.mkdir(parents=True,exist_ok=True)
    for name in ('board.h','board.cc','array.h','tuple_move.h','LICENSE'):
        shutil.copy2(VENDOR/name,source/name)
    original=(VENDOR/'tuple.h').read_text()
    revised=original.replace(
        'Tuple(double save_threshold) : tuple_moves(0xff), save_threshold(save_threshold) {\n    tuple_moves.Load(kTupleFile);',
        'Tuple(double save_threshold, const char* file = nullptr) : tuple_moves(0xff), save_threshold(save_threshold) {\n    tuple_moves.Load(file ? file : kTupleFile);')
    old='''  ~Tuple() {
    auto is_valid = [](const TupleMove& t) { return t.ValidProb() && t.Prob() > 0; };
    tuple_moves.Save(kTupleFile, is_valid, save_threshold);
  }'''
    assert old in revised
    revised=revised.replace(old,'  ~Tuple() = default;  // Frozen cache: never save or generate entries.')
    old='''          {
            // Don't use tuple_move pointer after Compute as it may be invalidated.
            auto tuple_move = tuple_moves.locate(v);
            if (!tuple_move || !tuple_move->ValidProb()) Compute();
          }
          *move = tuple_moves[v].move;'''
    new='''          const auto* entry = tuple_moves.locate(v);
          if (!entry || !entry->ValidProb()) {
            if (transposed) Transpose();
            return 0;  // Cache misses fall back to the other engine.
          }
          *move = entry->move;'''
    assert old in revised
    revised=revised.replace(old,new)
    assert 'return tuple_moves[v].Prob();' in revised
    # Only the public query return changes; recursive computation is unreachable.
    tail=revised.index('    float SuggestMove(int* move)')
    revised=revised[:tail]+revised[tail:].replace('return tuple_moves[v].Prob();','return entry->Prob();',1)
    assert 'const char* file = nullptr' in revised
    (source/'tuple.h').write_text(revised)
    (BUILD/'frozen.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),revised.splitlines(True),fromfile='upstream/tuple.h',tofile='frozen/tuple.h')))
    bridge=ROOT/'research/native/tablebase_bridge.cc'
    library=BUILD/'libfrozen_tables.dylib'
    command=['clang++','-std=c++17','-O3','-fPIC','-dynamiclib','-pthread','-I',str(source),str(bridge),str(source/'board.cc'),'-o',str(library)]
    subprocess.run(command,check=True)
    write(BUILD/'build.json',dict(command=command,library_sha256=digest(library),
        upstream_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=VENDOR,text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [bridge,*sorted(source.glob('*'))]},
        reason='Frozen lookup-only adapter for a controlled hybrid: no table computation, no saved-file writes, full exact tile ranks through17.'))
    print(library)


if __name__=='__main__':build()
