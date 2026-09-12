"""Bounded CPU pilot of the vendored search/tablebase player, with replay checks."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np

from research.afterstate_teacher import write, digest
from rl2048.game import ACTION_NAMES, legal_actions, move
from rl2048.view import COLORS


def parse_replay(log, seed):
    boards, actions, rows = [], [], []
    for line in log.splitlines():
        cells = line.split('|')
        if len(cells) == 6:
            rows.append([int(re.search(r'\d+', c).group()) if re.search(r'\d+', c) else 0
                         for c in cells[1:5]])
        score = re.match(r'^score (\d+) max (\d+) sum (\d+)', line)
        if score:
            if len(rows) != 4:
                raise ValueError('Native board rows missing')
            boards.append(dict(board=rows, score=int(score[1])))
            rows = []
        # Table construction prints a backspace spinner before some move lines.
        action = re.search(r'#(\d+):\s+(up|left|right|down)\b', line)
        if action:
            if int(action[1]) != len(actions)+1:
                raise ValueError('Native move sequence is incomplete')
            actions.append(ACTION_NAMES.index(action[2]))
    if len(boards) != len(actions)+1 or not actions:
        raise ValueError('Need a complete sequence of boards and actions')
    initial = np.array(boards[0]['board'])
    if (boards[0]['score'] != 0 or np.count_nonzero(initial) != 2
            or not np.isin(initial[initial != 0], [2,4]).all()):
        raise ValueError('Standard evaluation requires exactly two initial2/4 tiles')
    frames = [boards[0] | dict(action=None, reward=0)]
    total = 0
    for i, action in enumerate(actions):
        after, reward, changed = move(np.array(boards[i]['board']), action)
        actual = np.array(boards[i+1]['board'])
        spawn = actual != after
        if not changed or spawn.sum() != 1 or after[spawn][0] != 0 or actual[spawn][0] not in (2,4):
            raise ValueError('Native move/spawn differs from our game rules')
        total += reward
        if total != boards[i+1]['score']:
            raise ValueError('Native score differs from actual accumulated merge rewards')
        frames.append(boards[i+1] | dict(action=action, reward=reward))
    terminal = not legal_actions(np.array(boards[-1]['board'])).any()
    result = dict(seed=seed, score=total, length=len(actions), max_tile=int(np.max(boards[-1]['board'])),
        terminated=bool(terminal), truncated=False, complete=bool(terminal), known_state_fraction=None,
        note='Native libc RNG, different stream from Python/NumPy at the same numeric seed. Every recorded move, spawn and score was checked against the Python game rules.')
    return result, frames


def native_command(vendor, depth, seed, min_probability=None, lookup_probability=None):
    command = [str(Path(vendor)/'2048'), '-d', str(depth), '-i', '1', '-v']
    if min_probability is not None:
        if not np.isfinite(min_probability) or not 0 < min_probability <= 1:
            raise ValueError('Probability cutoff must be finite and in (0,1]')
        # -d resets the cutoff, so its explicit override must come afterwards.
        command += ['-p', str(min_probability)]
    if lookup_probability is not None:
        if not np.isfinite(lookup_probability) or not 0 <= lookup_probability < 1:
            raise ValueError('Lookup threshold must be finite and in [0,1)')
        command += ['-t', str(lookup_probability)]
    return command+[str(seed)]


def run(out, depth=3, seed=8949000, seconds=1200, rss_gib=8., vendor_path=None, min_probability=None, lookup_probability=None):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path.cwd()
    vendor = Path(vendor_path).resolve() if vendor_path else root/'third_party/2048-ai'
    cache = out.parent/'cache'
    cache.mkdir(exist_ok=True)
    stop = root/'runs/research/scaled_transformer/STOP'
    command = native_command(vendor, depth, seed, min_probability, lookup_probability)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=vendor, text=True).strip()
    protocol = dict(command=command, source_directory=str(vendor), source_revision=revision, binary_sha256=digest(vendor/'2048'),
        cache=str(cache), existing_tables=[p.name for p in cache.glob('tuple_moves.*')],
        depth=depth, min_probability_override=min_probability, lookup_probability_override=lookup_probability,
        seconds_budget=seconds, rss_gib_budget=rss_gib,
        hypothesis='Combine short-horizon search with solved smaller-board subproblems to improve high-tile survival. First verify standard game rules and measure the small-table implementation on this Mac before comparing depths or teacher use.',
        provenance='External GPL-3.0 program macroxue/2048-ai with documented Apple Silicon memory-mapping fixes; not a neural model trained locally or GPT choosing the moves.')
    write(out/'protocol.json', protocol)
    (out/'portability.patch').write_text(subprocess.check_output(['git', 'diff'], cwd=vendor, text=True))
    started, peak, reason = time.monotonic(), 0, None
    with (out/'native.log').open('w') as log:
        child = subprocess.Popen(command, cwd=cache, stdout=log, stderr=subprocess.STDOUT)
        while child.poll() is None:
            elapsed = time.monotonic()-started
            reading = subprocess.run(['ps', '-o', 'rss=', '-p', str(child.pid)], capture_output=True, text=True)
            if reading.stdout.strip():
                peak = max(peak, int(reading.stdout.strip())*1024)
            if stop.exists(): reason = 'user_stop'
            elif elapsed > seconds: reason = 'time_budget'
            elif peak > rss_gib*1024**3: reason = 'memory_budget'
            write(out/'status.json', dict(pid=child.pid, elapsed_seconds=elapsed,
                peak_sampled_rss_gib=peak/1024**3, stop_reason=reason))
            if reason:
                child.terminate()
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
                break
            time.sleep(2)
    result = dict(complete=False, returncode=child.returncode, stop_reason=reason,
        elapsed_seconds=time.monotonic()-started, peak_sampled_rss_gib=peak/1024**3)
    if child.returncode == 0:
        episode, frames = parse_replay((out/'native.log').read_text(), seed)
        episode['elapsed_seconds'] = result['elapsed_seconds']
        result.update(episode)
        data = dict(algorithm='External search and solved-pattern tables', result=episode,
            frames=frames, colors=COLORS, actions=ACTION_NAMES)
        write(out/'replay.json', data)
        template = (root/'rl2048/replay.html').read_text()
        (out/'replay.html').write_text(template.replace('__REPLAY_DATA__', json.dumps(data).replace('<', '\\u003c')))
    write(out/'result.json', result)
    write(out/'status.json', dict(phase='complete' if result['complete'] else 'stopped', pid=None,
        elapsed_seconds=result['elapsed_seconds'], peak_sampled_rss_gib=peak/1024**3,
        stop_reason=reason))
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--depth', type=int, default=3)
    p.add_argument('--seed', type=int, default=8949000)
    p.add_argument('--seconds', type=int, default=1200)
    p.add_argument('--rss-gib', type=float, default=8.)
    run(**vars(p.parse_args()))
