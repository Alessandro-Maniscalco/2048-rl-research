"""Record a game whose moves are chosen by the assistant, with no coded policy.

The command shows only the current board and legal moves. It never recommends
an action, searches future states, reads a teacher, or retries a failed game.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from research.afterstate_teacher import write
from rl2048.game import ACTION_NAMES, Game2048, legal_actions
from rl2048.view import COLORS


def play(folder, seed=None, action=None, expected_step=None, note=''):
    folder = Path(folder)
    path = folder/'game.json'
    env = Game2048()
    if path.exists():
        if seed is not None:
            raise ValueError('Existing game cannot be reset or replaced')
        saved = json.loads(path.read_text())
        env.reset(seed=saved['result']['seed'])
        state = saved['state']
        env.board = np.array(state['board'], dtype=np.int64)
        env.score, env.steps = state['score'], state['steps']
        env._terminated = state['terminated']
        env.np_random.bit_generator.state = state['rng']
    else:
        if seed is None or action is not None:
            raise ValueError('Start once with a seed and without an action')
        folder.mkdir(parents=True, exist_ok=False)
        board, info = env.reset(seed=seed)
        saved = dict(algorithm='GPT-6 assistant decisions; no coded policy or search',
            started_epoch=time.time(), result=dict(seed=seed), actions=ACTION_NAMES,
            colors=COLORS, frames=[dict(board=board.tolist(), score=0, action=None, reward=0)],
            decisions=[], protocol='One preselected fresh game. The assistant selects each move from the actual current board. No teacher predictions, search recommendations, undo or restarts. Wall time includes interleaved research and is not a model inference benchmark.')
    if action is not None:
        if expected_step != env.steps:
            raise ValueError('Expected step differs; do not accidentally repeat a move')
        index = ACTION_NAMES.index(action)
        if not legal_actions(env.board)[index]:
            raise ValueError('Chosen move is illegal; board and RNG unchanged')
        saved['decisions'].append(dict(step=env.steps, action=action, note=note))
        board, reward, terminal, _, info = env.step(index)
        saved['frames'].append(dict(board=board.tolist(), score=env.score, action=index, reward=reward))
    saved['state'] = dict(board=env.board.tolist(), score=env.score, steps=env.steps,
        terminated=env._terminated, rng=env.np_random.bit_generator.state)
    saved['result'].update(score=env.score, length=env.steps, max_tile=int(env.board.max()),
        terminated=env._terminated, truncated=False, complete=env._terminated,
        elapsed_seconds=time.time()-saved['started_epoch'], known_state_fraction=None)
    write(path, saved)
    # The replay contains actual frames; its title identifies the decision source.
    template = Path('rl2048/replay.html').read_text()
    public = {k:v for k,v in saved.items() if k != 'state'}
    (folder/'replay.html').write_text(template.replace('__REPLAY_DATA__', json.dumps(public).replace('<', '\\u003c')))
    print(f'Move {env.steps}; score {env.score}; terminal {env._terminated}')
    print(env.board)
    print('Legal:', ', '.join(name for name, ok in zip(ACTION_NAMES, legal_actions(env.board)) if ok))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--action', choices=ACTION_NAMES)
    parser.add_argument('--expected-step', type=int)
    parser.add_argument('--note', default='')
    play(**vars(parser.parse_args()))
