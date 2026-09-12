"""Turn previously validated native games into correctly aligned action labels.

Use frame[t].board -> frame[t+1].action. Entire games are held out. Native RNG
seeds are provenance, not equivalent Python environment seeds. No final return,
table probability or imagined Q-value is used as a supervised target.
"""
import json
from pathlib import Path
import time

import numpy as np
from numba import njit

from research.afterstate_teacher import digest, write
from rl2048.fast2048 import slide
from rl2048.agents.ntuple import row_tables
from rl2048.vector_game import legal_masks


@njit(cache=True)
def check_transitions(boards, actions, rewards, scores, rows, row_rewards):
    for t in range(len(actions)):
        after, gain, changed = slide(boards[t], actions[t], rows, row_rewards)
        if not changed or gain != rewards[t] or scores[t+1]-scores[t] != gain:
            return False
        changes = 0
        for cell in range(16):
            if after[cell] != boards[t+1, cell]:
                changes += 1
                if after[cell] != 0 or boards[t+1, cell] not in (1, 2):
                    return False
        if changes != 1:
            return False
    return True


def extract_game(path):
    game = json.loads(Path(path).read_text())
    if game['actions'] != ['up', 'right', 'down', 'left']:
        raise ValueError('Unexpected replay action order')
    result, frames = game['result'], game['frames']
    if not result['terminated'] or result['truncated']:
        raise ValueError('Need a complete naturally terminated expert game')
    raw = np.asarray([f['board'] for f in frames], dtype=np.int64).reshape(-1, 16)
    if np.any(raw < 0) or np.any((raw != 0) & ((raw & (raw-1)) != 0)):
        raise ValueError('Invalid tile value')
    ranks = np.zeros(raw.shape, dtype=np.uint8)
    ranks[raw > 0] = np.log2(raw[raw > 0]).astype(np.uint8)
    if ranks.max() > 17:
        raise ValueError('Dataset exceeds the declared 18-category representation')
    actions = np.asarray([f['action'] for f in frames[1:]], dtype=np.uint8)
    rewards = np.asarray([f['reward'] for f in frames[1:]], dtype=np.int64)
    scores = np.asarray([f['score'] for f in frames], dtype=np.int64)
    if scores[0] != 0 or np.count_nonzero(ranks[0]) != 2 or ranks[0].max() > 2:
        raise ValueError('Expert must start from a standard two-tile board')
    if np.any(actions > 3) or not check_transitions(ranks, actions, rewards, scores, *row_tables()):
        raise ValueError('Misaligned/invalid actions, spawns or score increments')
    legal = legal_masks(ranks, *row_tables())
    if legal[-1].any() or not legal[np.arange(len(actions)), actions].all():
        raise ValueError('Terminal state or expert legality mismatch')
    if result['score'] != scores[-1] or result['length'] != len(actions):
        raise ValueError('Expert result does not match its trajectory')
    return ranks[:-1], actions, legal[:-1], result


def build(protocol):
    out = Path(protocol['dataset'])
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    blocks, episodes = [], []
    for index, seed in enumerate(protocol['expert_seeds']):
        if Path(protocol['stop_file']).exists():
            write(out/'status.json', dict(complete=False, reason='stop_file'))
            return
        folder = Path(protocol['native_root'])/f'record_depth5_seed{seed}'
        path = folder/'replay.json'
        states, actions, masks, result = extract_game(path)
        heldout = index % 5 == 4
        blocks.append((states, actions, masks, np.full(len(actions), seed, np.int32),
                       np.full(len(actions), heldout, bool)))
        provenance = json.loads((folder/'protocol.json').read_text())
        episodes.append(dict(seed=seed, validation=heldout, score=result['score'],
            moves=len(actions), max_tile=result['max_tile'], replay=str(path.resolve()),
            replay_sha256=digest(path), native_command=provenance['command'],
            native_binary_sha256=provenance['binary_sha256'],
            source_revision=provenance['source_revision']))
        write(out/'status.json', dict(complete=False, processed_games=len(episodes),
            planned_games=len(protocol['expert_seeds']), moves=sum(e['moves'] for e in episodes)))
    arrays = {key:np.concatenate([b[i] for b in blocks]) for i,key in
              enumerate(('states', 'actions', 'legal', 'games', 'validation'))}
    if not arrays['validation'].any() or arrays['validation'].all():
        raise ValueError('Need fitting and held-out games')
    np.savez_compressed(out/'data.npz', **arrays)
    manifest = dict(complete=True, episodes=episodes, data_sha256=digest(out/'data.npz'),
        split='Prespecified first80record500games, every fifth WHOLE game held out:64fit/16validation.',
        labelled_states=len(arrays['states']), fitting_states=int((~arrays['validation']).sum()),
        validation_states=int(arrays['validation'].sum()),
        rank16_or_higher_states=int((arrays['states'].max(1)>=16).sum()),
        source_game_mean_score=float(np.mean([e['score'] for e in episodes])),
        label='Recorded expert action on the CURRENT board, our up/right/down/left order.',
        validation_restriction='No validation-game boards used by optimizer; validation labels measure imitation only.',
        provenance='Previously completed cache-safe depth5search/tablebase games, now repurposed as training data. '
            'The500game batch is no longer independent evidence for students trained from it. Native RNG differs from Python RNG.',
        all_transitions_rechecked=True, build_seconds=time.perf_counter()-started)
    write(out/'manifest.json', manifest)
    write(out/'status.json', dict(complete=True, processed_games=len(episodes)))
    return manifest


if __name__ == '__main__':
    protocol=json.loads(Path('runs/research/scaled_transformer/native_policy_protocol.json').read_text())
    print(json.dumps(build(protocol)), flush=True)
