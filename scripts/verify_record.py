"""Independently replay the public record with the readable reference game."""
import json
from pathlib import Path
from rl2048.game import Game2048


def main():
    record = json.loads((Path(__file__).parents[1] / 'docs/record/actions.json').read_text())
    env = Game2048()
    _, info = env.reset(seed=record['seed'])
    for action in record['actions']:
        assert info['action_mask'][action], 'Illegal recorded action'
        board, _, terminated, truncated, info = env.step(action)
        assert not truncated
    assert terminated, 'Record must end naturally'
    assert info['score'] == record['score']
    assert info['steps'] == record['length']
    assert board.tolist() == record['final_board']
    assert info['max_tile'] == record['max_tile']
    print(f"Verified {info['score']:,} points, {info['steps']:,} legal moves, "
          f"largest tile {info['max_tile']:,}; natural game over.")


if __name__ == '__main__':
    main()
