"""Teach identical MLPs the Q targets computed by different search depths.

The teacher is frozen. For depth d, labels are T^d Q_old: d exact move/spawn
Bellman backups. Students only see the current board and predict four numbers.
This is offline fitted value learning from a search teacher, not a claim that
the student simulates d moves internally. Data and initialization are matched.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from rl2048.agents.deep_q_planning import DeepQPlanner
from rl2048.agents.neural import NeuralAgent
from rl2048.agents.ntuple import encode, row_tables
from rl2048.afterstate_compare import evaluate_batch
from rl2048.game import Game2048
from rl2048.vector_game import legal_masks


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    parent = NeuralAgent.load(args.student_checkpoint)
    teacher = NeuralAgent.load('runs/research/dqn16_planning_selected/network')
    config = json.loads(Path('runs/research/dqn16_planning_selected/metadata.json').read_text())
    planner = DeepQPlanner(teacher, config['gamma'], config['reward_mode'], config['shaping_scale'], depth=2)
    data_dir = args.data_dir or args.out
    datafile = data_dir / 'boards.npz'
    if not datafile.exists():
        all_boards, all_games = [], []
        # Entire games belong either to fitting or label validation, never both.
        for game in range(16):
            env = Game2048()
            board, info = env.reset(seed=8610000 + game)
            candidates = []
            term = False
            while not term:
                b = encode(board).reshape(16)
                candidates.append(b.copy())
                action = planner.planned_values(b[None])[0].argmax()
                board, _, term, _, info = env.step(int(action))
            indices = rng.choice(len(candidates), size=min(args.boards_per_game, len(candidates)), replace=False)
            all_boards.extend(np.asarray(candidates)[indices])
            all_games.extend([game] * len(indices))
            print('COLLECT', game, info['score'], len(indices), flush=True)
        np.savez_compressed(datafile, boards=np.asarray(all_boards, np.uint8), games=np.asarray(all_games))
    data = np.load(datafile)
    boards, games = data['boards'], data['games']
    masks = legal_masks(boards, *row_tables())
    results = {}
    # Preserve the existing network as an additional no-search control.
    for depth in args.depths:
        labels_file = data_dir / f'labels_depth{depth}.npy'
        if not labels_file.exists():
            planner.depth = depth
            labels = []
            start = time.time()
            for i, board in enumerate(boards):
                labels.append(planner.planned_values(board[None])[0])
                if (i+1) % 128 == 0:
                    print('LABEL', depth, i+1, len(boards), time.time()-start, flush=True)
            np.save(labels_file, np.asarray(labels, np.float32))
        labels = np.load(labels_file)
        train_ids = np.flatnonzero(games < 12)
        val_ids = np.flatnonzero(games >= 12)
        x = torch.tensor(boards, dtype=torch.long, device='mps')
        mask = torch.tensor(masks, device='mps')
        target = torch.tensor(np.where(masks, labels, 0), dtype=torch.float32, device='mps')
        model = copy.deepcopy(parent.policy).to('mps')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Reset the shuffle RNG for each depth: identical batches and update budget.
        train_rng = np.random.default_rng(42)
        history = []
        start = time.time()
        for update in range(args.updates):
            ids = torch.tensor(train_rng.choice(train_ids, size=256), device='mps')
            pred = model(x[ids])
            error = torch.where(mask[ids], pred-target[ids], 0.)
            # Common value offset and action gaps both matter. Emphasize gaps,
            # since greedy action selection depends on their differences.
            centered = error - error.sum(1, keepdim=True)/mask[ids].sum(1, keepdim=True)
            loss = (error.square().sum() + 10*torch.where(mask[ids], centered, 0.).square().sum())/mask[ids].sum()
            if args.objective == 'ranking':
                # Controlled alternative: explicitly fit the teacher's best
                # move, while weakly retaining Q-value calibration.
                logits = (pred/.1).masked_fill(~mask[ids], -torch.inf)
                best = target[ids].masked_fill(~mask[ids], -torch.inf).argmax(1)
                loss = torch.nn.functional.cross_entropy(logits, best) + .01*loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            if (update+1) % 100 == 0:
                with torch.no_grad():
                    v = torch.tensor(val_ids, device='mps')
                    vp = model(x[v]).masked_fill(~mask[v], -torch.inf)
                    accuracy = (vp.argmax(1)==target[v].masked_fill(~mask[v], -torch.inf).argmax(1)).float().mean().item()
                history.append(dict(update=update+1, train_loss=loss.item(), validation_action_agreement=accuracy))
        metadata = dict(method='Offline regression to exact frozen-Q lookahead targets', teacher_depth=depth,
                        training_games=list(range(8610000,8610012)), label_validation_games=list(range(8610012,8610016)),
                        training_boards=len(train_ids), validation_boards=len(val_ids), updates=args.updates,
                        batch=256, learning_rate=args.lr, action_gap_loss_weight=10, seconds=time.time()-start,
                        objective=args.objective, data_dir=str(data_dir), evaluation_split='validation',
                        history=history, parent=str(args.student_checkpoint), teacher='runs/research/dqn16_planning_selected/network',
                        gamma=config['gamma'], reward_mode=config['reward_mode'], shaping_scale=config['shaping_scale'])
        student = NeuralAgent(model, 'fitted_lookahead_q', 'mps', parent.width)
        student.save(args.out / f'student_depth{depth}', metadata)
        (args.out / f'training_depth{depth}.json').write_text(json.dumps(metadata, indent=2))
        results[f'student_depth{depth}'] = student
        print('TRAIN', depth, metadata['seconds'], history[-1], flush=True)
    results['original_direct'] = NeuralAgent.load(args.student_checkpoint, 'mps')
    for name, agent in results.items():
        @torch.no_grad()
        def decision(b):
            q = agent.policy(torch.as_tensor(b, dtype=torch.long, device='mps')).cpu().numpy()
            return np.where(legal_masks(b, *row_tables()), q, -np.inf)
        result = evaluate_batch(None, range(8620000,8620100), deadline=time.time()+120, decision=decision)
        result['summary'].pop('depth')
        result['summary']['decision'] = 'direct legal argmax, no search'
        (args.out / f'evaluation_{name}.json').write_text(json.dumps(result, indent=2))
        print('EVAL', name, result['summary'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('runs/research/deeper_q/distillation'))
    p.add_argument('--depths', type=int, nargs='+', default=[2, 3])
    p.add_argument('--boards-per-game', type=int, default=128)
    p.add_argument('--updates', type=int, default=2000)
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--objective', choices=['regression','ranking'], default='regression')
    p.add_argument('--student-checkpoint', type=Path, default=Path('runs/research/dqn16_planning_selected/network'))
    p.add_argument('--lr', type=float, default=1e-4)
    run(p.parse_args())
