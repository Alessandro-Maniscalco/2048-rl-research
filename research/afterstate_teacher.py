"""A frozen n-tuple value teacher, followed by the student's own online TD.

Teacher labels are future RAW merge points after a slide, divided by 128.
Entire collection games belong to fitting or label validation, never both.
No monitoring, selection, final-test, or diagnostic game supplies training data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from rl2048.afterstate_compare import moves, tuple_values
from rl2048.agents.neural import optimize, tensor_boards
from rl2048.agents.ntuple import NTupleAgent, row_tables
from rl2048.offline_data import collect_game


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def collect(checkpoint, out, expert_games=32, random_games=8, seed_start=10700000,
            boards_per_game=256, stop_file=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    if seed_start < 10000000:
        raise ValueError('Use a new training collection range above 10,000,000')
    frozen = out / 'teacher'
    shutil.copytree(checkpoint, frozen)
    teacher = NTupleAgent.load(frozen, mmap_mode='r')
    # This selected local table was trained by raw-score undiscounted TD.
    # Its deployment depth/downgrade settings are deliberately not used here.
    checkpoint_hash = digest(frozen / 'weights.npy')
    rng = np.random.default_rng(771)
    samples, game_ids, episodes = [], [], []
    start = time.perf_counter()
    for game in range(expert_games + random_games):
        if stop_file and Path(stop_file).exists():
            write(out / 'status.json', dict(complete=False, reason='stop_file'))
            return
        random_policy = game >= expert_games
        states, actions, rewards, _, terminal = collect_game(
            teacher.weights, teacher.patterns, *row_tables(), seed_start + game, random_policy)
        if not len(states) or not terminal[-1]:
            raise RuntimeError('Teacher dataset requires complete uncapped games')
        indices = np.sort(rng.choice(len(states), min(boards_per_game, len(states)), replace=False))
        samples.append(states[indices])
        game_ids.extend([game] * len(indices))
        episodes.append(dict(game=game, seed=seed_start + game, random=random_policy,
                             length=len(actions), score=float(rewards.sum()), samples=len(indices)))
        print('TEACHER_GAME', json.dumps(episodes[-1]), flush=True)
    states = np.concatenate(samples)
    after, gains, legal = moves(states, *row_tables())
    labels = np.zeros(legal.shape, np.float32)
    # All legal alternatives are labelled, not just the teacher's chosen move.
    labels[legal] = tuple_values(after[legal], teacher.weights, teacher.patterns) / 128
    if not np.isfinite(labels).all():
        raise ValueError('Nonfinite teacher labels')
    games = np.asarray(game_ids, np.int32)
    validation = games % 4 == 3
    if not validation.any() or validation.all():
        raise ValueError('Need independent fitting and validation games')
    np.savez_compressed(out / 'data.npz', states=states, afterstates=after,
                        gains=gains, legal=legal, values=labels, games=games,
                        validation=validation)
    write(out / 'manifest.json', dict(complete=True, teacher_checkpoint=str(frozen.resolve()),
        teacher_sha256=checkpoint_hash, source_checkpoint=str(Path(checkpoint).resolve()),
        teacher_metadata=json.loads((frozen/'metadata.json').read_text()),
        data_sha256=digest(out/'data.npz'), label_units='future raw merge points / 128',
        gamma=1.0, collection_depth=1, root_downgrading=False, teacher_value_clipping=False,
        collection_policy='one exact slide plus n-tuple value; additional random complete games',
        teacher_search_nonnegative_value=True, training_seeds=[e['seed'] for e in episodes],
        episodes=episodes, source_transitions=sum(e['length'] for e in episodes),
        labelled_states=len(states), labelled_afterstates=int(legal.sum()),
        split='whole games: game index modulo4 equals3 for label validation',
        collection_seconds=time.perf_counter()-start,
        note='Local n-tuple teacher pretraining cost predates this experiment and is not free student experience.'))


def validation_metrics(model, data, device, batch=512):
    ids = np.flatnonzero(data['validation'])
    after, legal = data['afterstates'][ids], data['legal'][ids]
    target = data['values'][ids]
    prediction = np.zeros(legal.shape, np.float32)
    boards = after[legal]
    with torch.no_grad():
        chunks = [model(tensor_boards(boards[i:i+batch], device)).cpu().numpy()
                  for i in range(0, len(boards), batch)]
    prediction[legal] = np.concatenate(chunks)
    student_q = np.where(legal, data['gains'][ids]/128 + prediction, -np.inf)
    teacher_q = np.where(legal, data['gains'][ids]/128 + target, -np.inf)
    # Agreement accepts any tied teacher-best action, avoiding orientation ties.
    chosen = student_q.argmax(1)
    teacher_best = teacher_q.max(1)
    chosen_value = teacher_q[np.arange(len(ids)), chosen]
    errors = np.where(legal, prediction-target, 0.)
    centered = errors-errors.sum(1,keepdims=True)/legal.sum(1,keepdims=True)
    return dict(value_mse=float(np.mean(errors[legal]**2)),
                centered_value_mse=float(np.mean(centered[legal]**2)),
                teacher_action_agreement=float(np.mean(np.isclose(chosen_value, teacher_best, atol=1e-5, rtol=1e-6))),
                mean_teacher_action_gap=float(np.mean(teacher_best-chosen_value)))


def grouped_distillation_loss(prediction, target, legal, gap_weight):
    """Fit absolute values plus errors in relative action values on each board.

    Q adds the same known immediate reward to prediction and target, so the
    prediction error equals the U error here. Invalid moves contribute nothing.
    Centering removes each board's common error offset, emphasizing rankings.
    """
    error = torch.where(legal, prediction-target, 0.)
    offset = error.sum(1,keepdim=True)/legal.sum(1,keepdim=True).clamp_min(1)
    centered = torch.where(legal, error-offset, 0.)
    count = legal.sum().clamp_min(1)
    return error.square().sum()/count + gap_weight*centered.square().sum()/count


def warm_start(learner, config, out):
    """Fit only the online policy; sync target and reset Adam before online RL.

    No replay experiences are carried from the teacher into the online phase.
    The online trainer's RNG and exploration schedule are unchanged by this fit.
    """
    settings = config['teacher_pretrain']
    folder = Path(settings['dataset'])
    manifest = json.loads((folder/'manifest.json').read_text())
    if (not manifest['complete'] or manifest['gamma'] != config['gamma']
            or manifest['label_units'] != 'future raw merge points / 128'
            or digest(folder/'data.npz') != manifest['data_sha256']):
        raise ValueError('Teacher data are incomplete, changed, or use incompatible value units')
    if config.get('resume'):
        raise ValueError('Teacher warm start is a fresh initialization, not a training resume')
    with np.load(folder/'data.npz', allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    fit = ~data['validation']
    fit_states = np.flatnonzero(fit)
    boards = data['afterstates'][fit][data['legal'][fit]]
    labels = data['values'][fit][data['legal'][fit]]
    if not len(boards) or not np.isfinite(labels).all():
        raise ValueError('Empty or nonfinite teacher fitting data')
    rng = np.random.default_rng(settings.get('seed', 771))
    start = time.perf_counter()
    history = [dict(updates=0, seconds=0., **validation_metrics(learner.policy, data, learner.device))]
    updates = 0
    reason = 'update_budget'
    for update in range(settings['updates']):
        if config.get('stop_file') and Path(config['stop_file']).exists():
            reason = 'stop_file'; break
        if time.perf_counter()-start >= settings['seconds']:
            reason = 'teacher_time_budget'; break
        if settings.get('grouped_actions',False):
            ids = rng.choice(fit_states,size=settings['batch'])
            x = tensor_boards(data['afterstates'][ids].reshape(-1,16),learner.device)
            y = torch.as_tensor(data['values'][ids],device=learner.device)
            legal = torch.as_tensor(data['legal'][ids],device=learner.device)
            loss = grouped_distillation_loss(learner.policy(x).reshape(-1,4),y,legal,
                                             settings.get('gap_weight',0.))
        else:
            ids = rng.integers(len(boards), size=settings['batch'])
            x = tensor_boards(boards[ids], learner.device)
            y = torch.as_tensor(labels[ids], device=learner.device)
            loss = (learner.policy(x)-y).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite teacher distillation loss')
        optimize(learner.optimizer, loss, learner.policy.parameters())
        updates = update+1
        if updates % settings.get('validation_interval', 512) == 0:
            row = dict(updates=updates, seconds=time.perf_counter()-start,
                       **validation_metrics(learner.policy, data, learner.device))
            history.append(row)
            write(Path(out)/'teacher_curve.json', history)
            print('TEACHER_FIT', json.dumps(row), flush=True)
    if history[-1]['updates'] != updates:
        history.append(dict(updates=updates, seconds=time.perf_counter()-start,
                            **validation_metrics(learner.policy, data, learner.device)))
    learner.target.load_state_dict(learner.policy.state_dict())
    # Distillation gradients must not bias the first TD Adam moments.
    learner.optimizer.state.clear()
    result = dict(updates=updates, seconds=time.perf_counter()-start, stop_reason=reason,
        fitting_afterstates=len(boards), data_sha256=manifest['data_sha256'],
        teacher_sha256=manifest['teacher_sha256'], dataset=str(folder.resolve()),
        source_transitions=manifest['source_transitions'], final_validation=history[-1],
        grouped_actions=settings.get('grouped_actions',False),gap_weight=settings.get('gap_weight',0.),
        target_synchronized=True, optimizer_reset=True,
        online_teacher_loss=False, teacher_data_in_online_replay=False)
    write(Path(out)/'teacher_curve.json', history)
    write(Path(out)/'teacher_pretrain.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expert-games', type=int, default=32)
    parser.add_argument('--random-games', type=int, default=8)
    parser.add_argument('--seed-start', type=int, default=10700000)
    parser.add_argument('--boards-per-game', type=int, default=256)
    parser.add_argument('--stop-file', type=Path)
    args = parser.parse_args()
    collect(**vars(args))
