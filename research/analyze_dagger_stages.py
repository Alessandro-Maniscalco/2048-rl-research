"""Frozen stage diagnostic and common-seed pretrained-CNN rescore.

The stratified label sample diagnoses errors; only complete-game scores
measure playing strength. No optimizer or new fitting data is created.
"""
import json
from pathlib import Path
import time

import numpy as np
import torch

from research.afterstate_teacher import digest, write
from research.dagger_experiment import from_archive, validate
from research.teacher_policy_experiment import evaluate
from rl2048.agents.neural import NeuralAgent


def main():
    torch.set_num_threads(1)
    base = Path('runs/research/scaled_transformer')
    out = base/'dagger_stage_diagnostic'
    out.mkdir(exist_ok=False)
    archive = base/'transformer_teacher_data/data.npz'
    manifest = json.loads((archive.parent/'manifest.json').read_text())
    assert digest(archive) == manifest['data_sha256']
    with np.load(archive, allow_pickle=False) as saved:
        ids = np.flatnonzero(saved['validation'])
        held = from_archive({k:saved[k][ids] for k in ('states', 'legal', 'values', 'gains')})
    rng = np.random.default_rng(8958000)
    bins = np.digitize(held['states'].max(1), (8, 10, 12), right=True)
    samples = {}
    for stage in range(4):
        pool = np.flatnonzero(bins == stage)
        samples[stage] = np.sort(rng.choice(pool, min(2048, len(pool)), replace=False))
    actors = {name:base/f'transformer_dagger_stage_{name}_seed0/last'
              for name in ('uniform', 'balanced')}
    cnn = Path('runs/research/pretrained_cnn/original')
    hashes = {name:digest(path/'policy.pt') for name,path in actors.items()}
    hashes['pretrained_cnn'] = digest(cnn/'policy.pt')
    write(out/'protocol.json', dict(device='mps', sample_seed=8958000,
        data_sha256=manifest['data_sha256'], model_sha256=hashes,
        stage_max_rank_boundaries=[8, 10, 12],
        original_archive_indices={str(k):ids[v].tolist() for k,v in samples.items()},
        cnn_monitor_seeds=list(range(8900000, 8900128)),
        cnn_selection_seeds=list(range(8910000, 8910100)),
        note='Frozen diagnostic. Label games held out from fitting but previously reused for monitoring. '
             'Stage-balanced sample is not a population score estimate. CNN pretraining differs from Transformer.'))
    report = {}
    for name,path in actors.items():
        agent = NeuralAgent.load(path, 'mps')
        report[name] = {}
        for stage, chosen in samples.items():
            if not len(chosen):
                continue
            result = validate(agent.policy, {k:v[chosen] for k,v in held.items()}, 'mps', 128)
            result['mean_teacher_gap_raw_points'] = result['mean_teacher_action_gap']*128.
            report[name][str(stage)] = result | dict(sample_boards=len(chosen))
        del agent
        torch.mps.empty_cache()
    write(out/'stage_errors.json', report)
    print('STAGE_ERRORS', json.dumps(report), flush=True)
    agent = NeuralAgent.load(cnn, 'mps')
    for name,start,count in [('monitor', 8900000, 128), ('selection', 8910000, 100)]:
        before = time.perf_counter()
        result = evaluate(agent, range(start, start+count))
        result['wall_seconds'] = time.perf_counter()-before
        write(out/f'cnn_{name}.json', result)
        print('CNN_RESCORE', name, json.dumps(result['summary']), flush=True)
    for name,path in actors.items():
        assert digest(path/'policy.pt') == hashes[name]
    assert digest(cnn/'policy.pt') == hashes['pretrained_cnn']
    assert digest(archive) == manifest['data_sha256']
    write(out/'verified.json', dict(complete=True, weights_and_data_unchanged=True))


if __name__ == '__main__':
    main()
