import csv
import json
import numpy as np
from research.transformer_td_experiment import config, train_one
from rl2048.agents.neural import NeuralAgent


def test_dense_monitor_checkpoints_and_training_counts(tmp_path):
    c=config('mlp_q','exponents',3)|dict(width=16,warmup=1024,batch=32,
        monitor_games=8,monitor_interval=2048,monitor_initial=True,dense_metrics=True,
        save_monitor_checkpoints=True,monitor_seed_start=8790000,final_seed_start=8800000)
    out=tmp_path/'run'
    result=train_one(c,0,4096,out,'cpu')
    curve=json.loads((out/'curve.json').read_text())
    assert [r['transitions'] for r in curve]==[0,2048,4096]
    assert [r['updates'] for r in curve]==[0,9,25]
    assert all(r['completed_episodes']==8 and not r['truncated_episodes'] for r in curve)
    assert result['updates']==25 and result['replay_records']==4096
    with (out/'progress.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==25 and [int(r['updates']) for r in rows]==list(range(1,26))
    assert all(float(r['mean_live_score_128'])>=0 for r in rows)
    assert all(float(r['mean_live_moves_128'])>=0 for r in rows)
    for step in (0,2048,4096):
        path=out/'checkpoints'/f'{step:09d}'
        data=json.loads((path/'monitor_evaluation.json').read_text())
        assert [e['seed'] for e in data['episodes']]==list(range(8790000,8790008))
        assert data['summary']['mean_score']==np.mean([e['score'] for e in data['episodes']])
        NeuralAgent.load(path)
    # A continuation loads saved online/target/optimizer state and reports
    # cumulative experience, while explicitly rebuilding its replay buffer.
    resumed=train_one(c|dict(resume=str(out)),0,4096,tmp_path/'resume','cpu')
    assert resumed['prior_transitions']==4096 and resumed['total_transitions']==8192
    assert resumed['training_transitions']==4096 and resumed['updates']==25


def test_stop_file_saves_without_starting_more_games(tmp_path):
    stop=tmp_path/'STOP';stop.touch()
    c=config('mlp_q','exponents',1)|dict(width=16,stop_file=str(stop))
    result=train_one(c,0,4096,tmp_path/'stopped','cpu')
    assert result['training_transitions']==0 and result['updates']==0
    assert result['stop_reason']=='stop_file' and result['mean_score'] is None
    assert not (tmp_path/'stopped/evaluation.json').exists()
    NeuralAgent.load(tmp_path/'stopped/last')
