import numpy as np
import torch

from rl2048.agents.afterstate_mlp import AfterstateLearner, AfterstateMLPAgent, afterstate_targets
from research.neural_afterstate_experiment import AfterstateReplay, train_one
from rl2048.vector_game import VectorGame


def test_next_move_reward_double_selection_and_terminal():
    # Action1 wins online (4 + .5*8=8), but action0 has higher target value.
    # Illegal action2 must never win, despite enormous reward/value.
    gains = torch.tensor([[256., 512., 99999., 0.]]).repeat(3, 1)
    legal = torch.tensor([[True, True, False, False], [True, True, False, False], [False]*4])
    online = torch.tensor([[1., 8., 1000., 0.]]).repeat(3, 1)
    target = torch.tensor([[100., 6., 1000., 0.]]).repeat(3, 1)
    y = afterstate_targets(gains, legal, online, target, torch.tensor([False, True, False]), .5)
    torch.testing.assert_close(y, torch.tensor([7., 0., 0.]))


def test_replay_copies_actual_spawned_state_and_wraps():
    buffer = AfterstateReplay(3)
    b = np.arange(32, dtype=np.uint8).reshape(2, 16)
    buffer.add(b, b+1, np.array([False, True])); b[:] = 0
    assert buffer.afterstates[0, 1] == 1 and buffer.next_states[1, 0] == 17
    buffer.add(np.full((2, 16), 7, np.uint8), np.full((2, 16), 8, np.uint8), np.array([False, False]))
    assert buffer.size == 3 and buffer.position == 1
    assert buffer.terminated.tolist() == [False, True, False]


def test_saved_policy_and_training_update_roundtrip(tmp_path):
    torch.set_num_threads(1); torch.manual_seed(0)
    learner = AfterstateLearner(width=16)
    env = VectorGame(8, 4321)
    q, after, legal = learner.decision(env.boards)
    # Zero initial U means exact immediate merges determine initial rankings.
    assert np.isfinite(q[legal]).all()
    actions = q.argmax(1)
    selected = after[np.arange(8), actions]
    _, nxt, term, _, _ = env.step(actions)
    metrics = learner.update(selected, nxt, term)
    assert all(np.isfinite(v) for v in metrics.values())
    agent = AfterstateMLPAgent(learner)
    agent.save(tmp_path, dict(width=16))
    restored = AfterstateMLPAgent.load(tmp_path)
    np.testing.assert_array_equal(learner.decision(nxt)[0], restored.learner.decision(nxt)[0])


def test_replay_ratio_changes_updates_not_environment_budget(tmp_path):
    torch.set_num_threads(1)
    config=dict(algorithm='neural_afterstate',architecture='afterstate_sym_mlp',
        width=8,depth=2,input_encoding='embedding',embedding_dim=4,n=1,reward_mode='score',
        gamma=1.,lr=1e-4,target_tau=.005,weight_decay=.01,envs=8,batch=8,
        capacity=128,warmup=8,epsilon_end=.05,epsilon_steps=500000,
        monitor_interval=32,monitor_games=8,monitor_seed_start=11000000,final_seed_start=11001000)
    for ratio in (1,2):
        result=train_one(config|dict(updates_per_collection=ratio),0,32,tmp_path/f'ratio{ratio}','cpu')
        assert result['complete'] and result['training_transitions']==32
        assert result['updates']==4*ratio
        assert result['replay_examples_sampled']==32*ratio
