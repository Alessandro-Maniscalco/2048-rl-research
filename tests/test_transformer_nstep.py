import numpy as np
import pytest
import torch
from rl2048.agents.dqn import DQN, bellman_target
from rl2048.agents.neural import NeuralAgent
from rl2048.n_step import NStepReplay
from rl2048.rewards import learning_rewards


def transition(t, terminal=False, timeout=False):
    return dict(states=np.full((1,16), t, np.uint8),
        next_states=np.full((1,16), t+1, np.uint8), actions=np.array([t%4]),
        rewards=np.array([2.**t],np.float32), masks=np.ones((1,4),bool),
        next_masks=np.full((1,4),not terminal,bool),
        terminated=np.array([terminal]), truncated=np.array([timeout]))


@pytest.mark.parametrize('n', [1, 3, 5])
@pytest.mark.parametrize('terminal', [False, True])
def test_horizon_return_and_quantile_bootstrap_by_hand(n, terminal):
    from rl2048.agents.qr_dqn import quantile_targets
    replay = NStepReplay(20, 1, n=n, gamma=.5)
    for t in range(n):
        replay.add(**transition(t, terminal=terminal and t == n-1))
    data = replay.replay.data
    # Rewards 1,2,4,... discounted by .5 give exactly one per step.
    assert data['rewards'][0] == n
    assert data['discounts'][0] == .5**n
    assert data['next_states'][0, 0] == n
    q = torch.tensor([[[100., 100.], [2., 6.], [0., 0.], [0., 0.]]])
    target = quantile_targets(torch.tensor(data['rewards'][:1]),
        torch.tensor(data['terminated'][:1]), q, q,
        torch.tensor([[False, True, False, False]]), torch.tensor(data['discounts'][:1]))
    expected = torch.tensor([[float(n), float(n)]])
    if not terminal:
        expected += .5**n * torch.tensor([[2., 6.]])
    torch.testing.assert_close(target, expected)


def test_symmetry_keeps_nstep_rewards_discounts_and_terminal_labels():
    from rl2048.symmetry import transform_batch
    from rl2048.vector_game import legal_masks
    from rl2048.agents.ntuple import row_tables
    replay=NStepReplay(20,1,n=5,gamma=.99)
    for t in range(5):replay.add(**transition(t,terminal=t==4))
    batch={k:v[:replay.size].copy() for k,v in replay.replay.data.items()}
    # Check mask remapping independently on nonuniform boards.
    batch['next_states']=np.random.default_rng(11).integers(0,8,(5,16),dtype=np.uint8)
    batch['next_masks']=legal_masks(batch['next_states'],*row_tables())
    for rotation in range(4):
        for reflect in (False,True):
            transformed=transform_batch(batch,rotation,reflect)
            for key in ('rewards','discounts','terminated','truncated'):
                np.testing.assert_array_equal(transformed[key],batch[key])
            np.testing.assert_array_equal(transformed['next_masks'],
                legal_masks(transformed['next_states'],*row_tables()))


def test_online_symmetry_rejects_directional_shaping(tmp_path):
    from research.transformer_td_experiment import train_one
    with pytest.raises(ValueError,match='orientation-invariant'):
        train_one(dict(augment_symmetry=True,reward_mode='corner_snake'),0,128,tmp_path/'bad','cpu')
    assert not (tmp_path/'bad').exists()


@pytest.mark.parametrize('terminal,timeout', [(True,False),(False,True)])
def test_nstep_boundary_and_hand_targets(terminal,timeout):
    r=NStepReplay(10,1,n=3,gamma=.5)
    r.add(**transition(0)); r.add(**transition(1,terminal,timeout))
    assert r.size==2
    d=r.replay.data
    np.testing.assert_allclose(d['rewards'][:2],[2.,2.])
    np.testing.assert_allclose(d['discounts'][:2],[.25,.5])
    np.testing.assert_array_equal(d['next_states'][:2],2)
    next_q=torch.tensor([[100.,8.,200.,2.]]*2)
    target=bellman_target(torch.tensor(d['rewards'][:2]),torch.tensor(d['terminated'][:2]),
        next_q,next_q,torch.tensor([[False,True,False,True]]*2),torch.tensor(d['discounts'][:2]))
    torch.testing.assert_close(target,torch.tensor([2.,2.] if terminal else [4.,6.]))
    r.add(**transition(7));r.flush()
    assert d['states'][2,0]==7 and d['rewards'][2]==128 # no reset crossing


def test_three_step_overlap_flush_and_parallel_games():
    r=NStepReplay(20,2,n=3,gamma=.5)
    for t in range(4):
        a,b=transition(t),transition(t+5)
        r.add(**{k:np.concatenate((a[k],b[k])) for k in a})
    assert r.size==4
    d=r.replay.data
    np.testing.assert_allclose(d['rewards'][:4],[3.,96.,6.,192.])
    np.testing.assert_allclose(d['discounts'][:4],.125)
    r.flush()
    assert r.size==8
    assert all(not q for q in r.queues)


@pytest.mark.parametrize('encoding',['exponents','embedding','relative'])
def test_transformer_learns_positions_and_roundtrip(tmp_path,encoding):
    torch.manual_seed(9)
    learner=DQN(architecture='transformer_q',input_encoding=encoding,width=16,heads=4)
    b=torch.randint(0,12,(8,16))
    batch=dict(states=b,next_states=b,actions=torch.arange(8)%4,rewards=torch.ones(8),
        terminated=torch.zeros(8,dtype=torch.bool),next_masks=torch.ones(8,4,dtype=torch.bool),
        discounts=torch.full((8,),.99**3))
    before=learner.policy(b).detach().clone()
    assert torch.isfinite(learner.update(batch)['q_loss'])
    assert not torch.equal(before,learner.policy(b))
    assert learner.policy.positions.grad.abs().sum()>0
    assert learner.policy.tokenizer.weight.grad.abs().sum()>0
    learner.policy.eval()
    agent=NeuralAgent(learner.policy,'double_dqn',width=16)
    agent.save(tmp_path)
    loaded=NeuralAgent.load(tmp_path)
    torch.testing.assert_close(loaded.policy(b),learner.policy(b))
    assert loaded.act(np.full((4,4),2),np.array([False,False,True,False]))==2


def test_log_reward_is_changed_objective_and_no_position_bonus():
    b=np.zeros((3,16),np.uint8)
    result=learning_rewards(np.array([0,128,256]),b,b,np.zeros(3,bool),.99,'log_score')
    assert result[0]==0 and result[1]==pytest.approx(1)
    assert 1<result[2]<2


def test_relative_transformer_scale_invariance_empty_cells_and_entropy_path():
    from rl2048.agents.transformer_q import TransformerQNetwork
    model=TransformerQNetwork(width=16,input_encoding='relative',heads=4).eval()
    # Stored ranks represent [4,4,2,0] and [8,8,4,0].
    boards=torch.tensor([[2,2,1,0]*4,[3,3,2,0]*4,[0]*16,[17,1,0,16]*4])
    values=model.encode_inputs(boards)
    torch.testing.assert_close(values[0,:,0],torch.tensor([1.,1.,.5,0.]*4))
    torch.testing.assert_close(values[0],values[1])
    assert torch.equal(values[2],torch.zeros_like(values[2]))
    assert torch.isfinite(values).all() and values.min()>=0 and values.max()<=1
    assert values[3,1,0]==2**-16
    with torch.no_grad():
        q=model(boards);q_entropy,entropy=model.forward_with_entropy(boards)
    torch.testing.assert_close(q[0],q[1])
    torch.testing.assert_close(q,q_entropy)
    assert torch.isfinite(entropy).all()


def test_relative_rewards_scale_invariance_and_pre_move_maximum():
    # 4+4 -> 8 and 8+8 -> 16 are the same relative merge: reward ratio 2.
    states=np.array([[2,2,0,0]*4,[3,3,0,0]*4,[0]*16],dtype=np.uint8)
    after=np.array([[3,0,0,0]*4,[4,0,0,0]*4,[0]*16],dtype=np.uint8)
    raw=np.array([8,16,0]);terminated=np.array([False,True,False])
    args=(raw,states,after,terminated,.99)
    np.testing.assert_allclose(learning_rewards(*args,'relative_score'),[2,2,0])
    np.testing.assert_allclose(learning_rewards(*args,'log_relative_score'),[np.log(3),np.log(3),0])
    # Compress large ratios without changing the raw score or using next-state scale.
    np.testing.assert_array_equal(raw,[8,16,0])
    assert learning_rewards(*args,'log_relative_score').dtype==np.float32
