import numpy as np
import pytest
import torch
from rl2048.agents.ppo import advantages, clipped_policy_loss
from rl2048.agents.neural import masked_log_probs
from rl2048.vector_game import VectorGame, legal_masks, step_boards
from rl2048.agents.ntuple import encode, row_tables
from rl2048.game import move, legal_actions


def test_gae_terminal_and_timeout_hand_calculation():
    rewards=np.array([[1.,1.],[2.,2.]],np.float32)
    values=np.array([[3.,3.],[4.,4.]],np.float32)
    next_values=np.array([[4.,4.],[99.,5.]],np.float32)
    term=np.array([[False,False],[True,False]])
    trunc=np.array([[False,False],[False,True]])
    adv,returns=advantages(rewards,values,next_values,term,trunc,gamma=.5,lam=1)
    np.testing.assert_allclose(adv,[[-1.,.25],[-2.,.5]])
    np.testing.assert_allclose(returns,adv+values)


def test_ppo_clips_only_advantage_improving_direction():
    # positive advantage: cap at1.2; negative advantage: keep damaging ratio1.5
    new=torch.tensor([1.5,1.5]).log().requires_grad_()
    loss=clipped_policy_loss(new,torch.zeros(2),torch.tensor([2.,-2.]))
    assert loss.item()==pytest.approx(.3)
    loss.backward()
    assert new.grad[0].item()==0
    assert new.grad[1].item()==pytest.approx(1.5)


def test_mask_probabilities_and_terminal_finiteness():
    masks=torch.tensor([[True,False,True,False],[False,False,False,False]])
    logs=masked_log_probs(torch.tensor([[0.,100.,0.,100.],[1.,2.,3.,4.]]),masks)
    assert torch.isfinite(logs).all()
    torch.testing.assert_close(logs.exp()[0],torch.tensor([.5,0.,.5,0.]))


def test_vector_game_preserves_final_state_at_time_limit():
    env=VectorGame(1,3,max_steps=1)
    before=env.boards.copy()
    action=np.flatnonzero(env.masks()[0])[0]
    reward,final,term,trunc,episodes=env.step([action])
    assert trunc[0] and not term[0]
    assert episodes[0,1]==1 and env.lengths[0]==0
    assert np.count_nonzero(env.boards)==2
    assert np.count_nonzero(final)>=2
    assert not np.array_equal(final,env.boards)


def test_vector_masks_match_reference():
    rng=np.random.default_rng(11)
    rows,rewards=row_tables()
    boards=rng.integers(0,12,(100,16),dtype=np.uint8)
    masks=legal_masks(boards,rows,rewards)
    for board,mask in zip(boards,masks):
        actual=np.where(board>0,2**board.astype(np.int64),0).reshape(4,4)
        np.testing.assert_array_equal(mask,legal_actions(actual))


def test_expectile_and_conservative_penalty_hand_values():
    from rl2048.agents.iql import expectile_loss
    from rl2048.agents.cql import conservative_penalty
    assert expectile_loss(torch.tensor([-2.,2.]),.7).item()==pytest.approx(2.)
    q=torch.tensor([[0.,999.,0.,999.]])
    mask=torch.tensor([[True,False,True,False]])
    assert conservative_penalty(q,torch.tensor([0]),mask).item()==pytest.approx(np.log(2))


def test_sac_soft_value_exact_expectation():
    from rl2048.agents.sac import soft_value
    logs=masked_log_probs(torch.zeros(1,4),torch.tensor([[True,True,False,False]]))
    actual=soft_value(logs,torch.tensor([[2.,4.,999.,999.]]),.1)
    assert actual.item()==pytest.approx(3.+.1*np.log(2))


def test_offline_returns_and_replay_wrap():
    from rl2048.offline_data import discounted_returns,ReplayBuffer
    actual=discounted_returns(np.array([128.,256.,512.]),np.array([False,True,True]),np.zeros(3,bool),.5)
    np.testing.assert_array_equal(actual,[2.,2.,4.])
    replay=ReplayBuffer(3)
    for start in (0,2):
        batch={key:np.zeros((2,)+value.shape[1:],value.dtype) for key,value in replay.data.items()}
        batch['actions']=np.arange(start,start+2)
        replay.add(**batch)
    assert replay.size==3
    assert set(replay.data['actions'])=={1,2,3}


@pytest.mark.parametrize('method',['awr','iql','cql','sac'])
def test_algorithms_terminal_bellman_and_finite_updates(method):
    from rl2048.agents.awr import AWR
    from rl2048.agents.iql import IQL
    from rl2048.agents.cql import CQL
    from rl2048.agents.sac import SAC
    learner={'awr':AWR,'iql':IQL,'cql':CQL,'sac':SAC}[method](width=16,lr=0.)
    for network in vars(learner).values():
        if isinstance(network,torch.nn.Module):
            for param in network.parameters():
                torch.nn.init.zeros_(param)
    batch={'states':torch.zeros(2,16,dtype=torch.uint8),'next_states':torch.ones(2,16,dtype=torch.uint8),
           'actions':torch.tensor([0,2]),'rewards':torch.tensor([128.,256.]),
           'terminated':torch.tensor([True,True]),'truncated':torch.tensor([False,False]),
           'masks':torch.ones(2,4,dtype=torch.bool),'next_masks':torch.zeros(2,4,dtype=torch.bool),
           'returns':torch.tensor([1.,2.])}
    losses=learner.update(batch)
    assert all(torch.isfinite(v).all() for v in losses.values())
    if method in ('iql','sac'):
        assert losses['q_loss'].item()==pytest.approx(5.) # 2 critics * mean([1^2,2^2])
    if method=='cql':
        assert losses['bellman_loss'].item()==pytest.approx(2.5)
    if method=='awr':
        assert losses['value_loss'].item()==pytest.approx(2.5)


def test_neural_checkpoint_portable_exact(tmp_path):
    from rl2048.agents.neural import NeuralAgent,BoardNet
    from rl2048.agents.feature_q import load_agent
    agent=NeuralAgent(BoardNet(5,16),'ppo',width=16)
    agent.save(tmp_path/'checkpoint')
    restored=load_agent(tmp_path/'checkpoint')
    board=torch.randint(0,17,(2,16))
    torch.testing.assert_close(agent.policy(board),restored.policy(board))


def test_sampling_never_selects_zero_probability_even_at_zero_draw():
    from rl2048.agents.neural import sample_actions
    class ZeroRNG:
        def random(self,size):
            return np.zeros(size)
    probabilities=np.array([[0.,.5,.5,0.],[0.,0.,0.,1.]])
    np.testing.assert_array_equal(sample_actions(probabilities,ZeroRNG()),[1,3])


def test_potential_shaping_telescopes_at_terminal():
    from rl2048.rewards import learning_rewards,snake_potential
    states=np.arange(48,dtype=np.uint8).reshape(3,16)%12
    next_states=np.vstack([states[1:],states[:1]])
    term=np.array([False,False,True]); raw=np.array([4.,8.,16.])
    gamma=.9; scale=2.
    shaped=learning_rewards(raw,states,next_states,term,gamma,'bottom_left',scale)
    difference=np.dot(gamma**np.arange(3),shaped-raw/128)
    assert difference==pytest.approx(-scale*snake_potential(states[:1])[0],abs=1e-5)


def test_symmetry_action_mapping_commutes_with_game_move():
    from rl2048.symmetry import transform_batch
    from rl2048.agents.ntuple import decode
    rng=np.random.default_rng(7)
    boards=rng.integers(0,8,(30,16),dtype=np.uint8)
    for reflection in (False,True):
        for rotation in range(4):
            for action in range(4):
                sample={'states':boards,'actions':np.full(len(boards),action)}
                transformed=transform_batch(sample,rotation,reflection)
                for i,board in enumerate(boards):
                    expected,reward,_=move(decode(board),action)
                    if reflection:
                        expected=expected[:,::-1]
                    expected=np.rot90(expected,rotation)
                    actual,gain,_=move(decode(transformed['states'][i]),int(transformed['actions'][i]))
                    np.testing.assert_array_equal(actual,expected)
                    assert gain==reward


def test_sixteen_input_encodings_and_four_outputs():
    from rl2048.agents.dqn import QNetwork
    state=torch.tensor([[0,1,2,3]+[0]*12])
    expected={'exponents':[0,1/16,2/16,3/16], 'raw':[0,2/65536,4/65536,8/65536], 'relative':[0,.25,.5,1.]}
    for mode,values in expected.items():
        network=QNetwork(16,mode)
        torch.testing.assert_close(network.encode_inputs(state)[0,:4],torch.tensor(values))
        assert network.layers[0].in_features==16 and network(state).shape==(1,4)


def test_dqn_double_target_selection_and_terminal_mask():
    from rl2048.agents.dqn import bellman_target
    online=torch.tensor([[3.,2.,100.,0.],[3.,2.,100.,0.]])
    target=torch.tensor([[4.,10.,100.,0.],[4.,10.,100.,0.]])
    masks=torch.tensor([[True,True,False,False],[False,False,False,False]])
    reward=torch.tensor([1.,2.]);terminal=torch.tensor([False,True])
    torch.testing.assert_close(bellman_target(reward,terminal,online,target,masks,.5,True),torch.tensor([3.,2.]))
    torch.testing.assert_close(bellman_target(reward,terminal,online,target,masks,.5,False),torch.tensor([6.,2.]))


def test_dueling_aggregation_and_checkpoint(tmp_path):
    from rl2048.agents.dueling_dqn import DuelingQNetwork
    from rl2048.agents.neural import NeuralAgent
    model=DuelingQNetwork(16)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    with torch.no_grad():
        model.value_head.bias.fill_(2.)
        model.layers[-1].bias.copy_(torch.tensor([1.,2.,3.,4.]))
    state=torch.zeros(1,16,dtype=torch.long)
    torch.testing.assert_close(model(state),torch.tensor([[.5,1.5,2.5,3.5]]))
    NeuralAgent(model,'dueling_double_dqn',width=16).save(tmp_path)
    restored=NeuralAgent.load(tmp_path)
    torch.testing.assert_close(model(state),restored.policy(state))
def test_raw_tile_scale_survives_checkpoint(tmp_path):
    from rl2048.agents.dqn import QNetwork
    from rl2048.agents.neural import NeuralAgent
    model=QNetwork(input_encoding='raw',raw_divisor=16)
    board=torch.tensor([[0,1,2,3]+[0]*12])
    assert torch.allclose(model.encode_inputs(board)[0,:4],torch.tensor([0.,.125,.25,.5]))
    NeuralAgent(model,'double_dqn').save(tmp_path)
    loaded=NeuralAgent.load(tmp_path)
    assert loaded.policy.raw_divisor==16
    assert torch.allclose(model(board),loaded.policy(board))
