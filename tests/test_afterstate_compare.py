import numpy as np
import torch
from rl2048.afterstate_compare import (SymmetricValue,TupleValue,NeuralValue,AfterstateAgent,
    moves,plan,td_targets,evaluate_batch)
from rl2048.agents.ntuple import row_tables
from rl2048.fast2048 import chance_value
from rl2048.evaluate import rollout


def test_symmetry_is_exact_sharing():
    torch.manual_seed(7);net=SymmetricValue(16,'relational')
    torch.nn.init.normal_(net.base.layers[-1].weight)
    board=np.array([[1,2,0,4],[0,5,2,1],[4,3,2,0],[1,0,2,3]])
    versions=np.array([np.rot90(board,k).copy().ravel() for k in range(4)]+
                      [np.rot90(np.fliplr(board),k).copy().ravel() for k in range(4)])
    output=net(torch.tensor(versions)).detach().numpy()
    np.testing.assert_allclose(output,output[0],atol=1e-5)


def test_collision_normalization_changes_prediction_by_alpha_error():
    table=TupleValue('4x4',alpha=.1);b=np.zeros((1,16),np.uint8)
    table.fit(b,np.array([10.],np.float32))
    np.testing.assert_allclose(table.values(b),[1.],atol=1e-6)


def test_afterstate_target_uses_next_move_reward_and_terminal_zero():
    class Constant:
        def values(self,b,target=False):return np.full(len(b),10,np.float32)
    b=np.zeros((1,16),np.uint8);b[0,:2]=[1,1]
    np.testing.assert_allclose(td_targets(b,Constant()),[10+4/128])
    terminal=np.array([[1,2,1,2,2,1,2,1,1,2,1,2,2,1,2,1]],np.uint8)
    np.testing.assert_array_equal(td_targets(terminal,Constant()),[0])


def test_shared_depth2_matches_existing_exact_expectimax():
    table=TupleValue('4x4');rng=np.random.default_rng(4)
    table.weights[:]=rng.uniform(0,.1,table.weights.shape)
    board=np.array([[1,2,1,2,2,1,2,1,1,2,3,4,0,0,2,2]],np.uint8)
    after,rewards,legal=moves(board,*row_tables())
    reference=[]
    for a in range(4):
        reference.append(rewards[0,a]+chance_value(after[0,a],table.weights*128,table.patterns,*row_tables(),1,1.,0.) if legal[0,a] else -np.inf)
    np.testing.assert_allclose(plan(board,table,2)[0]*128,reference,rtol=1e-6)


def test_batched_evaluation_reproduces_reference_game_rng():
    torch.set_num_threads(1);model=NeuralValue(8,device='cpu')
    result=evaluate_batch(model,[8300111],max_steps=1000)
    reference,_=rollout(AfterstateAgent(model),seed=8300111,max_steps=1000)
    for key in ('score','length','max_tile'):
        assert result['episodes'][0][key]==reference[key]
