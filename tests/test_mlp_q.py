import numpy as np
import pytest
import torch
from rl2048.agents.mlp_q import MLPQNetwork
from rl2048.agents.dqn import DQN
from rl2048.agents.neural import NeuralAgent

@pytest.mark.parametrize('encoding', ['exponents','one_hot','embedding','relational'])
@pytest.mark.parametrize('depth,residual', [(2,False),(4,True)])
def test_train_and_checkpoint(tmp_path,encoding,depth,residual):
    torch.manual_seed(7)
    learner=DQN(architecture='mlp_q',input_encoding=encoding,width=32,depth=depth,residual=residual)
    b=torch.randint(0,12,(8,16))
    batch=dict(states=b,next_states=b,actions=torch.arange(8)%4,rewards=torch.ones(8),
               terminated=torch.zeros(8,dtype=torch.bool),next_masks=torch.ones(8,4,dtype=torch.bool))
    before=learner.policy(b).detach().clone()
    loss=learner.update(batch)['q_loss']
    assert torch.isfinite(loss)
    assert not torch.equal(before,learner.policy(b))
    agent=NeuralAgent(learner.policy,'double_dqn',width=32)
    agent.save(tmp_path)
    loaded=NeuralAgent.load(tmp_path)
    torch.testing.assert_close(loaded.policy(b),learner.policy(b))
    assert loaded.act(np.array([[2,2,4,0]]*4),np.array([False,False,True,False]))==2

def test_relational_scale_and_empty_semantics():
    m=MLPQNetwork(input_encoding='relational')
    b=torch.tensor([[2,2,1,0]+[0]*12,[3,3,2,0]+[0]*12])
    x=m.encode_inputs(b)
    assert x.shape==(2,80)
    assert not torch.equal(x[0,:16],x[1,:16])
    torch.testing.assert_close(x[0,16:],x[1,16:])
    assert x[0,32+1]==1/16 # exponent difference between 4 and 2
    assert x[0,32+2]==0 # no invented difference to an empty cell
    assert x[0,56]==1 # equal occupied neighbors
    assert x[0,56+3]==0 # two empty cells are not a merge
