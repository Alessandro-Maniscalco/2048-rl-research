import numpy as np
import torch
from rl2048.agents.pretrained2048 import Pretrained2048
from rl2048.agents.neural import NeuralAgent


def test_action_mapping_and_checkpoint_reload(tmp_path):
    torch.set_num_threads(1);model=Pretrained2048()
    with torch.no_grad():
        model._actor._out.weight.zero_()
        model._actor._out.bias.copy_(torch.tensor([10.,20.,30.,40.]))
    board=torch.zeros((1,16),dtype=torch.long)
    # Original logits translated by max40, reordered up/right/down/left.
    np.testing.assert_allclose(model(board)[0,:4].detach().numpy(),[-10,-20,0,-30])
    agent=NeuralAgent(model,'ppo_pretrained',width=1024);agent.save(tmp_path)
    restored=NeuralAgent.load(tmp_path)
    torch.testing.assert_close(restored.policy(board),model(board))


def test_critic_reset_preserves_actor_predictions(tmp_path):
    torch.set_num_threads(1);model=Pretrained2048();p=tmp_path/'external.pt'
    torch.save({'policy_state':model.state_dict()},p)
    restored=Pretrained2048.from_external(p,reset_critic=True)
    b=torch.arange(16).reshape(1,16)
    torch.testing.assert_close(restored(b)[:,:4],model(b)[:,:4])
    assert restored(b)[0,4].item()==0
