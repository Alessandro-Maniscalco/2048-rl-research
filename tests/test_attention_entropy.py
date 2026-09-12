import math
import torch
import pytest
from rl2048.agents.transformer_q import TransformerQNetwork
from rl2048.agents.attention_entropy import AttentionEntropyControl
from rl2048.agents.dqn import DQN
from rl2048.offline_train import training_state


def test_exposing_attention_preserves_q_values_and_entropy_bounds():
    torch.manual_seed(4)
    model=TransformerQNetwork(width=32,input_encoding='exponents',heads=4)
    b=torch.randint(0,12,(8,16))
    q,h=model.forward_with_entropy(b)
    torch.testing.assert_close(q,model(b),atol=1e-6,rtol=1e-5)
    assert h.shape==(2,) and (h>=0).all() and (h<=math.log(16)+1e-6).all()
    h.sum().backward()
    assert model.blocks[0].self_attn.in_proj_weight.grad.abs().sum()>0


def test_entropy_temperature_sign_and_gradient_separation():
    c=AttentionEntropyControl(2)
    entropy=torch.tensor([0.,math.log(16)],requires_grad=True)
    actor,temp=c.losses(entropy)
    temp.backward()
    assert entropy.grad is None
    assert c.log_alpha.grad[0]<0 # gradient descent raises alpha for collapsed attention
    assert c.log_alpha.grad[1]>0 # lowers alpha above target
    c.zero_grad();actor.backward()
    assert c.log_alpha.grad is None
    assert (entropy.grad<0).all() # gradient descent encourages entropy


def test_entropy_update_and_resume_all_learned_state(tmp_path):
    args=dict(architecture='transformer_q',input_encoding='exponents',width=16,heads=4,attention_entropy=.01,weight_decay=.01)
    a=DQN(**args);b=torch.randint(0,12,(8,16))
    batch=dict(states=b,next_states=b,actions=torch.arange(8)%4,rewards=torch.ones(8),
        terminated=torch.zeros(8,dtype=torch.bool),next_masks=torch.ones(8,4,dtype=torch.bool))
    metrics=a.update(batch)
    assert all(torch.isfinite(v) for v in metrics.values())
    state=training_state(a)
    assert 'entropy_control' in state and 'entropy_optimizer' in state
    torch.save(state,tmp_path/'training.pt')
    other=DQN(**args)
    for k,v in torch.load(tmp_path/'training.pt',weights_only=True).items():getattr(other,k).load_state_dict(v)
    torch.testing.assert_close(other.entropy_control.log_alpha,a.entropy_control.log_alpha)
    torch.testing.assert_close(other.policy(b),a.policy(b))
