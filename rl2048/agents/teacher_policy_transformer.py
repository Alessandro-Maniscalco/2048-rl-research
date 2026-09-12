"""A standard Transformer with separate move logits and a scalar value.

The eight board views share weights. Move outputs are rotated/reflected back
before averaging; the scalar value is invariant. Logits are not Q-values.
"""
import numpy as np
import torch
from torch import nn

from rl2048.agents.transformer_q import TransformerQNetwork


class TeacherPolicyTransformer(TransformerQNetwork):
    def __init__(self,width=128,input_encoding='embedding',depth=4,heads=4,
                 value_offset=0.,value_scale=1.):
        if not np.isfinite(value_scale) or value_scale<=0 or not np.isfinite(value_offset):
            raise ValueError('Finite offset and positive finite value scale required')
        super().__init__(width,input_encoding,depth,heads)
        self.architecture='transformer_policy_value';self.outputs=5
        self.readout=nn.Linear(16*width,5)
        nn.init.zeros_(self.readout.weight);nn.init.zeros_(self.readout.bias)
        cells=np.arange(16).reshape(4,4)
        self.register_buffer('symmetry_permutations',torch.tensor(np.stack(
            [np.rot90(cells,k).ravel() for k in range(4)]+
            [np.rot90(np.fliplr(cells),k).ravel() for k in range(4)])))
        self.register_buffer('action_permutations',torch.tensor(np.stack(
            [(np.arange(4)-k)%4 for k in range(4)]+
            [(np.array([0,3,2,1])-k)%4 for k in range(4)])))
        self.register_buffer('value_offset',torch.tensor(float(value_offset)))
        self.register_buffer('value_scale',torch.tensor(float(value_scale)))

    def forward(self,boards):
        views=boards.reshape(-1,16)[:,self.symmetry_permutations].reshape(-1,16)
        output=super().forward(views).reshape(-1,8,5)
        logits=output[:,:,:4].gather(2,self.action_permutations[None].expand(len(output),-1,-1)).mean(1)
        value=self.value_offset+self.value_scale*output[:,:,4].mean(1)
        return torch.cat((logits,value[:,None]),dim=1)


def copy_value_encoder(model,state):
    """Transfer only embeddings, positions and Transformer blocks; reset heads.

    New policy/value heads keep their initialization. Their targets and units
    differ from the original afterstate scalar, so its readout is not copied.
    """
    prefixes=('tokenizer.','positions','blocks.','norm.')
    expected={k for k in model.state_dict() if k.startswith(prefixes)}
    selected={k:v for k,v in state.items() if k.startswith(prefixes)}
    if set(selected)!=expected:raise ValueError('Encoder architecture mismatch')
    current=model.state_dict()
    for k,v in selected.items():
        if current[k].shape!=v.shape:raise ValueError(f'Encoder shape mismatch: {k}')
    current.update(selected);model.load_state_dict(current)
