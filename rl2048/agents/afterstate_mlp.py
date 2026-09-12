"""A neural value of the board AFTER a slide and BEFORE the random spawn.

U(x) predicts the future rewards starting at the next move, in points / 128.
Thus Q(s,a) = r(s,a)/128 + gamma * U(slide(s,a)). The simulator supplies the
slide and immediate reward; no teacher, strategic features or corner rule.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rl2048.afterstate_compare import moves
from rl2048.agents.afterstate_cnn import AfterstateCNN
from rl2048.agents.afterstate_transformer import SymmetricAfterstateTransformer
from rl2048.agents.mlp_q import MLPQNetwork
from rl2048.agents.neural import optimize, soft_update, tensor_boards
from rl2048.agents.ntuple import encode, row_tables


class AfterstateMLP(MLPQNetwork):
    def __init__(self, width=512, input_encoding='exponents', depth=2, embedding_dim=16):
        if input_encoding not in ('exponents', 'embedding', 'one_hot'):
            raise ValueError('Only tile inputs, no engineered relational features')
        super().__init__(width, input_encoding, depth, embedding_dim)
        self.architecture = 'afterstate_mlp'
        self.layers[-1] = nn.Linear(width, 1)
        # Initially, action selection uses exact merge points alone.
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, boards):
        return super().forward(boards).squeeze(-1)


class SymmetricAfterstateMLP(AfterstateMLP):
    """One shared MLP, averaged over the eight exact board symmetries.

Both predictions and TD targets use the average, and gradients pass through
all eight views. Parameter count is unchanged; inference/training cost grows.
"""
    def __init__(self, width=512, input_encoding='embedding', depth=2, embedding_dim=16):
        super().__init__(width,input_encoding,depth,embedding_dim)
        self.architecture='afterstate_sym_mlp'
        cells=np.arange(16).reshape(4,4)
        self.register_buffer('symmetry_permutations',torch.tensor(np.stack(
            [np.rot90(cells,k).ravel() for k in range(4)]+
            [np.rot90(np.fliplr(cells),k).ravel() for k in range(4)])))

    def forward(self,boards):
        views=boards.reshape(-1,16)[:,self.symmetry_permutations].reshape(-1,16)
        return super().forward(views).reshape(-1,8).mean(1)


def afterstate_targets(gains, legal, online_values, target_values, terminated, gamma):
    """Target for x_t uses NEXT move's reward, not the reward already earned.

    The actual spawn s_(t+1) is sampled by the environment. Enumerate its four
    deterministic slides, choose with online values, evaluate with target values.
    Natural terminal rows are zero; a time limit still bootstraps.
    """
    online_q = gains / 128 + gamma * online_values
    actions = online_q.masked_fill(~legal, -torch.inf).argmax(1)
    index = torch.arange(len(actions), device=actions.device)
    target = gains[index, actions] / 128 + gamma * target_values[index, actions]
    return torch.where(terminated | ~legal.any(1), torch.zeros_like(target), target)


class AfterstateLearner:
    def __init__(self, device='cpu', width=512, input_encoding='exponents', depth=2,
                 embedding_dim=16, gamma=1., lr=1e-4, target_tau=.005,
                 weight_decay=.01, spawn_target='sampled', target_batch=8192,
                 architecture='afterstate_mlp', heads=4, value_scale=1., value_offset=0., **_):
        if spawn_target not in ('sampled','expected') or target_batch < 1:
            raise ValueError('Choose sampled/expected spawn targets and a positive inference chunk')
        self.device, self.gamma, self.target_tau = device, gamma, target_tau
        self.spawn_target, self.target_batch = spawn_target, target_batch
        models = {'afterstate_mlp': AfterstateMLP, 'afterstate_cnn': AfterstateCNN,
                  'afterstate_sym_mlp': SymmetricAfterstateMLP,
                  'afterstate_sym_transformer': SymmetricAfterstateTransformer}
        if architecture not in models:
            raise ValueError('Choose a supported afterstate network')
        extra=(dict(heads=heads,value_scale=value_scale,value_offset=value_offset)
               if architecture=='afterstate_sym_transformer' else {})
        self.policy = models[architecture](width, input_encoding, depth, embedding_dim,**extra).to(device)
        self.target = deepcopy(self.policy).requires_grad_(False)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=weight_decay)

    @torch.no_grad()
    def decision(self, boards):
        after, gains, legal = moves(boards, *row_tables())
        values = self.policy(tensor_boards(after.reshape(-1, 16), self.device)).cpu().numpy().reshape(-1, 4)
        return np.where(legal, gains / 128 + self.gamma * values, -np.inf), after, legal

    @torch.no_grad()
    def expected_targets(self, afterstates):
        from rl2048.afterstate_expectation import spawn_outcomes, weighted_spawn_values
        states, owners, probabilities = spawn_outcomes(afterstates)
        after, gains, legal = moves(states, *row_tables())
        leaves = after.reshape(-1,16)
        online, target = [], []
        # Bound temporary activations without changing which chance outcomes
        # are included. All chunks use the same frozen-for-this-update weights.
        for start in range(0,len(leaves),self.target_batch):
            x = tensor_boards(leaves[start:start+self.target_batch],self.device)
            online.append(self.policy(x)); target.append(self.target(x))
        branch_values = afterstate_targets(
            torch.as_tensor(gains,device=self.device),
            torch.as_tensor(legal,device=self.device),
            torch.cat(online).reshape(-1,4),torch.cat(target).reshape(-1,4),
            torch.as_tensor(~legal.any(1),device=self.device),self.gamma)
        expected = weighted_spawn_values(branch_values,
            torch.as_tensor(owners,device=self.device),
            torch.as_tensor(probabilities,device=self.device),len(afterstates))
        return expected, len(states), len(leaves)

    def update(self, afterstates, next_states, terminated):
        counts = {}
        if self.spawn_target == 'expected':
            # The recorded spawn still happened in the real game. Here its
            # sampled label is replaced by the exact known chance expectation.
            target, outcomes, slides = self.expected_targets(afterstates)
            counts = dict(simulated_spawn_outcomes=outcomes, simulated_next_slides=slides)
        else:
            after, gains, legal = moves(next_states, *row_tables())
            x = tensor_boards(after.reshape(-1, 16), self.device)
            with torch.no_grad():
                target = afterstate_targets(
                    torch.as_tensor(gains, device=self.device),
                    torch.as_tensor(legal, device=self.device),
                    self.policy(x).reshape(-1, 4), self.target(x).reshape(-1, 4),
                    torch.as_tensor(terminated, device=self.device), self.gamma)
        predicted = self.policy(tensor_boards(afterstates, self.device))
        loss = (predicted - target).square().mean()
        optimize(self.optimizer, loss, self.policy.parameters())
        soft_update(self.target, self.policy, self.target_tau)
        return dict(value_mse=float(loss.detach().cpu()),
                    mean_value=float(predicted.detach().mean().cpu()),
                    mean_abs_td_error=float((predicted.detach()-target).abs().mean().cpu()), **counts)


class AfterstateMLPAgent:
    name = 'neural_afterstate'

    def __init__(self, learner):
        self.learner = learner
        network = {'afterstate_cnn':'CNN', 'afterstate_mlp':'MLP',
                   'afterstate_sym_mlp':'MLP with eight-view value sharing',
                   'afterstate_sym_transformer':'Transformer with eight-view value sharing'}[learner.policy.architecture]
        self.display_name = f'Neural afterstate {network} · exact slide + learned future value'
        self.rng = np.random.default_rng(0)

    def act(self, board, action_mask):
        q, _, legal = self.learner.decision(encode(board)[None])
        if not np.array_equal(legal[0], action_mask):
            raise ValueError('Game and afterstate action masks disagree')
        return int(q[0].argmax())

    def save(self, path, experiment):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        model = self.learner.policy
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path/'policy.pt.tmp')
        (path/'policy.pt.tmp').replace(path/'policy.pt')
        metadata = dict(architecture=model.architecture, width=experiment['width'],
                        input_encoding=model.input_encoding, depth=model.depth,
                        embedding_dim=model.embedding_dim, gamma=self.learner.gamma,
                        outputs=1, parameter_count=sum(p.numel() for p in model.parameters()),
                        experiment=experiment)
        if model.architecture=='afterstate_sym_transformer':
            metadata.update(heads=model.heads,value_scale=float(model.value_scale.cpu()),
                            value_offset=float(model.value_offset.cpu()))
        (path/'metadata.json').write_text(json.dumps(metadata, indent=2))

    @classmethod
    def load(cls, path, device='cpu'):
        path = Path(path); metadata = json.loads((path/'metadata.json').read_text())
        learner = AfterstateLearner(device=device, **metadata)
        learner.policy.load_state_dict(torch.load(path/'policy.pt', map_location='cpu', weights_only=True))
        learner.target.load_state_dict(learner.policy.state_dict())
        learner.policy.eval()
        return cls(learner)
