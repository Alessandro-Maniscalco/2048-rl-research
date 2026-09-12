"""Only shared neural plumbing: current board -> vector, masks, save/load.

Networks receive tile boards with no score history. Architectures define their
tile encoding. An optional, explicitly saved policy filter uses exact game
dynamics to restrict actions; the network itself predicts current-board outputs.
Algorithms live in separate modules so their learning updates remain visible.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from rl2048.agents.ntuple import encode


def tensor_boards(boards, device):
    return torch.as_tensor(boards, dtype=torch.long, device=device)


def policy_logits(model, boards):
    """Four move preferences, optionally skipping an unused value head."""
    if hasattr(model, 'forward_policy'):
        return model.forward_policy(boards)
    return model(boards)[:, :4]


class BoardNet(nn.Module):
    def __init__(self, outputs, width=256, architecture="mlp"):
        super().__init__()
        self.architecture = architecture
        if architecture == "cnn":
            self.encoder = nn.Sequential(nn.Conv2d(18, 128, 2), nn.ReLU(),
                                         nn.Conv2d(128, 128, 2), nn.ReLU(), nn.Flatten())
            inputs = 128 * 2 * 2
        elif architecture == "mlp":
            self.encoder = nn.Flatten()
            inputs = 16 * 18
        else:
            raise ValueError("Architecture must be mlp or cnn.")
        self.layers = nn.Sequential(nn.Linear(inputs, width), nn.ReLU(),
                                    nn.Linear(width, width), nn.ReLU(), nn.Linear(width, outputs))

    def forward(self, boards):
        x = F.one_hot(boards.long().clamp(0, 17), num_classes=18).float()
        if self.architecture == "cnn":
            x = x.reshape(-1, 4, 4, 18).permute(0, 3, 1, 2)
        return self.layers(self.encoder(x))


def masked_log_probs(logits, masks):
    # Terminal rows have no legal actions; use a harmless distribution, which
    # must be multiplied by zero in terminal Bellman targets.
    safe = masks | ~masks.any(dim=-1, keepdim=True)
    return F.log_softmax(logits.masked_fill(~safe, -1e9), dim=-1)


def sample_actions(probabilities, rng):
    """Categorical samples without selecting zero-probability (illegal) actions."""
    cumulative = np.cumsum(probabilities.astype(np.float64), axis=1)
    thresholds = rng.random(len(probabilities)) * cumulative[:, -1]
    return (thresholds[:, None] >= cumulative).sum(axis=1)


def chosen(values, actions):
    return values.gather(1, actions.long().reshape(-1, 1)).squeeze(1)


def optimize(optimizer, loss, parameters, max_norm=10.0):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(parameters, max_norm)
    optimizer.step()


@torch.no_grad()
def soft_update(target, source, tau=.005):
    for destination, origin in zip(target.parameters(), source.parameters(), strict=True):
        destination.lerp_(origin, tau)


class NeuralAgent:
    name = 'neural_current_state'
    def __init__(self, policy, algorithm, device='cpu', width=256, spawn_safety=False):
        self.policy = policy.to(device)
        self.algorithm, self.device, self.width = algorithm, device, width
        self.spawn_safety = spawn_safety
        self.display_name = algorithm.upper() + ' · current-state neural policy'
        if algorithm == 'fitted_lookahead_q':
            self.display_name = 'MLP trained on lookahead Q targets'
        self.rng = np.random.default_rng(0)
        if spawn_safety:
            self.display_name += ' + exact next-spawn risk filter'

    def policy_mask(self, boards, legal):
        """Optional restricted policy support; the environment's legality is unchanged."""
        if not self.spawn_safety:
            return legal
        from rl2048.agents.spawn_safety import minimum_risk_mask
        return minimum_risk_mask(boards, legal)

    @torch.no_grad()
    def act(self, board, action_mask):
        boards = encode(board)[None]
        logits = policy_logits(self.policy, tensor_boards(boards, self.device))
        mask = torch.as_tensor(self.policy_mask(boards, action_mask[None]), device=self.device)
        return int(logits.masked_fill(~mask, -1e9).argmax(1).item())

    def save(self, path, metadata=None):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({key: val.detach().cpu() for key, val in self.policy.state_dict().items()}, path / 'policy.pt.tmp')
        (path / 'policy.pt.tmp').replace(path / 'policy.pt')
        (path / 'metadata.json').write_text(json.dumps({'agent': self.name, 'algorithm': self.algorithm,
            'width': self.width, 'spawn_safety': self.spawn_safety, 'architecture': self.policy.architecture, 'input_encoding': getattr(self.policy,'input_encoding','one_hot'), 'outputs': self.policy.outputs if hasattr(self.policy,'outputs') else self.policy.layers[-1].out_features,
            'raw_divisor':getattr(self.policy,'raw_divisor',65536.),
            'depth':getattr(self.policy,'depth',2), 'embedding_dim':getattr(self.policy,'embedding_dim',16),
            'residual':getattr(self.policy,'residual',False), 'heads':getattr(self.policy,'heads',4),
            'num_quantiles':getattr(self.policy,'num_quantiles',None),
            'parameter_count':sum(p.numel() for p in self.policy.parameters()),
            'experiment': metadata or {}}, indent=2))

    @classmethod
    def load(cls, path, device='cpu'):
        path = Path(path)
        meta = json.loads((path / 'metadata.json').read_text())
        if meta.get('architecture') == 'transformer_qr':
            from rl2048.agents.qr_dqn import QuantileTransformer
            model=QuantileTransformer(meta['width'],meta['input_encoding'],meta['depth'],meta['heads'],meta['num_quantiles'])
        elif meta.get('architecture') == 'mlp_qr':
            from rl2048.agents.qr_mlp import QuantileMLP
            model=QuantileMLP(meta['width'],meta['input_encoding'],meta['depth'],
                meta['embedding_dim'],meta['residual'],meta['num_quantiles'])
        elif meta.get('architecture') == 'transformer_policy':
            from rl2048.agents.reinforce import PolicyTransformer
            model=PolicyTransformer(meta['width'],meta['input_encoding'],meta['depth'],meta['heads'])
        elif meta.get('architecture') == 'transformer_policy_value':
            from rl2048.agents.teacher_policy_transformer import TeacherPolicyTransformer
            model=TeacherPolicyTransformer(meta['width'],meta['input_encoding'],meta['depth'],meta['heads'])
        elif meta.get('architecture') == 'transformer_q':
            from rl2048.agents.transformer_q import TransformerQNetwork
            model=TransformerQNetwork(meta['width'],meta['input_encoding'],meta['depth'],meta['heads'])
        elif meta.get('architecture') == 'cnn_policy18':
            from rl2048.agents.native_policy import NativePolicy
            model=NativePolicy()
        elif meta.get('architecture') == 'pretrained_ml2048':
            from rl2048.agents.pretrained2048 import Pretrained2048
            model=Pretrained2048()
        elif meta.get('architecture') == 'mlp_q':
            from rl2048.agents.mlp_q import MLPQNetwork
            model = MLPQNetwork(meta['width'],meta['input_encoding'],meta['depth'],meta['embedding_dim'],meta['residual'])
        elif meta.get('architecture') in ('scalar','scalar_dueling'):
            from rl2048.agents.dqn import QNetwork
            network = QNetwork
            if meta['architecture'] == 'scalar_dueling':
                from rl2048.agents.dueling_dqn import DuelingQNetwork
                network = DuelingQNetwork
            model = network(meta['width'], meta['input_encoding'],meta.get('raw_divisor',65536.))
        else:
            model = BoardNet(meta['outputs'], meta['width'], meta.get('architecture','mlp'))
        model.load_state_dict(torch.load(path / 'policy.pt', map_location='cpu', weights_only=True))
        return cls(model.eval(), meta['algorithm'], device, meta['width'], meta.get('spawn_safety', False))
