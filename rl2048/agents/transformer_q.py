"""Sixteen cell tokens -> standard Transformer encoder -> four Q-values.

Every block includes multi-head self-attention AND a feed-forward MLP, with
residual connections and layer normalization. No handcrafted board features.
"""
import torch
from torch import nn


class TransformerQNetwork(nn.Module):
    def __init__(self, width=64, input_encoding='embedding', depth=2, heads=4):
        super().__init__()
        if input_encoding not in ('exponents', 'embedding', 'relative'):
            raise ValueError('Transformer inputs must be exponents, embedding or relative.')
        if width < 1 or heads < 1 or width % heads or depth < 1:
            raise ValueError('Positive width must be divisible by positive heads.')
        self.architecture = 'transformer_q'
        self.input_encoding, self.depth, self.heads = input_encoding, depth, heads
        self.outputs = 4
        self.tokenizer = nn.Embedding(18, width) if input_encoding == 'embedding' else nn.Linear(1, width)
        self.positions = nn.Parameter(torch.empty(1, 16, width))
        nn.init.normal_(self.positions, std=.02)
        # Construct separately: cloned TransformerEncoder layers otherwise start
        # with identical parameters. Dropout=0 keeps target inference deterministic.
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(width, heads, dim_feedforward=4*width,
                dropout=0., activation='gelu', batch_first=True, norm_first=True)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.readout = nn.Linear(16*width, 4)

    def encode_inputs(self, boards):
        """Convert stored exponents into the values seen by each cell token.

        Relative inputs divide actual tile values by the board maximum. Empty
        cells stay zero; an empty board is all zero. This deliberately discards
        absolute scale, including information about the fixed 2/4 spawn sizes.
        """
        b = boards.reshape(-1, 16)
        if self.input_encoding == 'embedding':
            return b.long().clamp(0, 17)
        if self.input_encoding == 'relative':
            ranks = b.float()
            # 2**(rank-max_rank) equals tile/max_tile without large intermediates.
            values = torch.where(b > 0, torch.exp2(ranks-ranks.amax(dim=1,keepdim=True)), 0.)
            return values.unsqueeze(-1)
        return b.float().unsqueeze(-1)/16

    def forward(self, boards):
        x = self.tokenizer(self.encode_inputs(boards))
        x = x + self.positions
        for block in self.blocks:
            x = block(x)
        return self.readout(self.norm(x).flatten(1))

    def forward_with_entropy(self, boards):
        """Same network, exposing attention entropy for diagnosis/regularization.

        Following TQL (Dong et al., 2026), average attention across heads BEFORE
        computing entropy. Our discrete four-Q adaptation has no VALUE/action
        tokens and no offline flow-policy extraction.
        """
        x=self.tokenizer(self.encode_inputs(boards))
        x=x+self.positions
        entropies=[]
        for block in self.blocks:
            norm=block.norm1(x)
            attended,weights=block.self_attn(norm,norm,norm,need_weights=True,average_attn_weights=True)
            x=x+block.dropout1(attended)
            norm=block.norm2(x)
            x=x+block.dropout2(block.linear2(block.dropout(block.activation(block.linear1(norm)))))
            entropies.append(-(weights*weights.clamp_min(1e-8).log()).sum(-1).mean())
        return self.readout(self.norm(x).flatten(1)),torch.stack(entropies)
