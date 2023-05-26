import math

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F


@gin.configurable('MLP')
class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, depth=1, activation='relu', do=0.0, ln=True):
        super().__init__()
        embedding_layers = []
        if activation == 'relu':
            activation_fn = nn.ReLU
        else:
            raise Exception("Activation has to be relu")
        for k in range(depth):
            if k == 0:
                embedding_layers.append(nn.Linear(input_dim, hidden_dim))
                if depth > 1:
                    if ln:
                        embedding_layers.append(nn.LayerNorm(hidden_dim))
                    embedding_layers.append(activation_fn())
                    embedding_layers.append(nn.Dropout(do))
            else:
                embedding_layers.append(nn.Linear(hidden_dim, hidden_dim))
                if k < depth - 1:
                    if ln:
                        embedding_layers.append(nn.LayerNorm(hidden_dim))
                    embedding_layers.append(activation_fn())
                    embedding_layers.append(nn.Dropout(do))
        self.embedding_layer = nn.Sequential(*embedding_layers)

    def forward(self, x):
        return self.embedding_layer(x)


class PositionalEncoding(nn.Module):
    """Positiona Encoding, mostly from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""

    def __init__(self, emb, max_len=3000):
        super().__init__()
        pe = torch.zeros(max_len, emb)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb, 2).float() * (-math.log(10000.0) / emb))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        bs, n, emb = x.size()
        return x + self.pe[:, :n, :]


class SelfAttentionSimple(nn.Module):
    def __init__(self, emb, mask=True):
        super().__init__()
        self.emb = emb
        self.mask = mask

    def forward(self, x):
        emb = self.emb
        queries = x[1].permute(1, 0, 2)  # (bs, 1, emb)
        keys = x[0].transpose(1, 2)
        values = x[0]

        queries = queries / (emb ** (1 / 2))
        keys = keys / (emb ** (1 / 2))
        dot = torch.bmm(queries, keys)
        dot = F.softmax(dot, dim=2)
        out = torch.bmm(dot, values)
        return out.squeeze()


class SelfAttention(nn.Module):
    """Multi Head Attention block from Attention is All You Need.
    Input has shape (batch_size, n_timestemps, emb).

    ----------
    emb:
        Dimension of the input vector.
    hidden:
        Dimension of query, key, value matrixes.
    heads:
        Number of heads.

    mask:
        Mask the future timestemps
    """

    def __init__(self, emb, hidden, heads=8, mask=True,
                 dropout_att=0.0):
        """Initialize the Multi Head Block."""
        super().__init__()

        self.emb = emb
        self.heads = heads
        self.hidden = hidden
        self.mask = mask
        self.drop_att = nn.Dropout(dropout_att)

        # Query, keys and value matrices
        self.w_keys = nn.Linear(emb, hidden * heads, bias=False)
        self.w_queries = nn.Linear(emb, hidden * heads, bias=False)
        self.w_values = nn.Linear(emb, hidden * heads, bias=False)

        # Output linear function
        self.unifyheads = nn.Linear(heads * hidden, emb)

    def forward(self, x):
        """
        x:
            Input data tensor with shape (batch_size, n_timestemps, emb)
        hidden:
            Hidden dim (dimension of query, key, value matrixes)

        Returns
            Self attention tensor with shape (batch_size, n_timestemps, emb)
        """
        # bs - batch_size, n - vectors number, emb - embedding dimensionality
        bs, n, emb = x.size()
        h = self.heads
        hidden = self.hidden

        keys = self.w_keys(x).view(bs, n, h, hidden)
        queries = self.w_queries(x).view(bs, n, h, hidden)
        values = self.w_values(x).view(bs, n, h, hidden)

        # fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(bs * h, n, hidden)
        queries = queries.transpose(1, 2).contiguous().view(bs * h, n, hidden)
        values = values.transpose(1, 2).contiguous().view(bs * h, n, hidden)

        # dive on the square oot of dimensionality
        queries = queries / (hidden ** (1 / 2))
        keys = keys / (hidden ** (1 / 2))

        # dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        if self.mask:  # We deal with different masking and recombination types here
            mask_tensor = torch.ones(n, n, device=dot.device).tril()
            dot = torch.where(mask_tensor.bool(),
                              dot,
                              torch.tensor(float('-inf')).cuda()).view(bs * h, n, n)

        # dot now has row-wise self-attention probabilities
        dot = F.softmax(dot, dim=2)

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(bs, h, n, hidden)

        # apply the dropout
        out = self.drop_att(out)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(bs, n, h * hidden)
        return self.unifyheads(out)


class TransformerBlock(nn.Module):

    def __init__(self, emb, hidden, heads, ff_hidden_mult, dropout=0.0, mask=True, dropout_att=0.0):
        super().__init__()

        self.attention = SelfAttention(emb, hidden, heads=heads, mask=mask, dropout_att=dropout_att)
        self.mask = mask
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention.forward(x)
        x = self.norm1(attended + x)
        x = self.drop(x)
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)

        x = self.drop(x)

        return x