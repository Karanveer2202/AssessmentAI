import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import math



class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        return self.weight * (x - mu) / (var + self.eps) + self.bias


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, attention_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.attention_dropout = attention_dropout

        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, hidden_size)

        self.attention_layer_norm = LayerNorm(hidden_size)

    def forward(self, q, k, v, mask=None):
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        q = q.view(q.size(0), q.size(1), self.num_heads, q.size(2) // self.num_heads)
        k = k.view(k.size(0), k.size(1), self.num_heads, k.size(2) // self.num_heads)
        v = v.view(v.size(0), v.size(1), self.num_heads, v.size(2) // self.num_heads)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(k.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, self.attention_dropout)

        out = torch.matmul(weights, v)
        out = out.view(q.size(0), q.size(1), self.hidden_size)
        out = self.out_linear(out)
        out = self.attention_layer_norm(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, attention_dropout=0.1, feed_forward_dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, attention_dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(feed_forward_dropout)
        )
        self.layer_norm_1 = LayerNorm(hidden_size)
        self.layer_norm_2 = LayerNorm(hidden_size)

    def forward(self, x, mask=None):
        out = self.attention(x, x, x, mask)
        out = self.layer_norm_1(out + x)
        out = self.feed_forward(out)
        out = self.layer_norm_2(out + x)
        return out


class GPT2(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, vocab_size, max_seq_len, attention_dropout=0.1, feed_forward_dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_embedding = nn.Embedding(max_seq_len, hidden_size)

        self.transformer_blocks = nn.Sequential(*[TransformerBlock(hidden_size, num_heads, attention_dropout, feed_forward_dropout) for _ in range(num_layers)])

        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, mask=None):
        out = self.token_embedding(x) + self.positional_embedding(torch.arange(x.size(1), device=x.device))
        out = self.transformer_blocks(out, mask)
        out = self.linear(out)
        return out


model = GPT2(12, 768, 12, 50257, 1024)