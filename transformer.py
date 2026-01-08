import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads):
        super().__init__()
        # Number of heads divides embedding dimension
        assert embedding_dim % n_heads == 0

        self.d_model = embedding_dim
        self.n_heads = n_heads
        head_dim = embedding_dim // n_heads
        # Head dimensions even for RoPE
        assert head_dim % 2 == 0
        self.d_head = head_dim

        self.qkv = nn.Linear(embedding_dim, 3*embedding_dim, bias=False)
        self.WO = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def rotate_half(self, x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).reshape_as(x)

    def apply_rope(self, q, k, cos, sin):
        q_rot = (cos * q) + (self.rotate_half(q) * sin)
        k_rot = (cos * k) + (self.rotate_half(k) * sin)
        return q_rot, k_rot

    def forward(self, x):
        B, T, _ = x.shape
        # Causal Mask
        mask = torch.triu(
            torch.full((T, T), float('-inf'), device=x.device),
            diagonal=1
        )

        # sin and cos for positional encoder
        pos = torch.arange(T, device=x.device)
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, self.d_head, 2, device=x.device) / self.d_head
        )
        angles = pos[:, None] * freqs[None, :]
        sin = torch.sin(angles)[None, None, :, :]
        cos = torch.cos(angles)[None, None, :, :]

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2) # B, n_heads, T, d_head
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Positional Encoding
        q, k = self.apply_rope(q, k, cos, sin)

        scores = torch.einsum('bhqd, bhkd -> bhqk', q, k)    # B, n_heads, T, T
        scores = scores/math.sqrt(self.d_head)
        scores = scores + mask
        #softmax over key dimension
        attn = torch.softmax(scores, dim=-1)

        out = torch.einsum('bhqk, bhkd -> bhqd', attn, v)  # B, n_heads, T, d_head
        out = out.transpose(1, 2).contiguous()             # B, T, n_heads, d_head
        out = out.view(B, T, self.d_model)                 # B, T, n_heads*d_head
        out = self.WO(out)

        return out
    
    class MLP(nn.Module):
        def __init__(self, embedding_dim, expansion=4):
            super().__init__()
            hidden_dim = expansion*embedding_dim
            self.fc1 = nn.Linear(embedding_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, embedding_dim)
            self.act = nn.GELU()

        def forward(self, x):
            x = self.act(self.fc1(x))
            x = self.act(self.fc2(x))
            return x