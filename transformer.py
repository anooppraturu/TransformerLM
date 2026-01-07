import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads):
        super().__init__()

        assert embedding_dim % n_heads == 0

        self.d_model = embedding_dim
        self.n_heads = n_heads
        head_dim = embedding_dim // n_heads
        self.d_head = head_dim

        self.qkv = nn.Linear(embedding_dim, 3*embedding_dim, bias=False)
        self.WO = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        mask = torch.triu(
            torch.full((T, T), float('-inf'), device=x.device),
            diagonal=1
        )

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2) # B, n_heads, T, d_head
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

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