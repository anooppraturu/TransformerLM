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
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
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
        sin = torch.repeat_interleave(sin, 2, dim=-1)
        cos = torch.repeat_interleave(cos, 2, dim=-1)

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
        return self.fc2(self.act(self.fc1(x)))
    
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, n_heads):
        super().__init__()

        assert embedding_dim % n_heads == 0

        self.d_model = embedding_dim
        self.n_heads = n_heads

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attn = MultiHeadSelfAttention(embedding_dim, n_heads)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x
    
class TransformerLM(nn.Module):
    def __init__(self, context_length, embedding_dim, depth, n_heads, vocab_size):
        super().__init__()

        assert embedding_dim % n_heads == 0

        self.T = context_length
        self.dim = embedding_dim
        self.n_heads = n_heads
        self.depth = depth

        self.tok_emb = nn.Embedding(vocab_size, embedding_dim)

        self.blocks = nn.ModuleList(
            [TransformerBlock(embedding_dim, n_heads) for _ in range(depth)]
        )

        self.ln_f = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        # get embedding of token
        x = self.tok_emb(x)
        # apply [depth] transformer blocks
        for block in self.blocks:
            x = block(x)
        # layer norm
        x = self.ln_f(x)
        # decode to vocab size to get logits
        logits = self.lm_head(x)

        return logits