import torch
from torch import nn
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads, max_context_length):
        super().__init__()
        # Number of heads divides embedding dimension
        assert embedding_dim % n_heads == 0

        self.log_attention = False
        self.last_attention = None

        self.d_model = embedding_dim
        self.n_heads = n_heads
        head_dim = embedding_dim // n_heads
        # Head dimensions even for RoPE
        assert head_dim % 2 == 0
        self.d_head = head_dim

        self.qkv = nn.Linear(embedding_dim, 3*embedding_dim, bias=False)
        self.WO = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Buffers for causal masking and RoPE trig functions
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((max_context_length, max_context_length), float('-inf')), diagonal=1)
        )

        pos = torch.arange(max_context_length)
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, self.d_head, 2) / self.d_head
        )
        angles = pos[:, None] * freqs[None, :]

        sin = torch.sin(angles)
        cos = torch.cos(angles)

        self.register_buffer("rope_sin", torch.repeat_interleave(sin, 2, dim=-1))
        self.register_buffer("rope_cos", torch.repeat_interleave(cos, 2, dim=-1))

    def rotate_half(self, x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).reshape_as(x)

    def apply_rope(self, q, k, cos, sin):
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        return q_rot, k_rot
    
    def enable_attention_logging(self):
        self.log_attention = True

    def disable_attention_logging(self):
        self.log_attention = False
        self.last_attention = None

    def forward(self, x):
        B, T, _ = x.shape
        assert T <= self.causal_mask.size(0), "Sequence length exceeds model context length"

        # Masking and positional encoding variables
        mask = self.causal_mask[:T, :T]
        sin = self.rope_sin[:T][None, None, :, :]
        cos = self.rope_cos[:T][None, None, :, :]

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

        if self.log_attention:
            self.last_attention = attn.detach()

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
    def __init__(self, embedding_dim, n_heads, context_length):
        super().__init__()

        assert embedding_dim % n_heads == 0

        self.d_model = embedding_dim
        self.n_heads = n_heads

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attn = MultiHeadSelfAttention(embedding_dim, n_heads, context_length)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x
    
class TransformerLM(nn.Module):
    def __init__(self, embedding_dim, depth, n_heads, vocab_size, context_length):
        super().__init__()

        assert embedding_dim % n_heads == 0

        self.dim = embedding_dim
        self.depth = depth
        self.n_heads = n_heads
        self.T = context_length

        self.tok_emb = nn.Embedding(vocab_size, embedding_dim)

        self.blocks = nn.ModuleList(
            [TransformerBlock(embedding_dim, n_heads, self.T) for _ in range(depth)]
        )

        self.ln_f = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def enable_attention_logging(self):
        for block in self.blocks:
            block.attn.enable_attention_logging()

    def disable_attention_logging(self):
        for block in self.blocks:
            block.attn.disable_attention_logging()

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
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature = 1.0, do_sample = True):
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.T:]  # truncate to context length
            logits = self(idx_cond)      # (B, T, vocab)
            logits = logits[:, -1, :] / temperature

            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_idx = torch.multinomial(probs, num_samples=1)
            else:
                next_idx = torch.argmax(logits, dim=-1, keepdim=True)

            idx = torch.cat([idx, next_idx], dim=1)

        return idx