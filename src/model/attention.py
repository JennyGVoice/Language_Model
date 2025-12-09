import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionHead(nn.Module):
    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        att = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        v = self.value(x)
        return att @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, n_embd, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [AttentionHead(head_size, n_embd, block_size) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))