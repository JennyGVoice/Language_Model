import torch
import torch.nn as nn
from .block import TransformerBlock

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd=768, block_size=128, n_head=8, n_layer=12):
        super().__init__()

        self.token_embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[TransformerBlock(n_embd, n_head, block_size) for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok = self.token_embed(idx)
        pos = self.pos_embed(torch.arange(T, device=idx.device))
        x = tok + pos

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx