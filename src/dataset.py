import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):

    def __init__(self, text, block_size=128, split="train"):
        self.block_size = block_size

        # Build vocabulary
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

        # Encode all text
        ids = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

        split_idx = int(0.9 * len(ids))
        if split == "train":
            self.data = ids[:split_idx]
        else:
            self.data = ids[split_idx:]

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return ''.join([self.itos[i] for i in ids])