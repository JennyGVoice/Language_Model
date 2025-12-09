import torch
from torch.utils.data import DataLoader

from src.dataset import CharDataset
from src.model.gpt_like import GPTLanguageModel
from src.model.metrics import calculate_ppl

def evaluate(checkpoint_path="experiments/checkpoints/model.pt"):
    text = open("data/input.txt").read()

    val_dataset = CharDataset(text, block_size=128, split="val")
    val_loader  = DataLoader(val_dataset, batch_size=64)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPTLanguageModel(val_dataset.vocab_size).to(device)

    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt["model"])

    ppl = calculate_ppl(model, val_loader)
    print("Validation PPL:", ppl)