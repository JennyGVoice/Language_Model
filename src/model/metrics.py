import torch
import math

@torch.no_grad()
def calculate_ppl(model, dataloader, device="cuda"):
    model.eval()
    losses = []
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return math.exp(sum(losses) / len(losses))

@torch.no_grad()
def calculate_loss(model, dataloader, device="cuda"):
    model.eval()
    losses = []
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)