import torch
import os
import requests
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import CharDataset
from src.model.gpt_like import GPTLanguageModel
from src.model.metrics import calculate_loss, calculate_ppl


def main():

    # load dataset
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url).text

    train_dataset = CharDataset(text, block_size=128, split="train")
    val_dataset   = CharDataset(text, block_size=128, split="val")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=64)

    # initial model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = GPTLanguageModel(train_dataset.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # TensorBoard
    writer = SummaryWriter("experiments/runs")
    os.makedirs("experiments/checkpoints", exist_ok=True)

    # train model
    max_iters = 6000
    eval_interval = 200

    step = 0

    for x, y in train_loader:
        if step > max_iters:
            break

        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("train/loss", loss.item(), step)

        if step % eval_interval == 0:
            val_loss = calculate_loss(model, val_loader, device)
            ppl = calculate_ppl(model, val_loader, device)

            print(f"step {step} | train loss {loss.item():.4f} "
                  f"| val loss {val_loss:.4f} | ppl {ppl:.2f}")

            writer.add_scalar("val/loss", val_loss, step)
            writer.add_scalar("val/ppl", ppl, step)

            # save checkpoint
            ckpt_path = f"experiments/checkpoints/step_{step}.pt"
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step},
                       ckpt_path)

            print("Saved:", ckpt_path)

        step += 1

    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    main()