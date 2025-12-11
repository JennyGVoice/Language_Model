import torch
import os
import requests
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import CharDataset
from src.model.gpt_like import GPTLanguageModel
from src.model.metrics import calculate_loss, calculate_ppl


def main():

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url).text

    train_dataset = CharDataset(text, block_size=128, split="train")
    val_dataset   = CharDataset(text, block_size=128, split="val")

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=128, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = GPTLanguageModel(train_dataset.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    writer = SummaryWriter("/content/drive/MyDrive/Language_Model/runs")

    max_iters = 5000
    eval_interval = 200

    best_val_loss = float("inf")
    step = 0

    progress_bar = tqdm(total=max_iters, desc="Training", ncols=100)

    for x, y in train_loader:
        if step >= max_iters:
            break

        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        writer.add_scalar("train/grad_norm", total_norm, step)
        optimizer.step()

        writer.add_scalar("train/loss", loss.item(), step)

        progress_bar.set_postfix({"train_loss": f"{loss.item():.4f}"})
        progress_bar.update(1)

        if step % eval_interval == 0:
            print("\nBegin evaluation")
            model.eval()
            with torch.no_grad():
                xb, yb = next(iter(val_loader))
                xb, yb = xb.to(device), yb.to(device)
                _, val_loss = model(xb, yb)
            model.train()

            ppl = torch.exp(val_loss).item()

            print(f"step {step} | train loss {loss.item():.4f} "
                  f"| val loss {val_loss:.4f} | ppl {ppl:.2f}")

            writer.add_scalar("val/loss", val_loss.item(), step)
            writer.add_scalar("val/ppl", ppl, step)

            ckpt_path = f"/content/drive/MyDrive/Language_Model/checkpoints/step_{step}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step
            }, ckpt_path)
            print("Saved:", ckpt_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = "/content/drive/MyDrive/Language_Model/checkpoints/best_model.pt"
                torch.save(model.state_dict(), best_path)
                print(f"New BEST model saved at step {step}")

        step += 1

    progress_bar.close()
    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    main()