import torch
import requests
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

from src.dataset import CharDataset
from src.model.gpt_like import GPTLanguageModel
from src.model.metrics import calculate_ppl, calculate_loss


# load data
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text

train_dataset = CharDataset(text, block_size=128, split="train")
val_dataset   = CharDataset(text, block_size=128, split="val")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=64)


# initial model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPTLanguageModel(train_dataset.vocab_size).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# tensorboard setup
log_dir = "experiments/runs"
writer = SummaryWriter(log_dir)
print(f"TensorBoard logs saved to: {log_dir}")

# save checkpoints
ckpt_dir = "experiments/checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)
print(f"Checkpoints will be saved to: {ckpt_dir}")

# train model
max_iters = 6000
eval_interval = 200
step_counter = 0

for batch in train_loader:

    if step_counter > max_iters:
        break

    x, y = batch
    x, y = x.to(device), y.to(device)

    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # TensorBoard: training loss
    writer.add_scalar("train/loss", loss.item(), step_counter)

    # evaluation
    if step_counter % eval_interval == 0:
        val_loss = calculate_loss(model, val_loader, device)
        ppl = calculate_ppl(model, val_loader, device)

        print(f"step {step_counter} | train loss {loss.item():.4f} | "
              f"val loss {val_loss:.4f} | ppl {ppl:.2f}")

        # TensorBoard: validation metrics
        writer.add_scalar("val/loss", val_loss, step_counter)
        writer.add_scalar("val/ppl", ppl, step_counter)

        # TensorBoard: generated text sample
        start = train_dataset.encode("O God, O God!")
        idx = torch.tensor([start]).to(device)
        generated = model.generate(idx, max_new_tokens=200)
        text_sample = train_dataset.decode(generated[0].tolist())
        writer.add_text("generated/sample", text_sample, step_counter)

        # save checkpoint
        ckpt_path = f"{ckpt_dir}/step_{step_counter}.pt"
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step_counter
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    step_counter += 1

writer.close()
print("Training finished.")