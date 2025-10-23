import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import ToyDataset
from model import Transformer
from optimizer import NoamScheduler

device = torch.accelerator.current_accelerator()


def create_masks(src, tgt, pad_idx):
    # Encoder mask: 1 for valid tokens
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)  # (N,1,1,T)
    # Decoder mask: 1 for valid + causal
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    T = tgt.size(1)
    causal_mask = torch.tril(torch.ones(T, T, device=tgt.device)).bool()
    tgt_mask = tgt_pad_mask & causal_mask
    return src_mask, tgt_mask


def train_one_epoch(model, dataloader, optimizer, scheduler, pad_idx, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc='Training', leave=False, dynamic_ncols=True)
    for src, tgt in progress_bar:
        src, tgt = src.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]  # teacher forcing
        src_mask, tgt_mask = create_masks(src, tgt_in, pad_idx)

        logits = model(src, tgt_in, src_mask, tgt_mask)  # (N, T, vocab)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1),
            ignore_index=pad_idx,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    progress_bar.close()
    print()

    avg_loss = total_loss / len(dataloader)
    ppl = np.exp(avg_loss)
    return avg_loss, ppl


model = Transformer(
    n_src_vocab=20,
    n_tgt_vocab=20,
    n_positions=64,
    n_encoder_layers=2,
    n_decoder_layers=2,
    d_model=128,
    n_heads=4,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
scheduler = NoamScheduler(optimizer, d_model=128, warmup_steps=200)
dataset = ToyDataset(num_samples=10000)
print(len(dataset))  # 10000
x, y = dataset[0]
print(x.shape, y.shape)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(1, 11):
    tqdm.write(f"\n[Epoch {epoch:02d}] ------------------------------")
    start = time.time()
    avg_loss, ppl = train_one_epoch(model, train_loader, optimizer, scheduler, pad_idx=0, device=device)
    end = time.time()
    lr = scheduler.get_last_lr()[0]
    tqdm.write(f"Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | LR: {lr:.2e} | Time: {end - start:.2f}s")
