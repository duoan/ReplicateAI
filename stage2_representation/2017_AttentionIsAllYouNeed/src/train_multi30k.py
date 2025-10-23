import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import Multi30KDataset
from model import Transformer
from optimizer import NoamScheduler

torch.autograd.set_detect_anomaly(True)
device = torch.accelerator.current_accelerator()


def train(model, dataloader, optimizer, scheduler, pad_idx, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc='Training', leave=False, dynamic_ncols=True)
    for batch in progress_bar:
        src, tgt = batch["src"], batch["tgt"]
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]  # teacher forcing
        # Forward
        logits = model(src, tgt_in)  # (N, T, vocab)

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

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    progress_bar.close()
    print()

    avg_loss = total_loss / len(dataloader)
    ppl = np.exp(avg_loss)
    return avg_loss, ppl


def evaluate(model, dataloader, pad_idx, device):
    model.eval()
    total_loss = 0
    total_bleu = []

    with torch.no_grad():
        progress = tqdm(dataloader, desc='Validation', leave=False, dynamic_ncols=True)
        for batch in progress:
            src, tgt = batch["src"], batch["tgt"]
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]  # for teacher-forcing loss
            logits = model(src, tgt_in)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt_out.reshape(-1),
                ignore_index=pad_idx,
            )
            total_loss += loss.item()

        progress.close()

    avg_loss = total_loss / len(dataloader)
    ppl = np.exp(avg_loss)
    return avg_loss, ppl


train_dataset = Multi30KDataset(split="train")
val_dataset = Multi30KDataset(split="validation")
test_dataset = Multi30KDataset(split="test")

model = Transformer(
    src_pad_id=train_dataset.pad_id,
    tgt_pad_id=train_dataset.pad_id,
    tgt_sos_id=train_dataset.eos_id,
    n_src_vocab=train_dataset.vocab_size,
    n_tgt_vocab=train_dataset.vocab_size,
    n_positions=train_dataset.max_len,
    n_encoder_layers=2,
    n_decoder_layers=2,
    d_model=128,
    n_heads=4,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
scheduler = NoamScheduler(optimizer, d_model=128, warmup_steps=4000)

print(f"Initial LR from scheduler: {scheduler.get_last_lr()[0]:.2e}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

for epoch in range(1, 11):
    tqdm.write(f"\n[Epoch {epoch:02d}] ------------------------------")
    start = time.time()
    train_loss, train_ppl = train(model, train_loader, optimizer, scheduler, pad_idx=train_dataset.pad_id,device=device)

    end = time.time()
    lr = scheduler.get_last_lr()[0]
    tqdm.write(f"Training Loss: {train_loss:.4f} | PPL: {train_ppl:.2f} | LR: {lr:.2e} | Time: {end - start:.2f}s")
    val_loss, val_ppl = evaluate(model, val_loader, train_dataset.pad_id, device)
    tqdm.write(f"Validation Loss: {val_loss:.4f} | PPL: {val_ppl:.2f}")
