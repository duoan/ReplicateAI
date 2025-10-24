import json
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
from tokenizer import BaseTokenizer

torch.autograd.set_detect_anomaly(True)
device = torch.accelerator.current_accelerator()


def train(model, dataloader, optimizer, scheduler, pad_idx):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False, dynamic_ncols=True)
    for batch in progress_bar:
        src, tgt = batch["src_ids"], batch["tgt_ids"]
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]  # teacher forcing
        # Forward
        logits = model(src, tgt_in)  # (N, T, vocab)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1),
            ignore_index=pad_idx,
            label_smoothing=0.1,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    progress_bar.close()
    avg_loss = total_loss / len(dataloader)
    ppl = np.exp(avg_loss)
    return avg_loss, ppl


def evaluate(model, dataloader, pad_idx):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        progress = tqdm(dataloader, desc="Validation", leave=False, dynamic_ncols=True)
        for batch in progress:
            src, tgt = batch["src_ids"], batch["tgt_ids"]
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]  # for teacher-forcing loss
            logits = model(src, tgt_in)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt_out.reshape(-1),
                ignore_index=pad_idx,
                label_smoothing=0.1,
            )
            total_loss += loss.item()

        progress.close()

    avg_loss = total_loss / len(dataloader)
    ppl = np.exp(avg_loss)
    return avg_loss, ppl


def main():
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    src_tokenizer = BaseTokenizer.load("tokenizer_en_bpe.json")
    tgt_tokenizer = BaseTokenizer.load("tokenizer_de_bpe.json")
    train_dataset = Multi30KDataset(
        split="train", src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer
    )
    val_dataset = Multi30KDataset(
        split="validation", src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer
    )
    test_dataset = Multi30KDataset(
        split="test", src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer
    )

    train_loader = DataLoader(
        train_dataset, batch_size=32, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4, pin_memory=True)

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

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 10
    train_history = []
    val_history = []
    for epoch in range(1, 100):
        print(f"[Epoch {epoch:02}] Starting data loading...")
        start = time.time()
        train_loss, train_ppl = train(
            model,
            train_loader,
            optimizer,
            scheduler,
            pad_idx=train_dataset.pad_id,
        )

        end = time.time()
        lr = scheduler.get_last_lr()[0]
        print(
            f"Training Loss: {train_loss:.4f} | PPL: {train_ppl:.2f} | LR: {lr:.2e} | Time: {end - start:.2f}s\n"
        )
        train_history.append((train_loss, train_ppl))

        val_loss, val_ppl = evaluate(model, val_loader, train_dataset.pad_id)
        print(
            f"Validation Loss: {val_loss:.4f} | Last Best Loss: {best_val_loss:.4f} | PPL: {val_ppl:.2f}\n"
        )
        val_history.append((val_loss, val_ppl))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping at epoch {epoch} due to no improvement in validation loss."
                )
                break

    torch.save(model.state_dict(), "model.pth")
    with open("train_history.json", "w+") as f:
        json.dump(train_history, f)
    with open("val_history.json", "w+") as f:
        json.dump(val_history, f)


if __name__ == "__main__":
    main()
