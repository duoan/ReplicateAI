from datetime import datetime
import os
from pathlib import Path
import time

import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchview import draw_graph
from data import Multi30KDataset
from model import Transformer
from optimizer import NoamScheduler
from tokenizer import get_tokenizer
import sacrebleu
import wandb
import getpass

SCRIPT_DIR = Path(__file__).resolve().parent

torch.autograd.set_detect_anomaly(True)
device = torch.accelerator.current_accelerator()

src_tokenizer = get_tokenizer(str(SCRIPT_DIR / "tokenizer_en.json"))
tgt_tokenizer = get_tokenizer(str(SCRIPT_DIR / "tokenizer_de.json"))

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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


def attention_to_wandb_image(tokens_x, tokens_y, attn_2d, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(attn_2d, cmap="viridis")

    ax.set_xticks(np.arange(len(tokens_x)))
    ax.set_yticks(np.arange(len(tokens_y)))
    ax.set_xticklabels(tokens_x, rotation=90, fontsize=8)
    ax.set_yticklabels(tokens_y, fontsize=8)

    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return wandb.Image(Image.open(buf))


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


def visualize_attention(
    model,
    dataloader,
    epoch,
    max_heads: int = 4,
    max_samples: int = 4,
):
    """Visualize attention for each sample in the first batch."""
    model.eval()
    batch = next(iter(dataloader))
    src, tgt = batch["src_ids"].to(device), batch["tgt_ids"].to(device)
    tgt_in = tgt[:, :-1]

    with torch.inference_mode():
        _, attn_dict = model(src, tgt_in, return_attn=True)

    # --- Encoder self-attention ---
    print("Visualizing encoder attentions")
    for layer_i, attn in enumerate(attn_dict["encoder"]):
        log_dict = {}
        # attn shape: (N, H, T, T)
        N, H, Tq, Tk = attn.shape
        for n in range(min(N, max_samples)):
            valid_len = (src[n] != src_tokenizer.pad_id).sum().item()
            src_tokens = src_tokenizer.convert_ids_to_tokens(
                src[n][:valid_len].tolist()
            )
            for head_i in range(min(H, max_heads)):
                attn_2d = attn[n, head_i].detach().cpu().numpy()
                attn_2d = attn_2d[:valid_len, :valid_len]
                log_dict[f"encoder/sample_{n}/layer_{layer_i}/head_{head_i}"] = (
                    attention_to_wandb_image(
                        src_tokens,
                        src_tokens,
                        attn_2d,
                        f"Encoder L{layer_i} H{head_i} Sample {n}",
                    )
                )
        wandb.log(log_dict, step=epoch)

    # --- Decoder self-attention ---
    print("Visualizing decoder self attention")
    for layer_i, attn in enumerate(attn_dict["decoder_self"]):
        log_dict = {}
        N, H, Tq, Tk = attn.shape
        for n in range(min(N, max_samples)):
            valid_len = (tgt[n] != tgt_tokenizer.pad_id).sum().item()
            tgt_tokens = tgt_tokenizer.convert_ids_to_tokens(
                tgt[n][:valid_len].tolist()
            )

            for head_i in range(min(H, max_heads)):
                attn_2d = attn[n, head_i].detach().cpu().numpy()
                attn_2d = attn_2d[:valid_len, :valid_len]

                log_dict[f"decoder_self/sample_{n}/layer_{layer_i}/head_{head_i}"] = (
                    attention_to_wandb_image(
                        tgt_tokens,
                        tgt_tokens,
                        attn_2d,
                        f"Decoder Self L{layer_i} H{head_i} Sample {n}",
                    )
                )
        wandb.log(log_dict, step=epoch)

    # ---------------- Decoder cross-attention ----------------
    print("Visualizing decoder cross attention")
    for layer_i, attn in enumerate(attn_dict["decoder_cross"]):
        log_dict = {}
        N, H, Tq, Tk = attn.shape
        for n in range(min(N, max_samples)):
            src_valid_len = (src[n] != src_tokenizer.pad_id).sum().item()
            tgt_valid_len = (tgt[n] != tgt_tokenizer.pad_id).sum().item()
            src_tokens = src_tokenizer.convert_ids_to_tokens(
                src[n][:src_valid_len].tolist()
            )
            tgt_tokens = tgt_tokenizer.convert_ids_to_tokens(
                tgt[n][:tgt_valid_len].tolist()
            )

            for head_i in range(min(H, max_heads)):
                attn_2d = attn[n, head_i].detach().cpu().numpy()
                attn_2d = attn_2d[:tgt_valid_len, :src_valid_len]

                log_dict[f"decoder_cross/sample_{n}/layer_{layer_i}/head_{head_i}"] = (
                    attention_to_wandb_image(
                        src_tokens,
                        tgt_tokens,
                        attn_2d,
                        f"Decoder Cross L{layer_i} H{head_i} Sample {n}",
                    )
                )
        wandb.log(log_dict, step=epoch)


def compute_bleu(
    model: Transformer,
    dataloader,
    tgt_tokenizer,
    max_len=128,
) -> float:
    model.eval()
    references = []
    hypotheses = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating BLEU", dynamic_ncols=True)):
            src = batch["src_ids"].to(device)
            preds = model.greedy_decode(src, max_len=max_len)
            for i in range(src.size(0)):
                pred_tokens = tgt_tokenizer.decode(
                    preds[i].tolist(), skip_special_tokens=True
                )
                tgt_tokens = tgt_tokenizer.decode(
                    batch["tgt_ids"][i].tolist(), skip_special_tokens=True
                )

                hypotheses.append(pred_tokens)
                references.append(tgt_tokens)
                
                # DEBUG: Log first few examples
                if batch_idx == 0 and i < 3:
                    print(f"\n[DEBUG] Example {i}:")
                    print(f"  Prediction IDs: {preds[i][:20].tolist()}")
                    print(f"  Prediction: '{pred_tokens}'")
                    print(f"  Reference: '{tgt_tokens}'")
                    print(f"  Model SOS ID: {model.tgt_sos_id}")
                    print(f"  Tokenizer SOS ID: {tgt_tokenizer.sos_id}")
                    print(f"  Tokenizer EOS ID: {tgt_tokenizer.eos_id}")

    # DEBUG: Check if predictions are empty
    print(f"\n[DEBUG] Total samples: {len(hypotheses)}")
    print(f"[DEBUG] Empty predictions: {sum(1 for h in hypotheses if len(h.strip()) == 0)}")
    print(f"[DEBUG] First 3 hypotheses: {hypotheses[:3]}")
    print(f"[DEBUG] First 3 references: {references[:3]}")
    
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])

    print(f"BLEU score = {bleu.score:.2f}")
    return bleu.score


def main():
    torch.manual_seed(42)
    batch_size = 64

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
        train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )

    run = wandb.init(
        entity="reproduce-ai",
        project="2017_AttentionIsAllYouNeed",
        config={
            "src_pad_id": src_tokenizer.pad_id,
            "tgt_pad_id": tgt_tokenizer.pad_id,
            "tgt_sos_id": tgt_tokenizer.sos_id,
            "n_src_vocab": src_tokenizer.vocab_size,
            "n_tgt_vocab": tgt_tokenizer.vocab_size,
            "n_positions": train_dataset.max_len,
            "n_encoder_layers": 6,
            "n_decoder_layers": 6,
            "d_model": 512,
            "n_heads": 8,
            "d_ff": 2048,
            "dropout_prob": 0.1,
        },
        id=f"{getpass.getuser()}-multi30k-{datetime.now().strftime("%Y%m%d-%H%M%S")}",
    )

    model = Transformer(
        src_pad_id=train_dataset.pad_id,
        tgt_pad_id=tgt_tokenizer.pad_id,
        tgt_sos_id=tgt_tokenizer.sos_id,
        n_src_vocab=train_dataset.vocab_size,
        n_tgt_vocab=tgt_tokenizer.vocab_size,
        n_positions=train_dataset.max_len,
        n_encoder_layers=6,
        n_decoder_layers=6,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        dropout_prob=0.1,
    ).to(device)

    # visualize the model architecture
    example_input = next(iter(train_loader))
    draw_graph(
        model,
        input_data=[example_input["src_ids"], example_input["tgt_ids"][:, :-1]],
        expand_nested=True,
        save_graph=True,
        graph_dir="BT",
        filename=str(SCRIPT_DIR.parent / "figures" / "model"),
    )

    optimizer = optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=128, warmup_steps=4000)

    print(f"Initial LR from scheduler: {scheduler.get_last_lr()[0]:.2e}")

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 10
    train_history = []
    val_history = []
    for epoch in range(1, 101):
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

        visualize_attention(model, val_loader, epoch)

        val_loss, val_ppl = evaluate(model, val_loader, train_dataset.pad_id)
        print(
            f"Validation Loss: {val_loss:.4f} | Last Best Loss: {best_val_loss:.4f} | PPL: {val_ppl:.2f}\n"
        )
        val_history.append((val_loss, val_ppl))

        run.log(
            {
                "train_loss": train_loss,
                "train_ppl": train_ppl,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
            },
            step=epoch,
        )

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

    print("Evaluating BLEU on test set...")
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )
    bleu_score = compute_bleu(model, test_loader, tgt_tokenizer, max_len=128)
    print(f"Final BLEU score: {bleu_score:.2f}")

    torch.save(model.state_dict(), SCRIPT_DIR / "model.pth")
    run.log({"bleu_score": bleu_score})
    run.log_model(SCRIPT_DIR / "model.pth", "model")
    run.finish()

def test():
    model = Transformer(
        src_pad_id=src_tokenizer.pad_id,
        tgt_pad_id=tgt_tokenizer.pad_id,
        tgt_sos_id=tgt_tokenizer.sos_id,
        n_src_vocab=src_tokenizer.vocab_size,
        n_tgt_vocab=tgt_tokenizer.vocab_size,
        n_positions=128,
        n_encoder_layers=6,
        n_decoder_layers=6,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        dropout_prob=0.1,
    )
    
    state_dict = torch.load(SCRIPT_DIR / "model.pth")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print("Evaluating BLEU on test set...")
    test_dataset = Multi30KDataset(
        split="test", src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, num_workers=4, pin_memory=True
    )
    bleu_score = compute_bleu(model, test_loader, tgt_tokenizer, max_len=128)
    print(f"Final BLEU score: {bleu_score:.2f}")

if __name__ == "__main__":
    main()
