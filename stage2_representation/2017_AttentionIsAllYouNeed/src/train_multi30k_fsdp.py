from datetime import datetime
import os
from pathlib import Path
import time

import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, StateDictType
import torch.distributed as dist
from tqdm.auto import tqdm
from data import Multi30KDataset
from model import Transformer
from optimizer import NoamScheduler
from tokenizer import get_tokenizer
import sacrebleu
import wandb
import getpass

SCRIPT_DIR = Path(__file__).resolve().parent

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_distributed():
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        print("Not using distributed mode")
        return 0, 1, 0
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_fsdp_config(use_mixed_precision=False):
    """Configure FSDP settings optimized for A100 GPUs."""
    config = {
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "device_id": torch.cuda.current_device(),
        "limit_all_gathers": True,
        "use_orig_params": True,
    }
    
    # Only use mixed precision if explicitly enabled
    if use_mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        config["mixed_precision"] = mixed_precision_policy
    
    return config


def train(model, dataloader, optimizer, scheduler, pad_idx, rank, world_size):
    model.train()
    total_loss = 0
    num_batches = 0
    
    if rank == 0:
        progress_bar = tqdm(dataloader, desc="Training", leave=False, dynamic_ncols=True)
    else:
        progress_bar = dataloader
    
    for batch in progress_bar:
        src, tgt = batch["src_ids"], batch["tgt_ids"]
        src = src.cuda()
        tgt = tgt.cuda()

        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]  # teacher forcing
        
        # Forward
        logits = model(src, tgt_in)  # (N, T, vocab)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1),
            ignore_index=pad_idx,
            label_smoothing=0.05,
        )

        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            if rank == 0:
                print(f"Warning: Invalid loss detected: {loss.item()}, skipping batch")
            continue

        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1

        if rank == 0:
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    if rank == 0:
        progress_bar.close()
    
    # Gather loss from all ranks
    total_loss_tensor = torch.tensor(total_loss, device='cuda')
    num_batches_tensor = torch.tensor(num_batches, device='cuda')
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
    
    avg_loss = total_loss_tensor.item() / max(num_batches_tensor.item(), 1)
    ppl = np.exp(min(avg_loss, 100))  # Cap to prevent overflow
    
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


def evaluate(model, dataloader, pad_idx, rank, world_size):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        if rank == 0:
            progress = tqdm(dataloader, desc="Validation", leave=False, dynamic_ncols=True)
        else:
            progress = dataloader
            
        for batch in progress:
            src, tgt = batch["src_ids"], batch["tgt_ids"]
            src = src.cuda()
            tgt = tgt.cuda()

            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            logits = model(src, tgt_in)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt_out.reshape(-1),
                ignore_index=pad_idx,
                label_smoothing=0.05,
            )
            
            # Check for NaN/Inf
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                num_batches += 1
            
        if rank == 0:
            progress.close()

    # Gather loss from all ranks
    total_loss_tensor = torch.tensor(total_loss, device='cuda')
    num_batches_tensor = torch.tensor(num_batches, device='cuda')
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
    
    avg_loss = total_loss_tensor.item() / max(num_batches_tensor.item(), 1)
    ppl = np.exp(min(avg_loss, 100))  # Cap to prevent overflow
    
    return avg_loss, ppl


def visualize_attention(
    model,
    dataloader,
    epoch,
    rank,
    src_tokenizer,
    tgt_tokenizer,
    max_heads: int = 4,
    max_samples: int = 4,
):
    """Visualize attention for each sample in the first batch (only on rank 0)."""
    if rank != 0:
        return
    
    model.eval()
    batch = next(iter(dataloader))
    src, tgt = batch["src_ids"].cuda(), batch["tgt_ids"].cuda()
    tgt_in = tgt[:, :-1]

    with torch.inference_mode():
        _, attn_dict = model(src, tgt_in, return_attn=True)

    # --- Encoder self-attention ---
    print("Visualizing encoder attentions")
    for layer_i, attn in enumerate(attn_dict["encoder"]):
        log_dict = {}
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

    # --- Decoder cross-attention ---
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
    rank,
    world_size,
    max_len=128,
) -> float:
    model.eval()
    references = []
    hypotheses = []

    with torch.no_grad():
        if rank == 0:
            progress = tqdm(dataloader, desc="Evaluating BLEU", dynamic_ncols=True)
        else:
            progress = dataloader
            
        for batch_idx, batch in enumerate(progress):
            src = batch["src_ids"].cuda()
            preds = model.greedy_decode(src, max_len=max_len, eos_id=tgt_tokenizer.eos_id)
            
            for i in range(src.size(0)):
                pred_tokens = tgt_tokenizer.decode(
                    preds[i].tolist(), skip_special_tokens=True
                )
                tgt_tokens = tgt_tokenizer.decode(
                    batch["tgt_ids"][i].tolist(), skip_special_tokens=True
                )

                hypotheses.append(pred_tokens)
                references.append(tgt_tokens)
                
                # DEBUG: Log first few examples (only on rank 0)
                if rank == 0 and batch_idx == 0 and i < 3:
                    print(f"\n[DEBUG] Example {i}:")
                    print(f"  Prediction: '{pred_tokens}'")
                    print(f"  Reference: '{tgt_tokens}'")

    # Gather predictions from all ranks
    all_hypotheses = [None] * world_size
    all_references = [None] * world_size
    dist.all_gather_object(all_hypotheses, hypotheses)
    dist.all_gather_object(all_references, references)
    
    if rank == 0:
        # Flatten the gathered lists
        hypotheses = [h for sublist in all_hypotheses for h in sublist]
        references = [r for sublist in all_references for r in sublist]
        
        print(f"\n[DEBUG] Total samples: {len(hypotheses)}")
        print(f"[DEBUG] Empty predictions: {sum(1 for h in hypotheses if len(h.strip()) == 0)}")
        
        bleu = sacrebleu.corpus_bleu(hypotheses, [references])
        print(f"BLEU score = {bleu.score:.2f}")
        return bleu.score
    
    return 0.0


def main():
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Set random seed for reproducibility
    torch.manual_seed(42 + rank)
    
    # Load tokenizers (only on rank 0, then broadcast if needed)
    src_tokenizer = get_tokenizer(str(SCRIPT_DIR / "tokenizer_en.json"))
    tgt_tokenizer = get_tokenizer(str(SCRIPT_DIR / "tokenizer_de.json"))
    
    # Adjust batch size per GPU (reduced for FP32)
    batch_size_per_gpu = 32  # Adjust based on your GPU memory
    
    # Create datasets
    train_dataset = Multi30KDataset(
        split="train", src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer
    )
    val_dataset = Multi30KDataset(
        split="validation", src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer
    )
    test_dataset = Multi30KDataset(
        split="test", src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_per_gpu,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize wandb only on rank 0
    if rank == 0:
        run = wandb.init(
            entity="replicate-ai",
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
                "world_size": world_size,
                "batch_size_per_gpu": batch_size_per_gpu,
                "effective_batch_size": batch_size_per_gpu * world_size,
            },
            id=f"{getpass.getuser()}-multi30k-fsdp-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        )

    # Create model
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
    )
    
    # Initialize weights properly
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0, std=0.02)
    
    model.apply(init_weights)
    model = model.cuda()

    # Wrap model with FSDP (disable mixed precision for now)
    fsdp_config = get_fsdp_config(use_mixed_precision=False)
    model = FSDP(model, **fsdp_config)

    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)

    steps_per_epoch = len(train_dataset) // (batch_size_per_gpu * world_size)
    warmup_steps = steps_per_epoch * 2  # 2 epochs warmup

    scheduler = NoamScheduler(optimizer, d_model=512, warmup_steps=warmup_steps)

    if rank == 0:
        print(f"Initial LR from scheduler: {scheduler.get_last_lr()[0]:.2e}")
        print(f"Training on {world_size} GPUs with effective batch size: {batch_size_per_gpu * world_size}")
        print(f"Using FP32 precision (mixed precision disabled)")

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 10
    
    for epoch in range(1, 101):
        # Set epoch for distributed sampler
        train_sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"[Epoch {epoch:02}] Starting training...")
        
        start = time.time()
        train_loss, train_ppl = train(
            model,
            train_loader,
            optimizer,
            scheduler,
            pad_idx=train_dataset.pad_id,
            rank=rank,
            world_size=world_size,
        )

        end = time.time()
        lr = scheduler.get_last_lr()[0]
        
        if rank == 0:
            print(
                f"Training Loss: {train_loss:.4f} | PPL: {train_ppl:.2f} | LR: {lr:.2e} | Time: {end - start:.2f}s\n"
            )

        # Visualize attention (only on rank 0)
        if rank == 0:
            visualize_attention(model, val_loader, epoch, rank, src_tokenizer, tgt_tokenizer)

        val_loss, val_ppl = evaluate(
            model, val_loader, train_dataset.pad_id, rank, world_size
        )
        
        if rank == 0:
            print(
                f"Validation Loss: {val_loss:.4f} | Last Best Loss: {best_val_loss:.4f} | PPL: {val_ppl:.2f}\n"
            )

            run.log(
                {
                    "train_loss": train_loss,
                    "train_ppl": train_ppl,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "lr": lr,
                },
                step=epoch,
            )

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save checkpoint (only on rank 0)
            if rank == 0:
                # Save FSDP model state
                save_policy = StateDictType.FULL_STATE_DICT
                with FSDP.state_dict_type(model, save_policy):
                    state_dict = model.state_dict()
                    torch.save(state_dict, SCRIPT_DIR / "model_fsdp.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if rank == 0:
                    print(
                        f"Early stopping at epoch {epoch} due to no improvement in validation loss."
                    )
                break

    # Evaluate BLEU on test set
    if rank == 0:
        print("Evaluating BLEU on test set...")
    
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_per_gpu,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
    )
    
    bleu_score = compute_bleu(
        model, test_loader, tgt_tokenizer, rank, world_size, max_len=128
    )
    
    if rank == 0:
        print(f"Final BLEU score: {bleu_score:.2f}")
        run.log({"bleu_score": bleu_score})
        run.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()