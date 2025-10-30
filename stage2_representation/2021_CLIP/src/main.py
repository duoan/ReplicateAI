"""
Main implementation file for the paper reproduction.

Example:
    python main.py --config configs/default.yaml
"""
import getpass
import os
from datetime import datetime
from typing import Any, Optional, Union
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import CLIPTokenizerFast, Trainer, TrainingArguments
from PIL import Image

from model import CLIP

DEVICE = torch.accelerator.current_accelerator()


def compute_clip_loss(model, images, texts):
    logits_per_image, logits_per_text = model(images, texts)
    labels = torch.arange(images.size(0), device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    loss = (loss_i + loss_t) * 0.5
    return loss, logits_per_image, logits_per_text

def main():
    print("🚀 ReplicateAI: Implementation entry point")

    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # --- Load a small image-text pair dataset from Hugging Face for testing ---
    # Using Flickr8k which is small and provides image-caption pairs
    if torch.cuda.is_available():
        # 40.5K samples
        # split into 80% train and 20% test
        hf_dataset = load_dataset("ariG23498/flickr8k", split="train").shuffle(seed=42)
        split = hf_dataset.train_test_split(test_size=0.2, seed=42, shuffle=True)
        hf_train = split["train"]
        hf_test = split["test"]
    else:
        hf_train = load_dataset("ariG23498/flickr8k", split="train[:1024]")  # subset for quick tests
        hf_test = load_dataset("ariG23498/flickr8k", split="train[-512:]")

    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    class ImageTextPairDataset(torch.utils.data.Dataset):
        def __init__(self, hf_split, image_transform, text_tokenizer, max_length: int = 77):
            self.hf_split = hf_split
            self.image_transform = image_transform
            self.text_tokenizer = text_tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.hf_split)

        def __getitem__(self, idx: int):
            example = self.hf_split[idx]
            image = example["image"]  # PIL.Image
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            image_tensor = self.image_transform(image)
            text = example["caption"]
            tokenized = self.text_tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = tokenized.input_ids.squeeze(0)
            return {"pixel_values": image_tensor, "input_ids": input_ids}

    class DataCollator:
        def __call__(self, batch):
            images = torch.stack([item["pixel_values"] for item in batch], dim=0)
            input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
            return {"pixel_values": images, "input_ids": input_ids}

    train_dataset = ImageTextPairDataset(hf_train, transform, tokenizer)
    test_dataset = ImageTextPairDataset(hf_test, transform, tokenizer)

    model = CLIP(pad_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)

    from torchinfo import summary
    from torchview import draw_graph
    dataloader = DataLoader(train_dataset, batch_size=32, collate_fn=DataCollator())
    example = next(iter(dataloader))
    summary(model, input_data=[example["pixel_values"], example["input_ids"]])
    draw_graph(
        model,
        input_data=[example["pixel_values"], example["input_ids"]],
        device='cpu',
        expand_nested=True,
        save_graph=True,
        graph_dir="BT",
        filename=str(os.path.abspath('../figures/model')),
    )

    model.to(DEVICE)
    model.train()

    class CLIPTrainer(Trainer):
        def compute_loss(
                self,
                model: nn.Module,
                inputs: dict[str, Union[torch.Tensor, Any]],
                return_outputs: bool = False,
                num_items_in_batch: Optional[torch.Tensor] = None,
        ):
            images = inputs["pixel_values"]
            texts = inputs["input_ids"]
            loss, logits_per_image, logits_per_text = compute_clip_loss(model, images, texts)
            if return_outputs:
                return loss, {
                    "logits_per_image": logits_per_image,
                    "logits_per_text": logits_per_text
                }
            return loss

        def prediction_step(
                self,
                model: nn.Module,
                inputs: dict[str, Union[torch.Tensor, Any]],
                prediction_loss_only: bool,
                ignore_keys: Optional[list[str]] = None,
        ):
            images = inputs["pixel_values"]
            texts = inputs["input_ids"]
            with torch.no_grad():
                loss, logits_per_image, logits_per_text = compute_clip_loss(model, images, texts)
            if prediction_loss_only:
                return loss, None, None
            return loss, (logits_per_image, logits_per_text), None

    # https://docs.wandb.ai/models/integrations/huggingface#custom-logging-log-and-view-evaluation-samples-during-training
    os.environ["WANDB_PROJECT"] = "2021_CLIP"  # name your W&B project
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints
    os.environ["WANDB_WATCH"] = "all"  # log gradients and parameters

    batch_size = 1024 if torch.cuda.is_available() else 32
    args = TrainingArguments(
        output_dir="./outputs/",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=32 if torch.cuda.is_available() else 2,
        learning_rate=1e-3,
        logging_steps=10,
        eval_strategy="epoch",
        do_train=True,
        do_eval=True,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        weight_decay=0.05,
        max_grad_norm=1.0,
        disable_tqdm=False,
        fp16=True if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False,
        dataloader_pin_memory=True if torch.cuda.is_available() else False,
        dataloader_num_workers=8 if torch.cuda.is_available() else 0,
        dataloader_prefetch_factor=8 if torch.cuda.is_available() else None,
        report_to=["wandb"],
        run_name=f"{getpass.getuser()}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    )

    trainer = CLIPTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollator(),
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)


if __name__ == "__main__":
    main()
