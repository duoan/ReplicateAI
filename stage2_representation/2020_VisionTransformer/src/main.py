"""
Main implementation file for the paper reproduction.
Optimized for local MacBook Air (CPU/MPS) testing with ImageNette subset.
"""
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch import stack, tensor
from torch.utils.data import Subset
from torchvision import datasets, transforms
from transformers import TrainingArguments, Trainer

from model import ViTConfig, VisionTransformer


def main():
    print("🚀 ReplicateAI: Implementation entry point for local MacBook Air test.")

    # ----------------------------------------------------
    # --- 1. CONFIGURATION FOR LOCAL TESTING ---
    # ----------------------------------------------------

    # Check for Apple Silicon (MPS) or default to CPU
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print("⚡ Using Apple Silicon (MPS) device.")
    else:
        DEVICE = torch.device("cpu")
        print("🐌 MPS not available, falling back to CPU.")

    # Drastically reduced batch size for stability and memory limits
    TEST_BATCH_SIZE = 4

    # Use a tiny subset for quick sanity checks
    MAX_TRAIN_SAMPLES = 128
    MAX_EVAL_SAMPLES = 32
    # ----------------------------------------------------

    # Define the root directory where the dataset will be downloaded
    DATA_ROOT = "./data/imagenette"
    os.makedirs(DATA_ROOT, exist_ok=True)

    # Standard ViT/ImageNet preprocessing pipeline (224x224 output)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # Normalization is crucial for ViT, even for testing
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the ImageNette dataset
    full_train_ds = datasets.Imagenette(
        root=DATA_ROOT, split='train', size='full', download=True, transform=transform
    )
    full_test_ds = datasets.Imagenette(
        root=DATA_ROOT, split='val', size='full', download=True, transform=transform
    )

    # Create tiny subsets for fast execution
    if torch.cuda.is_available():
        train_ds = full_train_ds
        test_ds = full_test_ds
    else:
        train_ds = Subset(full_train_ds, range(MAX_TRAIN_SAMPLES))
        test_ds = Subset(full_test_ds, range(MAX_EVAL_SAMPLES))

    print(f"Dataset subset size: Train={len(train_ds)}, Eval={len(test_ds)}")

    # ViT Config for 10 classes and 224x224 size
    config = ViTConfig(num_classes=10, image_size=224, patch_size=16, encoder_stride=16)
    model = VisionTransformer(config)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # Labels from torchvision dataset subset are typically tensors/arrays, not tuples
        # Ensure labels are extracted correctly (Trainer handles this mostly, but good practice)
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, average='weighted', zero_division=0),
            "recall": recall_score(labels, preds, average='weighted', zero_division=0),
        }

    def custom_data_collator(samples):
        """
        Collate function to handle (image, label) tuples from torchvision datasets
        and format them into a dictionary for the Hugging Face Trainer.
        """
        # 1. Unpack the list of (image, label) tuples
        images = [sample[0] for sample in samples]
        labels = [sample[1] for sample in samples]

        # 2. Stack the tensors and convert the labels to a tensor
        # torch.stack creates a batch tensor from a list of tensors (for images)
        # torch.tensor converts a list of numbers to a tensor (for labels)

        return {
            # The Trainer expects the input image tensor under the key 'pixel_values'
            "pixel_values": stack(images),
            # The Trainer expects the labels under the key 'labels'
            "labels": tensor(labels),
        }

    args = TrainingArguments(
        "vit-imagenette-macbook-test",  # Unique name for test runs
        per_device_train_batch_size=TEST_BATCH_SIZE,
        per_device_eval_batch_size=TEST_BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="no",  # No need to save checkpoints for a quick test
        num_train_epochs=2,  # Run for only 2 epochs
        learning_rate=3e-4,
        weight_decay=0.05,
        logging_steps=10,
        load_best_model_at_end=False,  # No need to load best model
        report_to="none",
        # Pass the device to TrainingArguments (important for non-CUDA systems)
        optim="adamw_torch",  # Explicitly use a common optimizer
        disable_tqdm=False,  # Keep progress bar for visibility
        dataloader_pin_memory=True if torch.cuda.is_available() else False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        data_collator=custom_data_collator,
    )

    # Manually move model to the determined device (Trainer will handle it, but this is a good check)
    model.to(DEVICE)

    trainer.train()


if __name__ == "__main__":
    main()
