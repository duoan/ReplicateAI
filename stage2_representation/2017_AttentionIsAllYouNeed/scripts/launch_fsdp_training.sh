#!/bin/bash

# Launch script for FSDP training on 8 A100 GPUs
# Usage: bash scripts/launch_fsdp_training.sh

# Number of GPUs
NUM_GPUS=8

# Navigate to the src directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(dirname "$SCRIPT_DIR")/src"

cd "$SRC_DIR" || exit 1

echo "Launching FSDP training on $NUM_GPUS GPUs..."
echo "Working directory: $(pwd)"

# Launch with torchrun (recommended for PyTorch 2.0+)
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    train_multi30k_fsdp.py

# Alternative: Use torch.distributed.launch (older PyTorch versions)
# python -m torch.distributed.launch \
#     --nproc_per_node=$NUM_GPUS \
#     --use_env \
#     train_multi30k_fsdp.py

echo "Training completed!"