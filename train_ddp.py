#!/bin/bash

# Distributed YOLOv8 Training Launch Script
# Usage: bash train_ddp.sh <num_gpus>
# Example: bash train_ddp.sh 4

# Number of GPUs (default: all available)
NUM_GPUS=${1:-$(nvidia-smi -L | wc -l)}

echo "=================================="
echo "Distributed YOLOv8 Training"
echo "=================================="
echo "Number of GPUs: $NUM_GPUS"
echo "=================================="

# Check if GPUs are available
if [ $NUM_GPUS -eq 0 ]; then
    echo "Error: No GPUs detected!"
    exit 1
fi

# Training configuration
MODEL="yolov8m-seg.pt"
DATA="data/processed/dataset.yaml"
EPOCHS=100
BATCH_SIZE=16  # Per GPU
IMG_SIZE=640
SAVE_DIR="models/epri_distribution_ddp"

# Launch distributed training
if [ $NUM_GPUS -eq 1 ]; then
    echo "Running single GPU training..."
    python train_yolo_ddp.py \
        --model $MODEL \
        --data $DATA \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --img-size $IMG_SIZE \
        --save-dir $SAVE_DIR
else
    echo "Running multi-GPU training with DDP..."
    python -m torch.distributed.run \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        train_yolo_ddp.py \
        --model $MODEL \
        --data $DATA \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --img-size $IMG_SIZE \
        --save-dir $SAVE_DIR
fi

echo "=================================="
echo "Training completed!"
echo "Models saved to: $SAVE_DIR"
echo "=================================="