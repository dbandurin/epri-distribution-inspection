"""
Distributed Data Parallel Training for YOLOv8 Segmentation
Multi-GPU training using PyTorch DistributedDataParallel

Usage:
    Single GPU:
        python train_yolo_ddp.py
    
    Multi-GPU (4 GPUs):
        python -m torch.distributed.run --nproc_per_node=4 train_yolo_ddp.py
"""

import os
import argparse
from pathlib import Path
import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from ultralytics import YOLO
from ultralytics.data import YOLODataset
from ultralytics.utils import LOGGER
import time
from datetime import datetime


def setup_ddp():
    """Initialize distributed training environment"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl',  # Use NCCL for GPU
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        # Set device
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        # Single GPU mode
        return 0, 1, 0


def cleanup_ddp():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_dataloader(data_path, img_size, batch_size, rank, world_size, split='train'):
    """Create distributed dataloader"""
    # Load data config
    with open(data_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get image and label paths
    if split == 'train':
        img_path = Path(data_config['path']) / data_config['train']
    else:
        img_path = Path(data_config['path']) / data_config['val']
    
    # Create dataset
    dataset = YOLODataset(
        img_path=str(img_path.parent),
        imgsz=img_size,
        augment=(split == 'train'),
        cache=False,
        prefix=f'{split}: '
    )
    
    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=(split == 'train')
    ) if world_size > 1 else None
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and split == 'train'),
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        collate_fn=getattr(dataset, 'collate_fn', None)
    )
    
    return loader, sampler


def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch, rank, world_size):
    """Train for one epoch"""
    model.train()
    
    if rank == 0:
        print(f"\nEpoch {epoch + 1}")
        print("-" * 60)
    
    total_loss = 0
    num_batches = 0
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        images = batch['img'].to(device, non_blocking=True)
        
        # Forward pass with automatic mixed precision
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            # YOLO model forward pass
            loss = model(images, batch)
            
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Print progress (only on rank 0)
        if rank == 0 and batch_idx % 10 == 0:
            avg_loss = total_loss / num_batches
            print(f"  Batch [{batch_idx}/{len(dataloader)}] - Loss: {avg_loss:.4f}")
    
    # Calculate epoch statistics
    epoch_time = time.time() - start_time
    avg_loss = total_loss / num_batches
    
    # Aggregate loss across all GPUs
    if world_size > 1:
        loss_tensor = torch.tensor([avg_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()
    
    if rank == 0:
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Throughput: {len(dataloader) * world_size / epoch_time:.2f} batches/sec")
    
    return avg_loss


def main(args):
    """Main training function"""
    # Setup DDP
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    
    if rank == 0:
        print("=" * 60)
        print("Distributed YOLOv8 Training")
        print("=" * 60)
        print(f"World Size: {world_size}")
        print(f"Rank: {rank}")
        print(f"Local Rank: {local_rank}")
        print(f"Device: {device}")
        print("=" * 60)
    
    # Load base YOLO model
    base_model = YOLO(args.model)
    
    # Extract PyTorch model
    model = base_model.model.to(device)
    
    # Convert BatchNorm to SyncBatchNorm for better multi-GPU training
    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Setup optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # Create dataloaders
    train_loader, train_sampler = get_dataloader(
        args.data,
        args.img_size,
        args.batch_size,
        rank,
        world_size,
        split='train'
    )
    
    val_loader, _ = get_dataloader(
        args.data,
        args.img_size,
        args.batch_size,
        rank,
        world_size,
        split='val'
    )
    
    # Training loop
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Train one epoch
        avg_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch,
            rank,
            world_size
        )
        
        # Step scheduler
        scheduler.step()
        
        # Save checkpoint (only on rank 0)
        if rank == 0:
            if avg_loss < best_loss:
                best_loss = avg_loss
                
                # Save model
                save_dir = Path(args.save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                }
                
                torch.save(checkpoint, save_dir / 'best.pt')
                torch.save(checkpoint, save_dir / 'last.pt')
                
                print(f"  ✓ Saved checkpoint (loss: {avg_loss:.4f})")
    
    total_time = time.time() - start_time
    
    if rank == 0:
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Total Time: {total_time / 3600:.2f} hours")
        print(f"Best Loss: {best_loss:.4f}")
        print("=" * 60)
    
    # Cleanup
    cleanup_ddp()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributed YOLOv8 Training')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='yolov8m-seg.pt',
                        help='Base model to use')
    parser.add_argument('--data', type=str, default='data/processed/dataset.yaml',
                        help='Path to dataset YAML config')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size per GPU')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Image size')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay')
    
    # Output
    parser.add_argument('--save-dir', type=str, default='models/epri_distribution_ddp',
                        help='Directory to save models')
    
    args = parser.parse_args()
    
    main(args)