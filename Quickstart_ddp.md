# Quick Start: Distributed Training

Get started with multi-GPU training in 5 minutes!

## Prerequisites

```bash
# Verify you have GPUs
nvidia-smi

# Install dependencies
pip install -r requirements.txt
```

## Step 1: Test Your Setup

```bash
# Run test suite to verify everything works
python test_ddp.py
```

Expected output:
```
✓ CUDA is available with 4 GPU(s)
✓ NCCL version: (2, 18, 3)
✓ torch.distributed imported successfully
✓ SyncBatchNorm conversion successful
✓ Mixed precision training works
✓ All tests passed!
```

## Step 2: Download Data

```bash
# Download EPRI dataset
python download_data.py

# Parse labels to YOLO format
python parse_labels.py
```

## Step 3: Run Training

### Single GPU (for testing)
```bash
python train_yolo_ddp.py \
    --model yolov8n-seg.pt \
    --epochs 10 \
    --batch-size 16
```

### Multi-GPU (4 GPUs)
```bash
bash train_ddp.sh 4
```

Or manually:
```bash
python -m torch.distributed.run \
    --nproc_per_node=4 \
    train_yolo_ddp.py \
    --model yolov8m-seg.pt \
    --epochs 100 \
    --batch-size 16
```

## Step 4: Monitor Training

Watch GPU utilization:
```bash
watch -n 1 nvidia-smi
```

Training progress will show:
```
==================================================================
Distributed YOLOv8 Training
==================================================================
World Size: 4
Rank: 0
==================================================================

Epoch 1
------------------------------------------------------------
  Batch [0/1234] - Loss: 2.4532
  Batch [10/1234] - Loss: 2.3891
  ...
  Average Loss: 2.1234
  Time: 180.5s
  Throughput: 27.3 batches/sec
  ✓ Saved checkpoint (loss: 2.1234)
```

## Step 5: Evaluate Results

Models are saved to `models/epri_distribution_ddp/`:
- `best.pt` - Best model by validation loss
- `last.pt` - Most recent checkpoint

```bash
# Evaluate model
python evaluate_model.py --model models/epri_distribution_ddp/best.pt

# Run inference
python inference.py \
    --model models/epri_distribution_ddp/best.pt \
    --source data/processed/val/images \
    --output outputs/predictions
```

## Optional: Profile Performance

### GPU Memory and Speed Profiling
```bash
python profile_gpu.py \
    --model yolov8m-seg.pt \
    --batch-sizes 4 8 16 32 64
```

Outputs:
- `outputs/profiling/gpu_profile.png` - Performance visualizations
- `outputs/profiling/profiling_report.txt` - Detailed metrics

### Multi-GPU Scaling Benchmark
```bash
python benchmark_ddp.py
```

Automatically tests 1, 2, 4, 8 GPUs and generates:
- `outputs/benchmarks/multi_gpu_benchmark.png` - Scaling charts
- `outputs/benchmarks/benchmark_results.csv` - Raw data

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size
```bash
python train_yolo_ddp.py --batch-size 8  # or 4
```

### Issue: "Address already in use"
**Solution:** Change port
```bash
python -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=29501 \
    train_yolo_ddp.py
```

### Issue: "NCCL error"
**Solution:** Check NCCL compatibility
```bash
python -c "import torch; print(torch.cuda.nccl.version())"
```

### Issue: Slow training on multiple GPUs
**Possible causes:**
1. Batch size too small per GPU (increase if memory allows)
2. Data loading bottleneck (increase `num_workers`)
3. Network overhead (check inter-GPU bandwidth)

## Performance Tips

### 1. Optimal Batch Size
Find the largest batch size that fits in memory:
```bash
python profile_gpu.py --batch-sizes 4 8 16 32 64 128
```

### 2. Mixed Precision Training
Already enabled by default in `train_yolo_ddp.py` via:
```python
with torch.cuda.amp.autocast():
    # Training happens here
```

### 3. Data Loading
Increase workers if CPU is not saturated:
```python
# In train_yolo_ddp.py, modify:
num_workers=16  # Default is 8
```

### 4. Gradient Accumulation
For larger effective batch sizes:
```python
# Add to train_yolo_ddp.py
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Architecture Overview

```
train_yolo_ddp.py
├── setup_ddp()              # Initialize distributed environment
├── get_dataloader()         # Create distributed data loader
│   └── DistributedSampler   # Ensures unique data per GPU
├── train_one_epoch()
│   ├── AMP autocast         # Mixed precision
│   ├── loss.backward()      # Gradient computation
│   └── all_reduce()         # Gradient synchronization (automatic in DDP)
└── cleanup_ddp()            # Clean up process group
```

## Comparison: Standard vs DDP Training

| Feature | train_yolo.py | train_yolo_ddp.py |
|---------|--------------|-------------------|
| Framework | Ultralytics API | Raw PyTorch |
| Multi-GPU | DataParallel | DistributedDataParallel |
| Scaling | ~1.5x (4 GPUs) | ~3.6x (4 GPUs) |
| Customization | Limited | Full control |
| Learning curve | Easy | Moderate |
| Production ready | Yes | Yes |

## Next Steps

1. **Experiment with hyperparameters:**
   ```bash
   python train_yolo_ddp.py --lr 0.02 --weight-decay 1e-4
   ```

2. **Try different model sizes:**
   ```bash
   # Nano (fastest)
   python train_yolo_ddp.py --model yolov8n-seg.pt
   
   # Large (most accurate)
   python train_yolo_ddp.py --model yolov8l-seg.pt
   ```

3. **Fine-tune on your own dataset:**
   - Update `dataset.yaml` with your data paths
   - Ensure YOLO format labels
   - Run training!

4. **Deploy to production:**
   - Export to ONNX: `model.export(format='onnx')`
   - Export to TensorRT: `model.export(format='engine')`
   - See `inference.py` for deployment examples

## Resources

- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [NVIDIA NCCL Docs](https://docs.nvidia.com/deeplearning/nccl/)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)

## Getting Help

If you encounter issues:
1. Check test output: `python test_ddp.py`
2. Review GPU status: `nvidia-smi`
3. Check logs in `models/epri_distribution_ddp/`
4. Open an issue on GitHub with:
   - Error message
   - GPU info (`nvidia-smi`)
   - PyTorch version (`python -c "import torch; print(torch.__version__)"`)

---

**Ready to train?** Start with: `bash train_ddp.sh 4`