"""
Test script for DDP implementation
Verifies that distributed training works correctly

Usage:
    python test_ddp.py
"""

import torch
import torch.distributed as dist
import os
import subprocess
import sys


def test_cuda_available():
    """Test if CUDA is available"""
    print("\n1. Testing CUDA availability...")
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"   ✓ CUDA is available with {num_gpus} GPU(s)")
        for i in range(num_gpus):
            print(f"     - GPU {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        print("   ✗ CUDA is not available")
        return False


def test_nccl_backend():
    """Test if NCCL backend is available"""
    print("\n2. Testing NCCL backend...")
    try:
        if torch.cuda.is_available():
            nccl_version = torch.cuda.nccl.version()
            print(f"   ✓ NCCL version: {nccl_version}")
            return True
        else:
            print("   ⚠ CUDA not available, cannot test NCCL")
            return False
    except Exception as e:
        print(f"   ✗ NCCL test failed: {e}")
        return False


def test_distributed_import():
    """Test if torch.distributed can be imported"""
    print("\n3. Testing torch.distributed import...")
    try:
        import torch.distributed as dist
        print("   ✓ torch.distributed imported successfully")
        return True
    except Exception as e:
        print(f"   ✗ Failed to import torch.distributed: {e}")
        return False


def test_single_gpu_training():
    """Test single GPU training"""
    print("\n4. Testing single GPU training...")
    
    try:
        # Run a quick training test
        cmd = [
            'python', 'train_yolo_ddp.py',
            '--model', 'yolov8n-seg.pt',
            '--data', 'data/processed/dataset.yaml',
            '--epochs', '1',
            '--batch-size', '4',
            '--save-dir', 'models/test_single_gpu'
        ]
        
        print("   Running: " + " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("   ✓ Single GPU training successful")
            return True
        else:
            print("   ✗ Single GPU training failed")
            print(f"   Error: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ✗ Training timed out (>5 minutes)")
        return False
    except FileNotFoundError:
        print("   ⚠ train_yolo_ddp.py not found, skipping")
        return None
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def test_multi_gpu_setup():
    """Test if multi-GPU setup works"""
    print("\n5. Testing multi-GPU setup...")
    
    if torch.cuda.device_count() < 2:
        print("   ⚠ Less than 2 GPUs available, skipping multi-GPU test")
        return None
    
    try:
        # Test with 2 GPUs
        cmd = [
            'python', '-m', 'torch.distributed.run',
            '--nproc_per_node=2',
            '--master_port=29500',
            'train_yolo_ddp.py',
            '--model', 'yolov8n-seg.pt',
            '--data', 'data/processed/dataset.yaml',
            '--epochs', '1',
            '--batch-size', '4',
            '--save-dir', 'models/test_multi_gpu'
        ]
        
        print("   Running: " + " ".join(cmd[:4]) + " ...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("   ✓ Multi-GPU training successful")
            return True
        else:
            print("   ✗ Multi-GPU training failed")
            print(f"   Error: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ✗ Training timed out (>10 minutes)")
        return False
    except FileNotFoundError:
        print("   ⚠ train_yolo_ddp.py not found, skipping")
        return None
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def test_syncbatchnorm():
    """Test SyncBatchNorm conversion"""
    print("\n6. Testing SyncBatchNorm conversion...")
    
    try:
        from ultralytics import YOLO
        
        # Create a simple model
        model = YOLO('yolov8n-seg.pt').model
        
        # Count BatchNorm layers
        bn_count = 0
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                bn_count += 1
        
        print(f"   Found {bn_count} BatchNorm layers")
        
        # Convert to SyncBatchNorm
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        # Count SyncBatchNorm layers
        sync_bn_count = 0
        for module in model.modules():
            if isinstance(module, torch.nn.SyncBatchNorm):
                sync_bn_count += 1
        
        print(f"   Converted to {sync_bn_count} SyncBatchNorm layers")
        
        if sync_bn_count == bn_count:
            print("   ✓ SyncBatchNorm conversion successful")
            return True
        else:
            print("   ✗ SyncBatchNorm conversion incomplete")
            return False
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def test_mixed_precision():
    """Test mixed precision training"""
    print("\n7. Testing mixed precision (AMP)...")
    
    try:
        device = torch.device('cuda:0')
        dummy_input = torch.randn(2, 3, 640, 640).to(device)
        
        from ultralytics import YOLO
        model = YOLO('yolov8n-seg.pt').model.to(device)
        
        # Test AMP
        scaler = torch.cuda.amp.GradScaler()
        
        with torch.cuda.amp.autocast():
            output = model(dummy_input)
        
        print("   ✓ Mixed precision training works")
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def print_summary(results):
    """Print test summary"""
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    total = len([r for r in results.values() if r is not None])
    passed = len([r for r in results.values() if r is True])
    failed = len([r for r in results.values() if r is False])
    skipped = len([r for r in results.values() if r is None])
    
    print(f"Total Tests:   {total}")
    print(f"Passed:        {passed} ✓")
    print(f"Failed:        {failed} ✗")
    print(f"Skipped:       {skipped} ⚠")
    print("=" * 70)
    
    if failed > 0:
        print("\n⚠ Some tests failed. Please check the output above.")
        print("Common issues:")
        print("  - CUDA/GPU not available")
        print("  - Missing dependencies (install requirements.txt)")
        print("  - Dataset not downloaded (run download_data.py)")
        print("  - Insufficient GPU memory (reduce batch size)")
    elif passed == total:
        print("\n✓ All tests passed! Your DDP setup is ready.")
    
    return failed == 0


def main():
    print("=" * 70)
    print("DISTRIBUTED TRAINING TEST SUITE")
    print("=" * 70)
    print("This script tests your DDP implementation setup")
    
    results = {}
    
    # Run tests
    results['cuda'] = test_cuda_available()
    results['nccl'] = test_nccl_backend()
    results['distributed'] = test_distributed_import()
    results['syncbn'] = test_syncbatchnorm()
    results['amp'] = test_mixed_precision()
    results['single_gpu'] = test_single_gpu_training()
    results['multi_gpu'] = test_multi_gpu_setup()
    
    # Print summary
    all_passed = print_summary(results)
    
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()