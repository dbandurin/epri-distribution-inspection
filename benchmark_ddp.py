"""
Benchmark multi-GPU training performance
Measures throughput and speedup with different GPU configurations
"""

import subprocess
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def run_benchmark(num_gpus, epochs=3, batch_size=16):
    """Run training benchmark with specified number of GPUs"""
    print(f"\n{'='*60}")
    print(f"Benchmarking with {num_gpus} GPU(s)")
    print(f"{'='*60}")
    
    cmd = [
        'python', '-m', 'torch.distributed.run',
        f'--nproc_per_node={num_gpus}',
        '--master_port=29500',
        'train_yolo_ddp.py',
        '--model', 'yolov8n-seg.pt',  # Use nano for faster benchmarking
        '--data', 'data/processed/dataset.yaml',
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--img-size', '640',
        '--save-dir', f'models/benchmark_{num_gpus}gpu'
    ]
    
    if num_gpus == 1:
        cmd = [
            'python', 'train_yolo_ddp.py',
            '--model', 'yolov8n-seg.pt',
            '--data', 'data/processed/dataset.yaml',
            '--epochs', str(epochs),
            '--batch-size', str(batch_size),
            '--img-size', '640',
            '--save-dir', f'models/benchmark_{num_gpus}gpu'
        ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed_time = time.time() - start_time
    
    # Parse output for performance metrics
    throughput = None
    if result.returncode == 0:
        # Extract throughput from output
        for line in result.stdout.split('\n'):
            if 'Throughput:' in line:
                try:
                    throughput = float(line.split('Throughput:')[1].split('batches/sec')[0].strip())
                except:
                    pass
    
    return {
        'num_gpus': num_gpus,
        'total_time': elapsed_time,
        'time_per_epoch': elapsed_time / epochs,
        'batch_size_per_gpu': batch_size,
        'total_batch_size': batch_size * num_gpus,
        'throughput': throughput,
        'success': result.returncode == 0
    }


def calculate_metrics(results):
    """Calculate speedup and efficiency metrics"""
    baseline = results[0]['time_per_epoch']
    
    for result in results:
        result['speedup'] = baseline / result['time_per_epoch']
        result['efficiency'] = result['speedup'] / result['num_gpus'] * 100
    
    return results


def plot_results(results, output_dir='outputs/benchmarks'):
    """Create visualization of benchmark results"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('YOLOv8 Multi-GPU Training Performance', fontsize=16, fontweight='bold')
    
    # 1. Training Time per Epoch
    ax1 = axes[0, 0]
    ax1.bar(df['num_gpus'], df['time_per_epoch'], color='steelblue', alpha=0.8)
    ax1.set_xlabel('Number of GPUs', fontsize=12)
    ax1.set_ylabel('Time per Epoch (seconds)', fontsize=12)
    ax1.set_title('Training Time vs Number of GPUs', fontsize=14)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(df['time_per_epoch']):
        ax1.text(df['num_gpus'].iloc[i], v, f'{v:.1f}s', ha='center', va='bottom')
    
    # 2. Speedup
    ax2 = axes[0, 1]
    ax2.plot(df['num_gpus'], df['speedup'], marker='o', linewidth=2, 
             markersize=8, color='darkgreen', label='Actual Speedup')
    ax2.plot(df['num_gpus'], df['num_gpus'], '--', linewidth=2, 
             color='red', alpha=0.5, label='Linear Speedup')
    ax2.set_xlabel('Number of GPUs', fontsize=12)
    ax2.set_ylabel('Speedup', fontsize=12)
    ax2.set_title('Speedup vs Number of GPUs', fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(df['speedup']):
        ax2.text(df['num_gpus'].iloc[i], v, f'{v:.2f}x', ha='center', va='bottom')
    
    # 3. Efficiency
    ax3 = axes[1, 0]
    ax3.bar(df['num_gpus'], df['efficiency'], color='coral', alpha=0.8)
    ax3.set_xlabel('Number of GPUs', fontsize=12)
    ax3.set_ylabel('Efficiency (%)', fontsize=12)
    ax3.set_title('Scaling Efficiency', fontsize=14)
    ax3.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Ideal (100%)')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(df['efficiency']):
        ax3.text(df['num_gpus'].iloc[i], v, f'{v:.1f}%', ha='center', va='bottom')
    
    # 4. Throughput
    ax4 = axes[1, 1]
    if df['throughput'].notna().any():
        ax4.bar(df['num_gpus'], df['throughput'], color='purple', alpha=0.8)
        ax4.set_xlabel('Number of GPUs', fontsize=12)
        ax4.set_ylabel('Throughput (batches/sec)', fontsize=12)
        ax4.set_title('Training Throughput', fontsize=14)
        ax4.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(df['throughput']):
            if pd.notna(v):
                ax4.text(df['num_gpus'].iloc[i], v, f'{v:.2f}', ha='center', va='bottom')
    else:
        ax4.text(0.5, 0.5, 'Throughput data not available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Training Throughput', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/multi_gpu_benchmark.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_dir}/multi_gpu_benchmark.png")
    
    return fig


def main():
    """Run comprehensive benchmarks"""
    print("=" * 60)
    print("YOLOv8 Multi-GPU Training Benchmark")
    print("=" * 60)
    
    # Detect available GPUs
    try:
        import torch
        num_available_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {num_available_gpus}")
    except:
        print("Error: Could not detect GPUs")
        return
    
    if num_available_gpus == 0:
        print("Error: No GPUs available!")
        return
    
    # Run benchmarks for different GPU counts
    gpu_configs = [1, 2, 4, 8] if num_available_gpus >= 8 else \
                  [1, 2, 4] if num_available_gpus >= 4 else \
                  [1, 2] if num_available_gpus >= 2 else [1]
    
    gpu_configs = [n for n in gpu_configs if n <= num_available_gpus]
    
    results = []
    for num_gpus in gpu_configs:
        result = run_benchmark(num_gpus, epochs=3, batch_size=16)
        results.append(result)
        
        if not result['success']:
            print(f"Warning: Benchmark with {num_gpus} GPU(s) failed")
    
    # Calculate metrics
    results = calculate_metrics(results)
    
    # Save results
    output_dir = Path('outputs/benchmarks')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary table
    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'benchmark_results.csv', index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)
    
    # Create visualizations
    plot_results(results, output_dir)
    
    print(f"\nResults saved to: {output_dir}/")


if __name__ == '__main__':
    main()