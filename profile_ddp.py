"""
GPU Memory and Performance Profiling for YOLOv8 Training

Monitors:
- GPU utilization
- Memory usage
- Training throughput
- Bottlenecks

Usage:
    python profile_gpu.py --model yolov8m-seg.pt --batch-size 16
"""

import torch
import argparse
import time
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
import psutil
import GPUtil


def get_gpu_memory():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # GB
    return 0


def get_gpu_utilization():
    """Get GPU utilization percentage"""
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            return gpus[0].load * 100
    except:
        pass
    return 0


def profile_model_memory(model_name, img_size, batch_size):
    """Profile memory usage of different model components"""
    print(f"\nProfiling {model_name} with batch_size={batch_size}, img_size={img_size}")
    print("-" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    results = {}
    
    # 1. Empty cache baseline
    baseline_mem = get_gpu_memory()
    results['baseline'] = baseline_mem
    print(f"Baseline GPU Memory: {baseline_mem:.2f} GB")
    
    # 2. Model loading
    model = YOLO(model_name)
    model.model.to(device)
    model_mem = get_gpu_memory()
    results['model'] = model_mem - baseline_mem
    print(f"Model Memory: {results['model']:.2f} GB")
    
    # 3. Input tensor
    dummy_input = torch.randn(batch_size, 3, img_size, img_size).to(device)
    input_mem = get_gpu_memory()
    results['input'] = input_mem - model_mem
    print(f"Input Memory: {results['input']:.2f} GB")
    
    # 4. Forward pass
    with torch.no_grad():
        _ = model.model(dummy_input)
    forward_mem = get_gpu_memory()
    results['forward'] = forward_mem - input_mem
    print(f"Forward Pass Memory: {results['forward']:.2f} GB")
    
    # 5. Backward pass (with dummy loss)
    model.model.train()
    output = model.model(dummy_input)
    
    # Create dummy loss
    loss = output[0].sum() if isinstance(output, (list, tuple)) else output.sum()
    loss.backward()
    
    backward_mem = get_gpu_memory()
    results['backward'] = backward_mem - forward_mem
    print(f"Backward Pass Memory: {results['backward']:.2f} GB")
    
    # Peak memory
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        results['peak'] = peak_mem
        print(f"\nPeak GPU Memory: {peak_mem:.2f} GB")
    
    print("-" * 60)
    
    return results


def profile_training_speed(model_name, img_size, batch_sizes, num_iterations=50):
    """Profile training speed with different batch sizes"""
    print(f"\nProfiling training speed for {model_name}")
    print("-" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    
    for batch_size in batch_sizes:
        try:
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load model
            model = YOLO(model_name)
            model.model.to(device)
            model.model.train()
            
            # Create optimizer
            optimizer = torch.optim.SGD(model.model.parameters(), lr=0.01)
            
            # Warm-up
            dummy_input = torch.randn(batch_size, 3, img_size, img_size).to(device)
            for _ in range(5):
                optimizer.zero_grad()
                output = model.model(dummy_input)
                loss = output[0].sum() if isinstance(output, (list, tuple)) else output.sum()
                loss.backward()
                optimizer.step()
            
            # Benchmark
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            for _ in range(num_iterations):
                optimizer.zero_grad()
                output = model.model(dummy_input)
                loss = output[0].sum() if isinstance(output, (list, tuple)) else output.sum()
                loss.backward()
                optimizer.step()
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed_time = time.time() - start_time
            
            # Calculate metrics
            images_per_sec = (batch_size * num_iterations) / elapsed_time
            ms_per_batch = (elapsed_time / num_iterations) * 1000
            
            result = {
                'batch_size': batch_size,
                'images_per_sec': images_per_sec,
                'ms_per_batch': ms_per_batch,
                'gpu_memory_gb': get_gpu_memory(),
                'success': True
            }
            
            print(f"Batch Size {batch_size:3d}: {images_per_sec:6.1f} img/s, "
                  f"{ms_per_batch:6.1f} ms/batch, {result['gpu_memory_gb']:.2f} GB")
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                result = {
                    'batch_size': batch_size,
                    'images_per_sec': 0,
                    'ms_per_batch': 0,
                    'gpu_memory_gb': 0,
                    'success': False
                }
                print(f"Batch Size {batch_size:3d}: OUT OF MEMORY")
            else:
                raise e
        
        results.append(result)
    
    print("-" * 60)
    return results


def plot_profiling_results(memory_results, speed_results, output_dir='outputs/profiling'):
    """Create visualization of profiling results"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('YOLOv8 GPU Performance Profile', fontsize=16, fontweight='bold')
    
    # 1. Memory breakdown
    ax1 = axes[0, 0]
    components = ['model', 'input', 'forward', 'backward']
    memory_values = [memory_results.get(c, 0) for c in components]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = ax1.bar(components, memory_values, color=colors, alpha=0.8)
    ax1.set_ylabel('Memory (GB)', fontsize=12)
    ax1.set_title('Memory Usage Breakdown', fontsize=14)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, memory_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f} GB', ha='center', va='bottom')
    
    # 2. Training throughput
    ax2 = axes[0, 1]
    df_speed = pd.DataFrame([r for r in speed_results if r['success']])
    
    if not df_speed.empty:
        ax2.plot(df_speed['batch_size'], df_speed['images_per_sec'], 
                marker='o', linewidth=2, markersize=8, color='green')
        ax2.set_xlabel('Batch Size', fontsize=12)
        ax2.set_ylabel('Throughput (images/sec)', fontsize=12)
        ax2.set_title('Training Throughput vs Batch Size', fontsize=14)
        ax2.grid(alpha=0.3)
        
        # Add value labels
        for _, row in df_speed.iterrows():
            ax2.text(row['batch_size'], row['images_per_sec'], 
                    f"{row['images_per_sec']:.0f}", 
                    ha='center', va='bottom', fontsize=9)
    
    # 3. Memory vs Batch Size
    ax3 = axes[1, 0]
    if not df_speed.empty:
        ax3.plot(df_speed['batch_size'], df_speed['gpu_memory_gb'],
                marker='s', linewidth=2, markersize=8, color='purple')
        ax3.set_xlabel('Batch Size', fontsize=12)
        ax3.set_ylabel('GPU Memory (GB)', fontsize=12)
        ax3.set_title('Memory Usage vs Batch Size', fontsize=14)
        ax3.grid(alpha=0.3)
        
        # Add value labels
        for _, row in df_speed.iterrows():
            ax3.text(row['batch_size'], row['gpu_memory_gb'], 
                    f"{row['gpu_memory_gb']:.1f}", 
                    ha='center', va='bottom', fontsize=9)
    
    # 4. Latency
    ax4 = axes[1, 1]
    if not df_speed.empty:
        ax4.bar(df_speed['batch_size'], df_speed['ms_per_batch'], 
               color='coral', alpha=0.8)
        ax4.set_xlabel('Batch Size', fontsize=12)
        ax4.set_ylabel('Latency (ms/batch)', fontsize=12)
        ax4.set_title('Training Latency vs Batch Size', fontsize=14)
        ax4.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for _, row in df_speed.iterrows():
            ax4.text(row['batch_size'], row['ms_per_batch'], 
                    f"{row['ms_per_batch']:.0f}", 
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gpu_profile.png', dpi=300, bbox_inches='tight')
    print(f"\nProfile plot saved to: {output_dir}/gpu_profile.png")


def generate_report(memory_results, speed_results, output_dir='outputs/profiling'):
    """Generate detailed profiling report"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    report_path = Path(output_dir) / 'profiling_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("YOLOv8 GPU PERFORMANCE PROFILING REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        # System info
        f.write("SYSTEM INFORMATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"PyTorch Version: {torch.__version__}\n")
        f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"CUDA Version: {torch.version.cuda}\n")
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
        f.write("\n")
        
        # Memory profiling
        f.write("MEMORY PROFILING\n")
        f.write("-" * 70 + "\n")
        for key, value in memory_results.items():
            f.write(f"{key.capitalize():20s}: {value:.3f} GB\n")
        f.write("\n")
        
        # Speed profiling
        f.write("SPEED PROFILING\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Batch Size':<12} {'Throughput':<15} {'Latency':<15} {'Memory':<12} {'Status'}\n")
        f.write(f"{'':12} {'(img/s)':<15} {'(ms/batch)':<15} {'(GB)':<12}\n")
        f.write("-" * 70 + "\n")
        
        for result in speed_results:
            status = "✓" if result['success'] else "✗ OOM"
            f.write(f"{result['batch_size']:<12} "
                   f"{result['images_per_sec']:<15.1f} "
                   f"{result['ms_per_batch']:<15.1f} "
                   f"{result['gpu_memory_gb']:<12.2f} "
                   f"{status}\n")
        
        f.write("\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 70 + "\n")
        
        # Find optimal batch size
        successful = [r for r in speed_results if r['success']]
        if successful:
            optimal = max(successful, key=lambda x: x['images_per_sec'])
            f.write(f"• Optimal batch size: {optimal['batch_size']} "
                   f"({optimal['images_per_sec']:.0f} img/s)\n")
            
            # Memory efficiency
            if memory_results.get('peak', 0) > 0:
                if torch.cuda.is_available():
                    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    usage_pct = (memory_results['peak'] / total_mem) * 100
                    f.write(f"• Peak memory usage: {usage_pct:.1f}% of available GPU memory\n")
                    
                    if usage_pct > 90:
                        f.write("  ⚠ Consider reducing batch size to avoid OOM errors\n")
                    elif usage_pct < 50:
                        f.write("  ✓ You can increase batch size for better throughput\n")
        
        f.write("\n")
        f.write("=" * 70 + "\n")
    
    print(f"Detailed report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Profile YOLOv8 GPU Performance')
    parser.add_argument('--model', type=str, default='yolov8m-seg.pt',
                       help='Model to profile')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size')
    parser.add_argument('--batch-sizes', type=int, nargs='+',
                       default=[4, 8, 16, 32, 64],
                       help='Batch sizes to test')
    parser.add_argument('--output-dir', type=str, default='outputs/profiling',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("YOLOv8 GPU Performance Profiling")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("Error: CUDA not available! This script requires a GPU.")
        return
    
    # Memory profiling
    memory_results = profile_model_memory(args.model, args.img_size, batch_size=16)
    
    # Speed profiling
    speed_results = profile_training_speed(args.model, args.img_size, args.batch_sizes)
    
    # Generate visualizations
    plot_profiling_results(memory_results, speed_results, args.output_dir)
    
    # Generate report
    generate_report(memory_results, speed_results, args.output_dir)
    
    print("\n" + "=" * 70)
    print("Profiling Complete!")
    print(f"Results saved to: {args.output_dir}/")
    print("=" * 70)


if __name__ == '__main__':
    main()