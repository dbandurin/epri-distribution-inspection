"""
Evaluate trained model on validation/test set
"""
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd

CLASS_NAMES = [
    'conductor', 'other_wire', 'pole', 'crossarm', 
    'insulator', 'cutouts', 'transformer', 'background_structure'
]


def evaluate_model(model_path, data_yaml, split='val', save_dir='outputs/evaluation'):
    """
    Comprehensive model evaluation
    
    Args:
        model_path: Path to trained model weights
        data_yaml: Path to dataset YAML
        split: 'val' or 'test'
        save_dir: Directory to save evaluation results
    """
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Dataset: {data_yaml}")
    print(f"Split: {split}")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model = YOLO(model_path)
    
    # Run validation
    print("\nRunning validation...")
    results = model.val(
        data=data_yaml,
        split=split,
        save_json=True,
        save_hybrid=True,
        conf=0.001,  # Low threshold to get all detections for analysis
        iou=0.6,
        max_det=300,
        plots=True,
        verbose=True
    )
    
    # Print metrics
    print("\n" + "=" * 60)
    print("Overall Metrics")
    print("=" * 60)
    
    if hasattr(results, 'box'):
        print(f"Box mAP@0.5: {results.box.map50:.4f}")
        print(f"Box mAP@0.5:0.95: {results.box.map:.4f}")
        print(f"Precision: {results.box.mp:.4f}")
        print(f"Recall: {results.box.mr:.4f}")
    
    if hasattr(results, 'seg'):
        print(f"\nMask mAP@0.5: {results.seg.map50:.4f}")
        print(f"Mask mAP@0.5:0.95: {results.seg.map:.4f}")
        print(f"Mask Precision: {results.seg.mp:.4f}")
        print(f"Mask Recall: {results.seg.mr:.4f}")
    
    # Per-class metrics
    print("\n" + "=" * 60)
    print("Per-Class Metrics")
    print("=" * 60)
    
    if hasattr(results, 'box') and hasattr(results.box, 'ap_class_index'):
        metrics_data = []
        
        for idx, class_idx in enumerate(results.box.ap_class_index):
            class_name = CLASS_NAMES[int(class_idx)]
            
            metric_dict = {
                'Class': class_name,
                'Precision': results.box.p[idx] if hasattr(results.box, 'p') else 0,
                'Recall': results.box.r[idx] if hasattr(results.box, 'r') else 0,
                'mAP@0.5': results.box.ap50[idx] if hasattr(results.box, 'ap50') else 0,
                'mAP@0.5:0.95': results.box.ap[idx] if hasattr(results.box, 'ap') else 0,
            }
            
            metrics_data.append(metric_dict)
            print(f"\n{class_name}:")
            for key, value in metric_dict.items():
                if key != 'Class':
                    print(f"  {key}: {value:.4f}")
        
        # Save metrics to CSV
        df = pd.DataFrame(metrics_data)
        csv_path = save_dir / 'per_class_metrics.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Per-class metrics saved to: {csv_path}")
        
        # Create visualization
        plot_per_class_metrics(df, save_dir)
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"Results saved to: {save_dir}")
    print("=" * 60)
    
    return results


def plot_per_class_metrics(df, save_dir):
    """Create visualizations for per-class metrics"""
    
    metrics = ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Sort by metric value
        df_sorted = df.sort_values(metric, ascending=True)
        
        # Create bar plot
        colors = plt.cm.viridis(np.linspace(0, 1, len(df_sorted)))
        bars = ax.barh(df_sorted['Class'], df_sorted[metric], color=colors)
        
        ax.set_xlabel(metric, fontsize=12)
        ax.set_title(f'{metric} by Class', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', 
                   ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plot_path = save_dir / 'per_class_metrics.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics plot saved to: {plot_path}")
    plt.close()


def plot_confusion_matrix(model_path, data_yaml, save_dir='outputs/evaluation'):
    """Generate and plot confusion matrix"""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model = YOLO(model_path)
    
    # Validate to generate confusion matrix
    results = model.val(
        data=data_yaml,
        plots=True,
        save_json=True,
        conf=0.25
    )
    
    print(f"✓ Confusion matrix plots saved to validation results directory")


def analyze_errors(model_path, data_yaml, output_dir='outputs/error_analysis', 
                   conf_threshold=0.25, top_n=20):
    """
    Analyze model errors - find images with lowest confidence or most false positives
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = YOLO(model_path)
    
    print("Running error analysis...")
    print("This will help identify problematic images for further investigation")
    
    # Run validation with detailed output
    results = model.val(
        data=data_yaml,
        save_json=True,
        save_hybrid=True,
        conf=conf_threshold
    )
    
    print(f"\n✓ Error analysis complete")
    print(f"Check validation results for detailed per-image metrics")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate YOLO model')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    parser.add_argument('--data', type=str, default='data/processed/dataset.yaml', 
                       help='Path to dataset YAML')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--output', type=str, default='outputs/evaluation',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        split=args.split,
        save_dir=args.output
    )