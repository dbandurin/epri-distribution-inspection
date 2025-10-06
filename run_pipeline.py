"""
Complete pipeline runner for EPRI Distribution Inspection project
Run all steps from data download to model evaluation
"""
import sys
import subprocess
from pathlib import Path
import argparse


def run_command(cmd, description):
    """Run a command and handle errors"""
    print("\n" + "=" * 70)
    print(f"STEP: {description}")
    print("=" * 70)
    print(f"Running: {cmd}\n")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error in {description}")
        print(f"Command failed with exit code {e.returncode}")
        return False


def check_file_exists(filepath, description):
    """Check if a required file exists"""
    if Path(filepath).exists():
        print(f"✓ Found: {description} ({filepath})")
        return True
    else:
        print(f"✗ Missing: {description} ({filepath})")
        return False


def run_full_pipeline(skip_download=False, skip_training=False, 
                      model_size='m', epochs=100, batch_size=16):
    """
    Run the complete pipeline
    
    Args:
        skip_download: Skip data download if already downloaded
        skip_training: Skip training if model already exists
        model_size: YOLO model size (n, s, m, l, x)
        epochs: Number of training epochs
        batch_size: Training batch size
    """
    
    print("\n" + "=" * 70)
    print("EPRI DISTRIBUTION INSPECTION - COMPLETE PIPELINE")
    print("=" * 70)
    
    steps_completed = []
    steps_failed = []
    
    # Step 1: Download data
    if not skip_download:
        if run_command("python download_data.py", "Data Download"):
            steps_completed.append("Data Download")
        else:
            steps_failed.append("Data Download")
            print("\nPipeline stopped due to error in data download")
            return False
    else:
        print("\n⊳ Skipping data download (already downloaded)")
        steps_completed.append("Data Download (skipped)")
    
    # Verify data exists
    if not check_file_exists("data/raw/Overhead-Distribution-Labels.csv", "Labels CSV"):
        print("\nError: Required data files not found")
        return False
    
    # Step 2: Explore dataset
    if run_command("python explore_data.py", "Dataset Exploration"):
        steps_completed.append("Dataset Exploration")
    else:
        steps_failed.append("Dataset Exploration")
        # Non-critical, continue
    
    # Step 3: Parse labels and prepare dataset
    if run_command("python parse_labels.py", "Label Parsing & Dataset Preparation"):
        steps_completed.append("Label Parsing")
    else:
        steps_failed.append("Label Parsing")
        print("\nPipeline stopped due to error in label parsing")
        return False
    
    # Verify dataset prepared
    if not check_file_exists("data/processed/dataset.yaml", "Dataset YAML"):
        print("\nError: Dataset not properly prepared")
        return False
    
    # Step 4: Train model
    model_path = "models/epri_distribution/weights/best.pt"
    
    if not skip_training or not Path(model_path).exists():
        # Modify training script parameters
        train_cmd = f"python train_yolo.py"
        # Note: For custom parameters, edit train_yolo.py directly or pass via env vars
        
        if run_command(train_cmd, f"Model Training (YOLOv8{model_size})"):
            steps_completed.append("Model Training")
        else:
            steps_failed.append("Model Training")
            print("\nPipeline stopped due to error in training")
            return False
    else:
        print(f"\n⊳ Skipping training (model exists at {model_path})")
        steps_completed.append("Model Training (skipped)")
    
    # Verify model exists
    if not check_file_exists(model_path, "Trained Model"):
        print("\nError: Trained model not found")
        return False
    
    # Step 5: Evaluate model
    eval_cmd = f"python evaluate_model.py --model {model_path}"
    if run_command(eval_cmd, "Model Evaluation"):
        steps_completed.append("Model Evaluation")
    else:
        steps_failed.append("Model Evaluation")
        # Non-critical, continue
    
    # Step 6: Visualize samples
    vis_cmd = f"python visualize_samples.py --model {model_path} --samples 10"
    if run_command(vis_cmd, "Sample Visualization"):
        steps_completed.append("Sample Visualization")
    else:
        steps_failed.append("Sample Visualization")
        # Non-critical, continue
    
    # Step 7: Run inference on validation set (sample)
    inf_cmd = f"python inference.py --model {model_path} --source data/processed/val/images --output outputs/predictions"
    print("\n⊳ Running sample inference (this may take a while)...")
    if run_command(inf_cmd, "Inference on Validation Set"):
        steps_completed.append("Inference")
    else:
        steps_failed.append("Inference")
        # Non-critical, continue
    
    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    
    print(f"\n✓ Completed Steps ({len(steps_completed)}):")
    for step in steps_completed:
        print(f"  • {step}")
    
    if steps_failed:
        print(f"\n✗ Failed Steps ({len(steps_failed)}):")
        for step in steps_failed:
            print(f"  • {step}")
    
    print("\n" + "=" * 70)
    print("KEY OUTPUTS")
    print("=" * 70)
    print(f"  • Trained Model: {model_path}")
    print(f"  • Dataset Statistics: outputs/exploration/")
    print(f"  • Evaluation Metrics: outputs/evaluation/")
    print(f"  • Visualizations: outputs/visualizations/")
    print(f"  • Predictions: outputs/predictions/")
    
    print("\n" + "=" * 70)
    
    if not steps_failed:
        print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    else:
        print("⚠ PIPELINE COMPLETED WITH SOME ERRORS")
    
    print("=" * 70 + "\n")
    
    return len(steps_failed) == 0


def quick_inference(model_path, image_path):
    """Quick inference on a single image"""
    
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train a model first using: python train_yolo.py")
        return False
    
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        return False
    
    cmd = f"python inference.py --model {model_path} --source {image_path}"
    return run_command(cmd, "Quick Inference")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='EPRI Distribution Inspection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python run_pipeline.py --full
  
  # Run without downloading (data already downloaded)
  python run_pipeline.py --full --skip-download
  
  # Run without training (use existing model)
  python run_pipeline.py --full --skip-training
  
  # Quick inference on single image
  python run_pipeline.py --inference --model models/best.pt --image test.jpg
  
  # Custom training parameters
  python run_pipeline.py --full --model-size l --epochs 150 --batch 8
        """
    )
    
    parser.add_argument('--full', action='store_true',
                       help='Run full pipeline')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip data download step')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training step (use existing model)')
    parser.add_argument('--model-size', type=str, default='m',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLO model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Training batch size')
    
    # Quick inference mode
    parser.add_argument('--inference', action='store_true',
                       help='Quick inference mode')
    parser.add_argument('--model', type=str,
                       help='Path to model for inference')
    parser.add_argument('--image', type=str,
                       help='Path to image for inference')
    
    args = parser.parse_args()
    
    if args.inference:
        if not args.model or not args.image:
            print("Error: --model and --image required for inference mode")
            parser.print_help()
            sys.exit(1)
        
        success = quick_inference(args.model, args.image)
        sys.exit(0 if success else 1)
    
    elif args.full:
        success = run_full_pipeline(
            skip_download=args.skip_download,
            skip_training=args.skip_training,
            model_size=args.model_size,
            epochs=args.epochs,
            batch_size=args.batch
        )
        sys.exit(0 if success else 1)
    
    else:
        parser.print_help()
        sys.exit(0)