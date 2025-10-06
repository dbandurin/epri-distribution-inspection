"""
Visualize sample images with ground truth and predictions side by side
"""
import cv2
import numpy as np
import json
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import random

CLASS_NAMES = [
    'conductor', 'other_wire', 'pole', 'crossarm', 
    'insulator', 'cutouts', 'transformer', 'background_structure'
]

CLASS_COLORS = {
    0: (255, 0, 0),      # conductor - blue
    1: (0, 255, 255),    # other_wire - yellow
    2: (0, 165, 255),    # pole - orange
    3: (0, 255, 0),      # crossarm - green
    4: (255, 0, 255),    # insulator - magenta
    5: (255, 255, 0),    # cutouts - cyan
    6: (0, 0, 255),      # transformer - red
    7: (128, 128, 128),  # background_structure - gray
}


def draw_ground_truth(image, label_path):
    """Draw ground truth annotations from YOLO format label file"""
    
    vis_image = image.copy()
    h, w = image.shape[:2]
    
    if not Path(label_path).exists():
        return vis_image
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    overlay = vis_image.copy()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        
        class_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
        
        # Convert normalized coordinates to pixel coordinates
        points = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * w)
            y = int(coords[i + 1] * h)
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        
        # Draw polygon
        if len(points) > 2:
            cv2.fillPoly(overlay, [points], color)
            cv2.polylines(vis_image, [points], True, color, 2)
        elif len(points) == 2:
            # Draw line
            cv2.line(vis_image, tuple(points[0]), tuple(points[1]), color, 2)
    
    # Blend overlay
    vis_image = cv2.addWeighted(vis_image, 0.7, overlay, 0.3, 0)
    
    return vis_image


def draw_predictions(image, model, conf_threshold=0.25):
    """Draw model predictions"""
    
    results = model.predict(
        source=image,
        conf=conf_threshold,
        verbose=False
    )[0]
    
    vis_image = image.copy()
    
    if results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        boxes = results.boxes.data.cpu().numpy()
        
        overlay = vis_image.copy()
        
        for mask, box in zip(masks, boxes):
            class_id = int(box[5])
            confidence = box[4]
            
            # Resize mask
            mask_resized = cv2.resize(mask, (vis_image.shape[1], vis_image.shape[0]))
            mask_bool = mask_resized > 0.5
            
            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array(color) * 0.5
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"{CLASS_NAMES[class_id]} {confidence:.2f}"
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        vis_image = cv2.addWeighted(vis_image, 0.7, overlay, 0.3, 0)
    
    return vis_image


def visualize_comparison(model_path, images_dir, labels_dir, output_dir, 
                         num_samples=10, conf_threshold=0.25):
    """
    Create side-by-side comparison of ground truth and predictions
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Get all image files
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.JPG'))
    
    if len(image_files) == 0:
        print(f"No images found in {images_dir}")
        return
    
    # Sample random images
    sample_images = random.sample(image_files, min(num_samples, len(image_files)))
    
    print(f"Visualizing {len(sample_images)} samples...")
    
    for img_path in sample_images:
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # Get label path
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        # Draw ground truth
        gt_image = draw_ground_truth(image, label_path)
        
        # Draw predictions
        pred_image = draw_predictions(image, model, conf_threshold)
        
        # Create side-by-side comparison
        h, w = image.shape[:2]
        comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
        comparison[:, :w] = image
        comparison[:, w:2*w] = gt_image
        comparison[:, 2*w:] = pred_image
        
        # Add titles
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Ground Truth", (w + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Prediction", (2*w + 10, 30), font, 1, (255, 255, 255), 2)
        
        # Save
        output_path = output_dir / f"comparison_{img_path.stem}.jpg"
        cv2.imwrite(str(output_path), comparison)
        print(f"✓ Saved: {output_path.name}")
    
    print(f"\n✓ Visualizations saved to: {output_dir}")


def create_grid_visualization(model_path, images_dir, labels_dir, output_path,
                               num_samples=9, conf_threshold=0.25):
    """
    Create a grid of sample predictions
    """
    
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Get sample images
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.JPG'))
    sample_images = random.sample(image_files, min(num_samples, len(image_files)))
    
    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    axes = axes.flatten()
    
    for idx, img_path in enumerate(sample_images):
        # Read and predict
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        pred_image = draw_predictions(image, model, conf_threshold)
        pred_image_rgb = cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB)
        
        # Display
        axes[idx].imshow(pred_image_rgb)
        axes[idx].set_title(img_path.name, fontsize=8)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(sample_images), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Grid visualization saved to: {output_path}")
    plt.close()


def visualize_class_examples(model_path, images_dir, labels_dir, output_dir,
                              samples_per_class=3, conf_threshold=0.25):
    """
    Visualize examples for each class
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Find images for each class
    class_images = {cls: [] for cls in range(len(CLASS_NAMES))}
    
    for label_file in labels_dir.glob('*.txt'):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            
            class_id = int(parts[0])
            img_path = images_dir / f"{label_file.stem}.jpg"
            
            if not img_path.exists():
                img_path = images_dir / f"{label_file.stem}.JPG"
            
            if img_path.exists() and img_path not in class_images[class_id]:
                class_images[class_id].append(img_path)
    
    # Visualize samples for each class
    for class_id, class_name in enumerate(CLASS_NAMES):
        images = class_images[class_id][:samples_per_class]
        
        if not images:
            print(f"No images found for class: {class_name}")
            continue
        
        fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
        if len(images) == 1:
            axes = [axes]
        
        fig.suptitle(f"Class: {class_name}", fontsize=16, fontweight='bold')
        
        for idx, img_path in enumerate(images):
            image = cv2.imread(str(img_path))
            pred_image = draw_predictions(image, model, conf_threshold)
            pred_image_rgb = cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB)
            
            axes[idx].imshow(pred_image_rgb)
            axes[idx].set_title(img_path.name, fontsize=8)
            axes[idx].axis('off')
        
        plt.tight_layout()
        output_path = output_dir / f"class_{class_name}_examples.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
        plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize model predictions')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--images', type=str, default='data/processed/val/images',
                       help='Directory containing images')
    parser.add_argument('--labels', type=str, default='data/processed/val/labels',
                       help='Directory containing labels')
    parser.add_argument('--output', type=str, default='outputs/visualizations',
                       help='Output directory')
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--mode', type=str, default='comparison',
                       choices=['comparison', 'grid', 'class'],
                       help='Visualization mode')
    
    args = parser.parse_args()
    
    if args.mode == 'comparison':
        visualize_comparison(
            model_path=args.model,
            images_dir=args.images,
            labels_dir=args.labels,
            output_dir=args.output,
            num_samples=args.samples,
            conf_threshold=args.conf
        )
    elif args.mode == 'grid':
        output_path = Path(args.output) / 'grid_visualization.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        create_grid_visualization(
            model_path=args.model,
            images_dir=args.images,
            labels_dir=args.labels,
            output_path=output_path,
            num_samples=args.samples,
            conf_threshold=args.conf
        )
    elif args.mode == 'class':
        visualize_class_examples(
            model_path=args.model,
            images_dir=args.images,
            labels_dir=args.labels,
            output_dir=args.output,
            samples_per_class=3,
            conf_threshold=args.conf
        )