"""
Inference pipeline for EPRI Distribution Inspection
"""
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json
from tqdm import tqdm

# Class names
CLASS_NAMES = [
    'conductor', 'other_wire', 'pole', 'crossarm', 
    'insulator', 'cutouts', 'transformer', 'background_structure'
]

# Colors for visualization (BGR format)
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


def load_model(model_path):
    """Load trained YOLO model"""
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    return model


def predict_image(model, image_path, conf_threshold=0.25, iou_threshold=0.45):
    """Run prediction on a single image"""
    
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Run inference
    results = model.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold,
        save=False,
        verbose=False
    )
    
    return results[0], image


def visualize_results(image, result, show_labels=True, show_conf=True):
    """Visualize detection results on image"""
    
    # Create a copy for visualization
    vis_image = image.copy()
    
    # Check if segmentation masks exist
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.data.cpu().numpy()
        
        # Create overlay
        overlay = vis_image.copy()
        
        for i, (mask, box) in enumerate(zip(masks, boxes)):
            class_id = int(box[5])
            confidence = box[4]
            
            # Resize mask to image size
            mask_resized = cv2.resize(mask, (vis_image.shape[1], vis_image.shape[0]))
            mask_bool = mask_resized > 0.5
            
            # Apply color overlay
            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array(color) * 0.5
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            if show_labels:
                label = CLASS_NAMES[class_id]
                if show_conf:
                    label += f" {confidence:.2f}"
                
                # Draw label background
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(
                    vis_image,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    vis_image,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
        
        # Blend overlay with original image
        vis_image = cv2.addWeighted(vis_image, 0.7, overlay, 0.3, 0)
    
    else:
        # Fallback to bounding boxes only
        boxes = result.boxes.data.cpu().numpy()
        
        for box in boxes:
            x1, y1, x2, y2, conf, class_id = box
            x1, y1, x2, y2, class_id = map(int, [x1, y1, x2, y2, class_id])
            
            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            
            # Draw box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            if show_labels:
                label = CLASS_NAMES[class_id]
                if show_conf:
                    label += f" {conf:.2f}"
                
                cv2.putText(
                    vis_image,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
    
    return vis_image


def process_directory(model, input_dir, output_dir, conf_threshold=0.25):
    """Process all images in a directory"""
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(input_dir.glob(ext))
    
    print(f"Found {len(image_files)} images to process")
    
    results_list = []
    
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Run prediction
            result, original_image = predict_image(model, img_path, conf_threshold)
            
            # Visualize
            vis_image = visualize_results(original_image, result)
            
            # Save visualization
            output_path = output_dir / f"{img_path.stem}_pred{img_path.suffix}"
            cv2.imwrite(str(output_path), vis_image)
            
            # Save detection results as JSON
            detections = {
                'image': str(img_path.name),
                'detections': []
            }
            
            if result.boxes is not None:
                boxes = result.boxes.data.cpu().numpy()
                for box in boxes:
                    detection = {
                        'class_id': int(box[5]),
                        'class_name': CLASS_NAMES[int(box[5])],
                        'confidence': float(box[4]),
                        'bbox': [float(x) for x in box[:4]]
                    }
                    detections['detections'].append(detection)
            
            results_list.append(detections)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Save all results to JSON
    results_json_path = output_dir / "predictions.json"
    with open(results_json_path, 'w') as f:
        json.dump(results_list, f, indent=2)
    
    print(f"\n✓ Processing complete!")
    print(f"Visualizations saved to: {output_dir}")
    print(f"Results JSON saved to: {results_json_path}")


def process_single_image(model_path, image_path, output_path=None, conf_threshold=0.25):
    """Process a single image and display/save result"""
    
    model = load_model(model_path)
    result, original_image = predict_image(model, image_path, conf_threshold)
    vis_image = visualize_results(original_image, result)
    
    # Display result
    cv2.imshow('Detection Result', vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save if output path provided
    if output_path:
        cv2.imwrite(output_path, vis_image)
        print(f"Saved result to: {output_path}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='EPRI Distribution Inspection Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--source', type=str, required=True, help='Image or directory path')
    parser.add_argument('--output', type=str, default='outputs/predictions', help='Output directory')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model)
    
    # Check if source is directory or single image
    source_path = Path(args.source)
    
    if source_path.is_dir():
        process_directory(model, args.source, args.output, args.conf)
    elif source_path.is_file():
        output_path = Path(args.output) / f"{source_path.stem}_pred{source_path.suffix}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        process_single_image(args.model, args.source, str(output_path), args.conf)
    else:
        print(f"Error: Source not found: {args.source}")