"""
Decode and visualize YOLO format labels
This helps understand what the numbers in YOLO labels mean
"""
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Class names matching parse_labels.py
CLASS_NAMES = [
    'conductor',           # 0
    'other_wire',         # 1
    'pole',               # 2
    'crossarm',           # 3
    'insulator',          # 4
    'cutouts',            # 5
    'transformers',       # 6
    'background_structure' # 7
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

def decode_yolo_line(line, img_width, img_height):
    """
    Decode a single line from YOLO format label file
    
    Args:
        line: "4 0.1999 0.4378 0.2362 0.4456 ..." (class_id + normalized coordinates)
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        dict with class_id, class_name, and pixel coordinates
    """
    parts = line.strip().split()
    
    class_id = int(parts[0])
    class_name = CLASS_NAMES[class_id]
    
    # Convert normalized coordinates to pixel coordinates
    coords = [float(x) for x in parts[1:]]
    
    pixel_coords = []
    for i in range(0, len(coords), 2):
        x_norm = coords[i]      # Normalized x (0-1)
        y_norm = coords[i + 1]  # Normalized y (0-1)
        
        # Convert to pixels
        x_pixel = int(x_norm * img_width)
        y_pixel = int(y_norm * img_height)
        
        pixel_coords.append((x_pixel, y_pixel))
    
    return {
        'class_id': class_id,
        'class_name': class_name,
        'normalized_coords': [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)],
        'pixel_coords': pixel_coords,
        'num_points': len(pixel_coords)
    }

def visualize_label_file(image_path, label_path, output_path=None):
    """
    Visualize a YOLO label file overlaid on its image
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    h, w = image.shape[:2]
    print(f"Image dimensions: {w}x{h}")
    
    # Read label file
    if not Path(label_path).exists():
        print(f"Error: Label file not found {label_path}")
        return
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    print(f"\nFound {len(lines)} annotations in label file")
    print("=" * 70)
    
    # Create visualization
    vis_image = image.copy()
    overlay = image.copy()
    
    for idx, line in enumerate(lines):
        # Decode the line
        annotation = decode_yolo_line(line, w, h)
        
        print(f"\nAnnotation {idx + 1}:")
        print(f"  Class ID: {annotation['class_id']}")
        print(f"  Class Name: {annotation['class_name']}")
        print(f"  Number of points: {annotation['num_points']}")
        print(f"  First point (normalized): ({annotation['normalized_coords'][0][0]:.4f}, {annotation['normalized_coords'][0][1]:.4f})")
        print(f"  First point (pixels): {annotation['pixel_coords'][0]}")
        
        # Get color
        color = CLASS_COLORS.get(annotation['class_id'], (255, 255, 255))
        
        # Draw polygon
        points = np.array(annotation['pixel_coords'], dtype=np.int32)
        
        if len(points) >= 3:
            # Fill polygon on overlay
            cv2.fillPoly(overlay, [points], color)
            # Draw outline on main image
            cv2.polylines(vis_image, [points], True, color, 2)
        elif len(points) == 2:
            # Draw line
            cv2.line(vis_image, tuple(points[0]), tuple(points[1]), color, 3)
        
        # Add label text
        if len(points) > 0:
            center = points.mean(axis=0).astype(int)
            label_text = f"{annotation['class_name']}"
            
            # Draw text with background
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis_image, 
                         (center[0] - 5, center[1] - text_h - 5),
                         (center[0] + text_w + 5, center[1] + 5),
                         color, -1)
            cv2.putText(vis_image, label_text,
                       (center[0], center[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Blend overlay with original
    vis_image = cv2.addWeighted(vis_image, 0.7, overlay, 0.3, 0)
    
    # Save or display
    if output_path:
        cv2.imwrite(str(output_path), vis_image)
        print(f"\n✓ Saved visualization to: {output_path}")
    else:
        # Display
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title(f"YOLO Labels Visualization: {Path(image_path).name}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return vis_image

def explain_yolo_format():
    """
    Print explanation of YOLO format
    """
    print("=" * 70)
    print("YOLO SEGMENTATION FORMAT EXPLAINED")
    print("=" * 70)
    print("""
Each line in a YOLO label file represents ONE object annotation:

FORMAT:
    class_id x1 y1 x2 y2 x3 y3 ... xn yn

WHERE:
    - class_id: Integer (0-7 in our case) representing object class
    - x1, y1, x2, y2, ...: Normalized polygon/line coordinates
    
NORMALIZATION:
    All coordinates are normalized to 0-1 range:
    - x_normalized = x_pixel / image_width
    - y_normalized = y_pixel / image_height
    
    This makes the annotations scale-independent!

EXAMPLE:
    Line: "4 0.2000 0.4379 0.2362 0.4456 0.2426 0.4796"
    
    Means:
    - Class 4 = "insulator"
    - 3 points defining a polygon:
        Point 1: 20% from left, 44% from top
        Point 2: 24% from left, 45% from top  
        Point 3: 24% from left, 48% from top

CLASS IDs IN OUR PROJECT:
    0 = conductor
    1 = other_wire
    2 = pole
    3 = crossarm
    4 = insulator
    5 = cutouts
    6 = transformers
    7 = background_structure

WHY THIS FORMAT?
    ✓ Scale-independent (works with any image size)
    ✓ Simple text format (easy to parse)
    ✓ One file per image (organized)
    ✓ Standard format (works with all YOLO models)
    """)
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Decode and visualize YOLO format labels',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--explain', action='store_true',
                       help='Print explanation of YOLO format')
    parser.add_argument('--image', type=str,
                       help='Path to image file')
    parser.add_argument('--label', type=str,
                       help='Path to label file')
    parser.add_argument('--output', type=str,
                       help='Output path for visualization')
    parser.add_argument('--example', action='store_true',
                       help='Run with example from dataset')
    
    args = parser.parse_args()
    
    if args.explain:
        explain_yolo_format()
    
    elif args.example:
        # Try to find an example
        image_path = Path("data/processed/train/images/1 (950).JPG")
        label_path = Path("data/processed/train/labels/1 (950).txt")
        
        if image_path.exists() and label_path.exists():
            print("Running with example from dataset...")
            visualize_label_file(image_path, label_path, 
                               output_path="outputs/yolo_format_explained.jpg")
        else:
            print("Example files not found. Please run parse_labels.py first.")
    
    elif args.image and args.label:
        visualize_label_file(args.image, args.label, args.output)
    
    else:
        parser.print_help()
        print("\nTry: python decode_yolo_labels.py --explain")
        print("Or:  python decode_yolo_labels.py --example")

'''
python decode_yolo_labels.py \
    --image "data/processed/train/images/1 (950).JPG" \
    --label "data/processed/train/labels/1 (950).txt" \
    --output "outputs/label_visualization.jpg"
'''
