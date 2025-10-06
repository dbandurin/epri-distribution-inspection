"""
Parse EPRI Distribution Labels CSV and convert to YOLO format
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import yaml

# Class definitions
CLASSES = {
    'conductor': 0,
    'other_wire': 1,
    'pole': 2,
    'crossarm': 3,
    'insulator': 4,
    'cutouts': 5,
    'transformer': 6,
    'background_structure': 7
}

CLASS_NAMES = list(CLASSES.keys())

def parse_annotation(label_json):
    """Parse a single annotation JSON object"""
    try:
        data = eval(label_json)
        annotations = []
        
        for obj in data.get('objects', []):
            class_name = obj.get('value', '').lower().replace(' ', '_')
            
            if class_name not in CLASSES:
                continue
            
            class_id = CLASSES[class_name]
            
            # Handle polygon annotations
            if 'polygon' in obj:
                points = obj['polygon']
                coords = [(p['x'], p['y']) for p in points]
                annotations.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'type': 'polygon',
                    'coordinates': coords
                })
            
            # Handle line annotations
            elif 'line' in obj:
                points = obj['line']
                coords = [(p['x'], p['y']) for p in points]
                annotations.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'type': 'line',
                    'coordinates': coords
                })
        
        return annotations
    
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return []

def convert_to_yolo_format(annotations, img_width, img_height):
    """Convert annotations to YOLO segmentation format"""
    yolo_annotations = []
    
    for ann in annotations:
        coords = ann['coordinates']
        
        if ann['type'] == 'polygon' or ann['type'] == 'line':
            normalized_coords = []
            for x, y in coords:
                normalized_coords.extend([
                    x / img_width,
                    y / img_height
                ])
            
            yolo_line = f"{ann['class_id']} " + " ".join(map(str, normalized_coords))
            yolo_annotations.append(yolo_line)
    
    return yolo_annotations

def process_labels(csv_path, images_dir, output_dir, train_split=0.8):
    """Process all labels and create YOLO format dataset"""
    
    print("Loading labels CSV...")
    df = pd.read_csv(csv_path)
    
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    
    train_img_dir = output_dir / "train" / "images"
    train_lbl_dir = output_dir / "train" / "labels"
    val_img_dir = output_dir / "val" / "images"
    val_lbl_dir = output_dir / "val" / "labels"
    
    for dir_path in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * train_split)
    
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    print(f"Total images: {len(df)}")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    print("\\nProcessing training set...")
    process_split(train_df, images_dir, train_img_dir, train_lbl_dir)
    
    print("\\nProcessing validation set...")
    process_split(val_df, images_dir, val_img_dir, val_lbl_dir)
    
    create_dataset_yaml(output_dir)
    
    print("\\n✓ Dataset processing complete!")

def process_split(df, images_dir, output_img_dir, output_lbl_dir):
    """Process a single split"""
    processed = 0
    skipped = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_filename = row['External ID']
        label_json = row['Label']
        
        image_path = None
        for ext in ['.JPG', '.jpg', '.png', '.PNG']:
            potential_path = images_dir / image_filename.replace('.JPG', ext).replace('.jpg', ext)
            if potential_path.exists():
                image_path = potential_path
                break
        
        if not image_path or not image_path.exists():
            skipped += 1
            continue
        
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            skipped += 1
            continue
        
        annotations = parse_annotation(label_json)
        
        if not annotations:
            skipped += 1
            continue
        
        yolo_annotations = convert_to_yolo_format(annotations, img_width, img_height)
        
        label_filename = image_path.stem + '.txt'
        label_path = output_lbl_dir / label_filename
        
        with open(label_path, 'w') as f:
            f.write('\\n'.join(yolo_annotations))
        
        output_image_path = output_img_dir / image_path.name
        
        try:
            if not output_image_path.exists():
                output_image_path.symlink_to(image_path.absolute())
        except Exception:
            import shutil
            shutil.copy2(image_path, output_image_path)
        
        processed += 1
    
    print(f"Processed: {processed}, Skipped: {skipped}")

def create_dataset_yaml(output_dir):
    """Create YAML configuration file for YOLO"""
    
    yaml_content = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(CLASSES),
        'names': CLASS_NAMES
    }
    
    yaml_path = output_dir / 'dataset.yaml'
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"\\n✓ Created dataset configuration: {yaml_path}")

if __name__ == "__main__":
    csv_path = "data/raw/Overhead-Distribution-Labels.csv"
    images_dir = "data/raw/images"
    output_dir = "data/processed"
    
    process_labels(csv_path, images_dir, output_dir, train_split=0.8)