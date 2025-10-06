"""
Explore and visualize EPRI dataset statistics
"""
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

CLASS_NAMES = [
    'conductor', 'other_wire', 'pole', 'crossarm', 
    'insulator', 'cutouts', 'transformer', 'background_structure'
]

def explore_dataset(csv_path, output_dir='outputs/exploration'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    df = pd.read_csv(csv_path)
    print(f"Total images: {len(df)}")
    
    class_counts = Counter()
    
    n = 0
    for idx, row in df.iterrows():
        n += 1
        if n%100 == 0:
            print('n = ',n)
        if n>300:
            break
        try:
            data = eval(row['Label'])
            print(data)
            for obj in data.get('objects', []):
                class_name = obj.get('value', '').lower().replace(' ', '_')
                print(row['External ID'],class_name)
                if class_name in CLASS_NAMES:
                    class_counts[class_name] += 1
        except:
            continue
    
    print("\\nClass Distribution:")
    for class_name in CLASS_NAMES:
        print(f"{class_name:20s}: {class_counts[class_name]:5d}")
    
    # Simple bar chart
    plt.figure(figsize=(12, 6))
    classes = list(CLASS_NAMES)
    counts = [class_counts[cls] for cls in classes]
    plt.bar(classes, counts)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.png')
    print(f"\\n✓ Saved: {output_dir / 'class_distribution.png'}")

if __name__ == "__main__":
    explore_dataset("data/raw/Overhead-Distribution-Labels.csv")