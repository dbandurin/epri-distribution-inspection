"""
Generate all project files for EPRI Distribution Inspection
Run this script to create the complete project structure
"""
import os
from pathlib import Path

# Create project structure
def create_directory_structure():
    """Create all necessary directories"""
    directories = [
        'data/raw/images',
        'data/raw/labels',
        'data/processed/train/images',
        'data/processed/train/labels',
        'data/processed/val/images',
        'data/processed/val/labels',
        'data/processed/test/images',
        'data/processed/test/labels',
        'models',
        'outputs/predictions',
        'outputs/visualizations',
        'outputs/evaluation',
        'outputs/exploration',
        'logs',
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")

print("=" * 70)
print("EPRI Distribution Inspection - Project Setup")
print("=" * 70)
print("\nCreating directory structure...")
create_directory_structure()

print("\n" + "=" * 70)
print("Directory structure created successfully!")
print("=" * 70)
print("\nNext steps:")
print("1. Copy all Python files from the artifacts panel to this directory")
print("2. Create requirements.txt with the dependencies shown in artifacts")
print("3. Create setup_environment.sh with the setup script from artifacts")
print("4. Run: chmod +x setup_environment.sh")
print("5. Run: ./setup_environment.sh")
print("6. Run: source epri_venv/bin/activate")
print("7. Run: python run_pipeline.py --full")
print("=" * 70)