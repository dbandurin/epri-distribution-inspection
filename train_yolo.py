"""
Train YOLO model for EPRI Distribution Inspection
"""
from ultralytics import YOLO
import torch

def train_model():
    MODEL_NAME = 'yolov8m-seg.pt'
    DATA_YAML = 'data/processed/dataset.yaml'
    EPOCHS = 100
    IMGSZ = 640
    BATCH_SIZE = 16
    DEVICE = '0' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("Training Configuration:")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    print("=" * 60)
    
    model = YOLO(MODEL_NAME)
    
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        device=DEVICE,
        project='models',
        name='epri_distribution',
        patience=50,
        save=True,
        plots=True,
        exist_ok=True,
        pretrained=True,
        verbose=True,
        seed=42,
    )
    
    print("\\nTraining complete!")
    return model, results

if __name__ == "__main__":
    train_model()