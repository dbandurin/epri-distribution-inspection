# EPRI Distribution Inspection - Object Detection & Segmentation

AI-powered automatic detection and segmentation of electric utility infrastructure from drone imagery using YOLOv8.

## Project Overview

This project trains a deep learning model to automatically identify and segment power distribution equipment in aerial drone images, including poles, insulators, transformers, conductors, and other utility assets.

**Dataset**: EPRI Distribution Inspection Imagery (29,620 labeled drone images)  
**Model**: YOLOv8 Segmentation  
**Framework**: PyTorch + Ultralytics  

## Features

- Automatic download from Azure Blob Storage
- JSON to YOLO format label conversion
- Dataset exploration and visualization
- YOLOv8 training with GPU support
- Model evaluation with per-class metrics
- Inference on new images
- Visualization of predictions

## Object Classes

The model detects 8 classes of utility equipment:

| Class | Type | Description |
|-------|------|-------------|
| conductor | Polyline | Power transmission lines |
| other_wire | Polyline | Secondary wires |
| pole | Polygon | Utility poles |
| crossarm | Polygon | Horizontal pole supports |
| insulator | Polygon | Electrical insulators |
| cutouts | Polygon | Protective switches |
| transformers | Polygon | Power transformers |
| background_structure | Polygon | Other structures |

## Project Structure

```
epri-distribution-inspection/
├── data/
│   ├── raw/
│   │   ├── images/              # Downloaded drone images
│   │   └── Overhead-Distribution-Labels.csv
│   └── processed/
│       ├── train/               # 80% training data
│       │   ├── images/
│       │   └── labels/          # YOLO format (.txt)
│       ├── val/                 # 20% validation data
│       │   ├── images/
│       │   └── labels/
│       └── dataset.yaml         # YOLO config
├── models/                      # Trained model weights
│   └── epri_distribution/
│       └── weights/
│           ├── best.pt          # Best checkpoint
│           └── last.pt
├── outputs/
│   ├── evaluation/              # Metrics & confusion matrix
│   ├── exploration/             # Dataset statistics
│   ├── predictions/             # Inference results
│   └── visualizations/          # Annotated images
├── logs/                        # Training logs
├── download_data.py             # Download from Azure
├── parse_labels.py              # Convert to YOLO format
├── explore_data.py              # Dataset analysis
├── train_yolo.py                # Training script
├── evaluate_model.py            # Model evaluation
├── inference.py                 # Run predictions
├── visualize_samples.py         # Visualization tools
├── decode_yolo_labels.py        # Understand YOLO format
├── run_pipeline.py              # Automated pipeline
├── requirements.txt             # Python dependencies
└── README.md
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv epri_venv
source epri_venv/bin/activate  # On Windows: epri_venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python download_data.py
```

Downloads from Azure Blob Storage:
- Circuit1.zip, Circuit10.zip (drone images)
- Overhead-Distribution-Labels.csv (annotations)

### 3. Explore Dataset (Optional)

```bash
python explore_data.py
```

Generates statistics and visualizations about the dataset.

### 4. Prepare YOLO Format

```bash
python parse_labels.py
```

Converts CSV annotations to YOLO segmentation format and creates train/val split.

### 5. Train Model

```bash
python train_yolo.py
```

**Training Configuration:**
- Model: YOLOv8m-seg (medium, balanced)
- Epochs: 100
- Batch size: 16
- Image size: 640x640
- Device: GPU (auto-detects) or CPU

**Training Time:**
- With GPU (RTX 3090): ~8 hours
- With CPU: ~80-400 hours

### 6. Evaluate Model

```bash
python evaluate_model.py --model models/epri_distribution/weights/best.pt
```

Outputs:
- Overall mAP@0.5, mAP@0.5:0.95
- Per-class precision, recall
- Confusion matrix
- Visualization plots

### 7. Run Inference

```bash
# Single image
python inference.py \
    --model models/epri_distribution/weights/best.pt \
    --source path/to/image.jpg \
    --output outputs/predictions

# Directory of images
python inference.py \
    --model models/epri_distribution/weights/best.pt \
    --source path/to/images/ \
    --output outputs/predictions
```

## Advanced Usage

### Automated Pipeline

Run everything with one command:

```bash
python run_pipeline.py --full
```

### Visualize Ground Truth vs Predictions

```bash
python visualize_samples.py \
    --model models/epri_distribution/weights/best.pt \
    --images data/processed/val/images \
    --labels data/processed/val/labels \
    --output outputs/visualizations \
    --samples 10
```

### Understand YOLO Format

```bash
# Explain YOLO format
python decode_yolo_labels.py --explain

# Visualize a specific label file
python decode_yolo_labels.py \
    --image "data/processed/train/images/1 (950).JPG" \
    --label "data/processed/train/labels/1 (950).txt" \
    --output "outputs/label_explained.jpg"
```

### Custom Training Parameters

Edit `train_yolo.py`:

```python
MODEL_NAME = 'yolov8l-seg.pt'  # Use larger model
EPOCHS = 150                    # Train longer
BATCH_SIZE = 8                  # Reduce if GPU memory issues
IMGSZ = 1280                    # Higher resolution
```

### Resume Training

```python
from ultralytics import YOLO

model = YOLO('models/epri_distribution/weights/last.pt')
model.train(resume=True, epochs=50)
```

### Export Model

```python
from ultralytics import YOLO

model = YOLO('models/epri_distribution/weights/best.pt')

# ONNX (cross-platform)
model.export(format='onnx')

# TensorRT (NVIDIA GPU acceleration)
model.export(format='engine')

# CoreML (iOS)
model.export(format='coreml')
```

## Requirements

### System Requirements
- **OS**: Linux, macOS, or Windows
- **Python**: 3.8-3.11
- **GPU**: NVIDIA GPU with CUDA (recommended) or CPU
- **RAM**: 16GB+ recommended
- **Storage**: 50GB+ for dataset and models

### Python Dependencies

See `requirements.txt` for full list. Key packages:
- PyTorch 2.0+
- Ultralytics (YOLOv8)
- OpenCV
- Pandas, NumPy, Matplotlib

## Model Performance

Expected results after training:

| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.75-0.85 |
| mAP@0.5:0.95 | 0.50-0.60 |
| Inference Speed (GPU) | ~20ms/image |
| Inference Speed (CPU) | ~200ms/image |

## YOLO Format Explanation

YOLO segmentation format (one line per object):
```
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

Where:
- `class_id`: 0-7 (object class)
- `x1 y1 ... xn yn`: Normalized polygon coordinates (0-1)

Example:
```
4 0.200 0.438 0.236 0.446 0.243 0.480
```
Means: Class 4 (insulator) with 3 polygon points at (20%, 44%), (24%, 45%), (24%, 48%) of image dimensions.

## Troubleshooting

### Out of GPU Memory
```bash
# Reduce batch size in train_yolo.py
BATCH_SIZE = 8  # or even 4
```

### Slow Training on CPU
```bash
# Check CUDA availability
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Use smaller model
MODEL_NAME = 'yolov8n-seg.pt'  # Nano (fastest)
```

### OpenCV Installation Issues (macOS)
Use Docker instead:
```bash
# See Docker setup in project files
docker-compose up -d
docker-compose exec epri-training bash
```

## Cost & Time Estimates

| Task | GPU (RTX 3090) | CPU |
|------|----------------|-----|
| Download data | 10-30 min | 10-30 min |
| Parse labels | 5-10 min | 5-10 min |
| Train (100 epochs) | 8 hours | 80-400 hours |
| Inference (1000 images) | 30 seconds | 5 minutes |

## Citation

If you use this dataset or code, please cite:

```bibtex
@dataset{epri_distribution_2023,
  title={EPRI Distribution Inspection Imagery},
  author={Electric Power Research Institute},
  year={2023},
  publisher={Kaggle},
  url={https://www.kaggle.com/datasets/dexterlewis/epri-distribution-inspection-imagery}
}
```

## License

Dataset: Check EPRI's terms on Kaggle  
Code: MIT License (or your chosen license)

## Resources

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com)
- [EPRI Kaggle Dataset](https://www.kaggle.com/datasets/dexterlewis/epri-distribution-inspection-imagery)
- [PyTorch Documentation](https://pytorch.org/docs)

## Contributing

Contributions welcome! Please open issues or pull requests.

## Support

For questions or issues:
1. Check existing GitHub issues
2. Review documentation
3. Open a new issue with details

---

**Note**: This project is for research and educational purposes. Ensure proper licensing and permissions before commercial use.