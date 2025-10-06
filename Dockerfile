FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/workspace/.torch

# Create necessary directories
RUN mkdir -p /workspace/data/raw/images \
    /workspace/data/processed \
    /workspace/models \
    /workspace/outputs \
    /workspace/logs

# Set proper permissions
RUN chmod -R 777 /workspace

# Default command
CMD ["/bin/bash"]