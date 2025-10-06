#!/bin/bash

# EPRI Distribution Inspection - Docker Startup Script

echo "======================================================================"
echo "EPRI Distribution Inspection - Docker Setup"
echo "======================================================================"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed!"
    echo "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

echo "✓ Docker is installed"

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "Docker is not running!"
    echo "Please start Docker Desktop and try again"
    exit 1
fi

echo "✓ Docker is running"
echo ""

# Check for GPU support
if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "✓ GPU support detected"
    USE_GPU=true
else
    echo "⚠ No GPU detected, will use CPU mode"
    echo "  (Training will be slower)"
    USE_GPU=false
fi

echo ""
echo "======================================================================"
echo "Building Docker Image..."
echo "======================================================================"
echo ""

if [ "$USE_GPU" = true ]; then
    echo "Building GPU image..."
    docker-compose build epri-training
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Docker image built successfully!"
        echo ""
        echo "======================================================================"
        echo "Starting Container..."
        echo "======================================================================"
        docker-compose up -d epri-training
        
        echo ""
        echo "✓ Container started!"
        echo ""
        echo "======================================================================"
        echo "Quick Start Commands:"
        echo "======================================================================"
        echo ""
        echo "Enter container:"
        echo "  docker-compose exec epri-training bash"
        echo ""
        echo "Or use make commands:"
        echo "  make shell          # Enter container"
        echo "  make download       # Download dataset"
        echo "  make prepare        # Prepare data"
        echo "  make train          # Train model"
        echo ""
        echo "To stop:"
        echo "  docker-compose down"
        echo ""
    else
        echo "Failed to build Docker image!"
        exit 1
    fi
else
    echo "Building CPU image..."
    docker-compose --profile cpu build epri-training-cpu
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Docker image built successfully!"
        echo ""
        echo "======================================================================"
        echo "Starting Container..."
        echo "======================================================================"
        docker-compose --profile cpu up -d epri-training-cpu
        
        echo ""
        echo "✓ Container started!"
        echo ""
        echo "======================================================================"
        echo "Quick Start Commands:"
        echo "======================================================================"
        echo ""
        echo "Enter container:"
        echo "  docker-compose --profile cpu exec epri-training-cpu bash"
        echo ""
        echo "Or use make commands:"
        echo "  make shell-cpu      # Enter container"
        echo "  make download       # Download dataset"
        echo ""
        echo "⚠ Note: CPU training will be much slower than GPU (10-50x)"
        echo ""
        echo "To stop:"
        echo "  docker-compose down"
        echo ""
    else
        echo "Failed to build Docker image!"
        exit 1
    fi
fi

echo "======================================================================"