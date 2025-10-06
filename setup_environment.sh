#!/bin/bash

echo "Creating virtual environment..."
python3 -m venv epri_venv

echo "Activating virtual environment..."
source epri_venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

echo "Setup complete!"
echo "To activate the environment, run: source epri_venv/bin/activate"