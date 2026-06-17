#!/bin/bash

# DriveSense environment setup script

# Exit on error
set -e

echo "=================================================="
echo "           DriveSense Setup Script"
echo "=================================================="

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "[-] Error: python3 is not installed or not in PATH."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[+] Found Python version $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[+] Creating virtual environment 'venv'..."
    python3 -m venv venv
else
    echo "[+] Virtual environment 'venv' already exists."
fi

# Activate virtual environment
echo "[+] Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "[+] Upgrading pip..."
pip install --upgrade pip

# Install dependencies
if [ -f "requirements.txt" ]; then
    echo "[+] Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "[-] Error: requirements.txt not found!"
    exit 1
fi

echo "[+] Verifying MediaPipe face landmarker model..."
if [ ! -f "models/face_landmarker.task" ]; then
    echo "[!] Warning: models/face_landmarker.task is missing!"
    echo "    Please download it from Google MediaPipe and place it in the 'models/' folder."
else
    echo "[+] MediaPipe face landmarker model found."
fi

echo "=================================================="
echo "[+] Setup completed successfully!"
echo "    To activate the environment in your terminal, run:"
echo "    source venv/bin/activate"
echo "=================================================="
