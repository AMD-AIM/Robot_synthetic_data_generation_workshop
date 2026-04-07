#!/bin/bash
set -e

echo "=== Installing deps ==="
pip install -q lerobot genesis-world transformers accelerate safetensors matplotlib Pillow jupyter nbconvert ipykernel 2>&1 | tail -10

echo "=== Installing Xvfb & ffmpeg ==="
apt-get update -qq && apt-get install -y -qq xvfb ffmpeg > /dev/null 2>&1 || true

echo "=== Running notebook ==="
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=3600 \
  --ExecutePreprocessor.kernel_name=python3 \
  --allow-errors \
  --output /output/workshop_pipeline_executed.ipynb \
  workshop_pipeline.ipynb 2>&1

echo "=== Collecting artifacts ==="
ls -la /output/ 2>/dev/null || true
find /output -name "*.png" -o -name "*.mp4" -o -name "*.json" 2>/dev/null | head -30 || true
echo "=== DONE ==="
