#!/bin/bash
set -e

echo "=== Step 1: Install genesis + lerobot ==="
pip install -q genesis-world lerobot transformers accelerate safetensors matplotlib Pillow jupyter nbconvert ipykernel 2>&1 | tail -5

echo "=== Step 2: Rebuild skimage with pinned numpy ==="
pip install --force-reinstall --no-cache-dir -q "scikit-image>=0.22" "numpy==2.1.2" 2>&1 | tail -5

echo "=== Step 3: Patch genesis for ROCm (cuda.bindings) ==="
RIGID_SOLVER="/opt/conda/envs/py_3.12/lib/python3.12/site-packages/genesis/engine/solvers/rigid/rigid_solver.py"
python - "$RIGID_SOLVER" << 'PATCH'
import sys
fpath = sys.argv[1]
with open(fpath) as f:
    code = f.read()
old = '''                elif gs.device.type == "cuda":
                    from cuda.bindings import runtime  # Transitive dependency of torch CUDA

                    _, max_shared_mem = runtime.cudaDeviceGetAttribute(
                        runtime.cudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerBlockOptin, gs.device.index
                    )
                    max_shared_mem /= 1024.0'''
new = '''                elif gs.device.type == "cuda":
                    try:
                        from cuda.bindings import runtime
                        _, max_shared_mem = runtime.cudaDeviceGetAttribute(
                            runtime.cudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerBlockOptin, gs.device.index
                        )
                        max_shared_mem /= 1024.0
                    except (ImportError, Exception):
                        max_shared_mem = 64.0  # ROCm fallback'''
if old in code:
    code = code.replace(old, new)
    with open(fpath, 'w') as f:
        f.write(code)
    print(f"Patched {fpath}")
else:
    print("Already patched or different version")
PATCH

echo "=== Step 4: Install system deps ==="
apt-get update -qq && apt-get install -y -qq xvfb ffmpeg > /dev/null 2>&1 || true

echo "=== Step 5: Verify ==="
python -c "import numpy; print(f'numpy: {numpy.__version__}')"
python -c "import lerobot; print(f'lerobot: {lerobot.__version__}')"

echo "=== Step 6: Clean old output ==="
rm -rf /output/*.ipynb /output/*.png /output/*.mp4 /output/*.json /output/outputs /output/eval_* /output/franka_gen_pick 2>/dev/null || true

echo "=== Step 7: Run notebook ==="
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=3600 \
  --ExecutePreprocessor.kernel_name=python3 \
  --allow-errors \
  --output /output/workshop_pipeline_executed.ipynb \
  workshop_pipeline.ipynb 2>&1

echo "=== Collecting artifacts ==="
find /output -name "*.png" -o -name "*.mp4" -o -name "*.json" 2>/dev/null | head -30 || true
ls -laR /output/ 2>/dev/null | head -80 || true
echo "=== DONE ==="
