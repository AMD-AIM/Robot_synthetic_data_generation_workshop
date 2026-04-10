[English](README.md) | [中文](README-cn.md)

# Robot Synthetic Data Generation Workshop

End-to-end pipeline for robot manipulation on **AMD GPUs (ROCm)**: **Synthetic Data Generation → VLA Training → Simulation Evaluation**.

Verified on **CDNA3 (MI300X)** and **RDNA4 (Radeon AI PRO R9700)**.

```
┌──────────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│ 01_gen_data.py (default) │     │  02_train_vla.py     │     │  03_eval.py          │
│   flat plane + cube      │     │                      │     │                      │
│ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│────▶│  SmolVLA fine-tune   │────▶│  Closed-loop eval    │
│ 02_gen_data_custom_scene │     │  on LeRobot dataset  │     │  in Genesis sim      │
│   kitchen GLB + anchors  │     │  HF checkpoint out   │     │  success rate + video│
└──────────────────────────┘     └─────────────────────┘     └─────────────────────┘
     Franka 7-DOF                     lerobot/smolvla_base       render → VLA → PD
     pick red cube                    freeze vision encoder      action chunking
     2 cameras (up/side)              train expert + state_proj  randomized cube pos
```

---

## Pre-generated Dataset (Quick Path)

A 100-episode flat-scene dataset is available on HuggingFace. You can skip data generation and jump straight to training:

```bash
pip install lerobot==0.4.4 torchcodec

python scripts/02_train_vla.py \
  --dataset-id lidavidsh/franka-pick-100ep-genesis \
  --n-steps 2000 --batch-size 4 --num-workers 4 \
  --save-dir outputs/smolvla_genesis
```

Or load it directly in Python:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("lidavidsh/franka-pick-100ep-genesis")
print(f"Episodes: {dataset.meta.total_episodes}, Frames: {len(dataset)}")
```

| Item | Value |
|------|-------|
| HuggingFace | [`lidavidsh/franka-pick-100ep-genesis`](https://huggingface.co/datasets/lidavidsh/franka-pick-100ep-genesis) |
| Format | LeRobot v3.0, AV1 video |
| Episodes / Frames | 100 / 13,500 |
| Cameras | 2 (up + side), 640×480 |
| Size | ~80 MB |
| Generated on | RDNA4 (Radeon AI PRO R9700), Genesis 0.4.5, seed=42 |

---

## Quick Start (Full Pipeline)

The entire workshop is driven by `workshop_pipeline.ipynb`, running inside a Docker container on a remote AMD GPU node.

The notebook ships with pre-generated visualizations (in `images/`), so you can read through the pipeline even without executing it.

### Step 1 — SSH into the GPU node

```bash
ssh -A <your-user>@<gpu-node>
```

### Step 2 — Clone the repository

```bash
git clone git@github.com:<org>/Robot_synthetic_data_generation_workshop.git
cd Robot_synthetic_data_generation_workshop
```

### Step 3 — Launch a Docker container

<details>
<summary><b>CDNA3 (MI300X) — ROCm 6.x</b></summary>

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -e PYOPENGL_PLATFORM=egl \
  -e HSA_OVERRIDE_GFX_VERSION=9.4.2 \
  -v $(pwd):/workspace/workshop \
  -v /tmp/workshop_output:/output \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace/workshop \
  <genesis-amd-docker-image> \
  bash
```

</details>

<details>
<summary><b>RDNA4 (R9700) — ROCm 7.2</b></summary>

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -e PYOPENGL_PLATFORM=egl \
  -v $(pwd):/workspace/workshop \
  -v /tmp/workshop_output:/output \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace/workshop \
  rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1 \
  bash
```

> RDNA4 does not need `HSA_OVERRIDE_GFX_VERSION` — ROCm 7.2 natively supports gfx1201.

</details>

> The `-it` flag gives you an interactive shell. All subsequent steps run inside this container.

### Step 4 — Install dependencies (inside the container)

```bash
# Python packages
pip install -q genesis-world lerobot==0.4.4 transformers accelerate safetensors \
  matplotlib Pillow jupyter ipykernel

# Fix numpy / scikit-image ABI mismatch (Genesis requires numpy==2.1.2)
pip install --force-reinstall --no-cache-dir -q "scikit-image>=0.22" "numpy==2.1.2"

# System packages for headless rendering and video encoding
apt-get update -qq && apt-get install -y -qq xvfb ffmpeg > /dev/null 2>&1

# Apply the Genesis ROCm patch (see "ROCm Adaptations" section below)
python patch_genesis_rocm.py
```

> You can also run `bash fix_and_run.sh` to do all of the above in one shot and execute the notebook automatically. However, we recommend launching Jupyter manually and running cells one by one to understand the pipeline.

### Step 5 — Start Jupyter Notebook

```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Open an SSH tunnel from your local machine:

```bash
ssh -L 8888:localhost:8888 <your-user>@<gpu-node>
```

Then open `http://localhost:8888` in your browser and navigate to `workshop_pipeline.ipynb`. Run cells in order.

---

## Notebook Overview

| Section | Content | Output |
|---------|---------|--------|
| **0. Environment Setup** | GPU detection, dependency check, kitchen asset download | Environment ready |
| **1. Data Generation** | Flat-scene + kitchen-scene IK trajectory generation | LeRobot dataset + visualizations |
| **2. VLA Training** | SmolVLA post-training (frozen vision encoder) | Checkpoint + loss curve |
| **3. Evaluation** | Closed-loop sim eval (render → infer → execute → physics) | Success rate + videos |
| **4. Summary** | Artifact collection and inline display | PNG / MP4 / JSON |

Each section includes:
- **Background** — why this step matters and the underlying technical principles
- **Executable code** — run cells directly
- **Embedded visualizations** — pre-generated images plus live matplotlib plots at runtime

---

## File Structure

```
robot_synthetic_data_generation_workshop/
├── README.md                        ← this file (English)
├── README-cn.md                     ← Chinese version
├── workshop_pipeline.ipynb          ← ★ Jupyter Notebook (workshop main body)
├── fix_and_run.sh                   ← one-shot: install deps + ROCm patches + run notebook
├── run_pipeline.sh                  ← shell-only pipeline (no notebook)
├── patch_genesis_rocm.py            ← Genesis ROCm patch script
├── images/                          ← pre-generated visualizations (referenced by notebook)
│   ├── ep0_camera_views.png
│   ├── ep0_joint_trajectory.png
│   ├── cube_scatter_flat.png
│   └── cube_scatter_kitchen.png
├── scenes/
│   └── rustic_kitchen.json          ← kitchen scene config (anchors, mesh refs)
└── scripts/
    ├── 00_download_kitchen.py       ← download kitchen GLB assets
    ├── 01_gen_data.py               ← data generation (flat scene)
    ├── 02_gen_data_custom_scene.py  ← data generation (custom 3D scene)
    ├── 02_train_vla.py              ← SmolVLA post-training
    ├── 03_eval.py                   ← closed-loop eval (flat scene)
    ├── 04_eval_custom_scene.py      ← closed-loop eval (custom scene)
    ├── genesis_scene_utils.py       ← Genesis utility functions
    ├── pick_common.py               ← scene-agnostic pick task builder
    └── scene_placement.py           ← robot-local coordinate utilities
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `genesis-world` | 0.4.5 or main | Physics simulation + rendering (Taichi backend, ROCm native). Main branch removes `cuda.bindings` dependency. |
| `lerobot` | ≥0.4.4 | Dataset format + SmolVLA model |
| `torch` | ≥2.1 (ROCm) | Training and inference |
| `transformers` | ≥4.40 | SmolVLA backbone (Idefics3) |
| `accelerate` | latest | HuggingFace model loading |
| `numpy` | ==2.1.2 | Required by Genesis; must match scikit-image C extension ABI |
| `scikit-image` | ≥0.22 | Must be recompiled against numpy==2.1.2 |
| `xvfb` | system | Headless rendering (apt-get install) |
| `ffmpeg` | system | Video encoding (apt-get install) |

**Hardware**: AMD Instinct MI300X (ROCm 6.x) or AMD Radeon AI PRO R9700 (ROCm 7.2), ≥4 GB VRAM

---

## ROCm Adaptations

All fixes below are handled automatically by `fix_and_run.sh`. You only need to apply them manually if you set up the environment yourself.

### 1. Genesis `cuda.bindings` Patch

Genesis `<=0.4.5` calls `from cuda.bindings import runtime` to query GPU shared memory size. This module does not exist on ROCm. The patch wraps the call in a try-except and falls back to the LDS size (64 KB):

```bash
python patch_genesis_rocm.py
```

> **Note**: Genesis main branch ([`e807698`](https://github.com/Genesis-Embodied-AI/Genesis/commit/e807698b8aa773fad3a6dfb4556889b251c30924), 2026-04-09) has replaced `cuda.bindings` with Taichi's native `qd.lang.impl.get_max_shared_memory_bytes()`, so this patch is **no longer needed** when installing from main:
>
> ```bash
> pip install git+https://github.com/Genesis-Embodied-AI/Genesis.git@main
> ```

### Rendering Backend: CDNA3 vs RDNA4

| Architecture | EGL Renderer | Type |
|---|---|---|
| CDNA3 (MI300X) | llvmpipe | CPU software rasterization |
| RDNA4 (R9700) | radeonsi | **GPU hardware rasterization** |

CDNA3 lacks a graphics pipeline, so Genesis camera rendering falls back to `llvmpipe` (CPU). RDNA4 has a full graphics pipeline and uses `radeonsi` for hardware-accelerated rendering with zero code changes — this is the primary source of the 3.4–4.4× data generation speedup.

### 2. numpy / scikit-image ABI Fix

The base Docker image may ship scikit-image compiled against a different numpy version, causing `ValueError: numpy.dtype size changed` at runtime. Force-reinstalling both packages together recompiles the C extensions:

```bash
pip install --force-reinstall --no-cache-dir "scikit-image>=0.22" "numpy==2.1.2"
```

### 3. ROCm-specific Script Flags

**CDNA3 (MI300X)** — PNG mode (avoids torchcodec CUDA dependency):

```bash
python scripts/01_gen_data.py --no-bbox-detection --no-videos ...
python scripts/02_train_vla.py --num-workers 0 ...
python scripts/03_eval.py --no-bbox-detection ...
```

**RDNA4 (R9700)** — Video mode (after building torchcodec from source):

```bash
python scripts/01_gen_data.py --no-bbox-detection ...
python scripts/02_train_vla.py --num-workers 4 ...
python scripts/03_eval.py --no-bbox-detection ...
```

- `--no-bbox-detection` — bypass bounding box detection compatibility issues on AMD GPUs
- `--num-workers 0` — avoid torchcodec multi-process decoding crashes (CDNA3 with pip torchcodec)
- `--no-videos` — store images as PNG instead of MP4 (use when torchcodec is not available)

### 4. torchcodec on ROCm (Video mode)

The pip-installed torchcodec binary links against CUDA libraries (`libnvrtc.so`, `libcudart.so`) and fails to import on ROCm. To enable video dataset format, build torchcodec 0.10.0 from source (CPU-only):

```bash
git clone --depth 1 --branch v0.10.0 https://github.com/pytorch/torchcodec.git /tmp/torchcodec
pip install pybind11
cmake /tmp/torchcodec -DENABLE_CUDA= \
  -DTorch_DIR=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")/Torch \
  -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())") \
  -DTORCHCODEC_DISABLE_COMPILE_WARNING_AS_ERROR=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc) && cmake --install .
```

### 5. SmolVLA Compatibility

`lerobot>=0.5.0` may have a `dataclass` field ordering issue with `SmolVLAConfig`. If training fails with `TypeError: non-default argument follows default argument`, use `lerobot==0.4.4` or check the [LeRobot releases](https://github.com/huggingface/lerobot/releases).

---

## Reference Results

### Data Generation

| Scene | Episodes | Success Rate | RDNA4 Time | CDNA3 Time |
|---|---|---|---|---|
| Flat (default) | 100 | **100%** | **~6.3 s/ep** | ~28 s/ep |
| Kitchen (custom) | 10 | **100%** | ~18-20 s/ep | — |

RDNA4 achieves **3.4–4.4× faster data generation** thanks to EGL hardware rasterization (radeonsi) vs CPU software rendering (llvmpipe) on CDNA3.

| Flat Scene | Kitchen Scene |
|:---:|:---:|
| ![flat](./images/cube_scatter_flat.png) | ![kitchen](./images/cube_scatter_kitchen.png) |

### Training (100 episodes, 2000 steps, batch 4)

| Metric | CDNA3 (MI300X) | RDNA4 Video nw=4 |
|---|---|---|
| Wall time | ~78 min | **~24.5 min** |
| Per-step | ~2.43 s/step | **~0.73 s/step** |
| GPU utilization | ~9.5% | **96%** |
| Loss (start → end) | 0.346 → 0.022 | 0.535 → 0.053 |
| Peak VRAM | 2.2 GB | 2.38 GB |

> CDNA3 result uses PNG format with `num-workers=0` (data-loading bottlenecked).
> RDNA4 result uses Video format with `num-workers=4` (GPU-compute bottlenecked).
> The 3× speedup on RDNA4 comes from eliminating the data-loading bottleneck via video decoding + parallel workers.

### Evaluation (100 episodes, 2000 steps)

| Eval Set | CDNA3 | RDNA4 |
|---|---|---|
| Unseen positions (seed=99) | 4/10 = 40% | 8/10 = 80% |
| Training positions (seed=42) | 5/10 = 50% | 6/10 = 60% |

> With only 2000 steps (~0.6 epochs), evaluation variance is high — results depend heavily on which subset of data is sampled during training. Increasing to 10K+ steps is expected to stabilize results.

---

## Data Flow

```
Genesis Scene                    LeRobot Dataset                SmolVLA
┌──────────────┐                ┌──────────────┐              ┌──────────────┐
│ Franka Panda │                │ observation   │              │ Vision       │
│ Red Cube     │──IK plan──────▶│  .state [9D]  │──train──────▶│ Encoder      │
│ 2 Cameras    │   joint lerp   │  .images.up   │              │ (frozen)     │
│              │   render       │  .images.side │              │              │
│ Physics sim  │                │ action [9D]   │              │ Expert       │
│ (Genesis)    │                │ task (text)   │              │ Layers       │
└──────────────┘                └──────────────┘              │ (trainable)  │
  ▲ scene source:                                             │              │
  │ (a) flat plane (01)                                       │ → action     │
  │ (b) kitchen GLB (02)         same LeRobot format          │   chunk [50] │
                                                              │              │
Eval Loop:                                                    │              │
  render ─────────────────────────────────── inference ───────│              │
  observe state ──────────────────────────── predict ─────────│              │
  execute action[0] ──────── PD control ──── scene.step()     └──────────────┘
```

---

## References

- [LeRobot](https://github.com/huggingface/lerobot) — Robot learning framework (dataset + policies)
- [Genesis](https://genesis-embodied-ai.github.io/) — GPU-accelerated physics simulation (ROCm native via Taichi)
- [SmolVLA](https://huggingface.co/blog/smolvla) — Vision-Language-Action model
- [World Labs Marble](https://marble.worldlabs.ai/) — 3D scene generation for custom environments
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
