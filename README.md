[English](README.md) | [中文](README-cn.md)

# Robot Synthetic Data Generation Workshop

End-to-end pipeline for robot manipulation on **AMD MI300X (ROCm)**: **Synthetic Data Generation → VLA Training → Simulation Evaluation**.

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

## Quick Start

The entire workshop is driven by `workshop_pipeline.ipynb`, running inside a Docker container on a remote AMD MI300X GPU node.

The notebook ships with pre-generated visualizations (in `images/`), so you can read through the pipeline even without executing it.

### Step 1 — SSH into the GPU node

```bash
ssh -A <your-user>@<mi300x-node>
```

### Step 2 — Clone the repository

```bash
git clone git@github.com:<org>/Robot_synthetic_data_generation_workshop.git
cd Robot_synthetic_data_generation_workshop
```

### Step 3 — Launch a Docker container

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

> The `-it` flag gives you an interactive shell. All subsequent steps run inside this container.

### Step 4 — Install dependencies (inside the container)

```bash
# Python packages
pip install -q genesis-world lerobot transformers accelerate safetensors \
  matplotlib Pillow jupyter ipykernel

# Fix numpy / scikit-image ABI mismatch (Genesis requires numpy==2.1.2)
pip install --force-reinstall --no-cache-dir -q "scikit-image>=0.22" "numpy==2.1.2"

# System packages for headless rendering and video encoding
apt-get update -qq && apt-get install -y -qq xvfb ffmpeg > /dev/null 2>&1

# Apply the Genesis ROCm patch (see "ROCm Adaptations" section below)
# fix_and_run.sh Step 3 handles this automatically
```

> You can also run `bash fix_and_run.sh` to do all of the above in one shot and execute the notebook automatically. However, we recommend launching Jupyter manually and running cells one by one to understand the pipeline.

### Step 5 — Start Jupyter Notebook

```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Open an SSH tunnel from your local machine:

```bash
ssh -L 8888:localhost:8888 <your-user>@<mi300x-node>
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
| `genesis-world` | ≥0.4.1 | Physics simulation + rendering (Taichi backend, ROCm native) |
| `lerobot` | ≥0.4.4 | Dataset format + SmolVLA model |
| `torch` | ≥2.1 (ROCm) | Training and inference |
| `transformers` | ≥4.40 | SmolVLA backbone (Idefics3) |
| `accelerate` | latest | HuggingFace model loading |
| `numpy` | ==2.1.2 | Required by Genesis; must match scikit-image C extension ABI |
| `scikit-image` | ≥0.22 | Must be recompiled against numpy==2.1.2 |
| `xvfb` | system | Headless rendering (apt-get install) |
| `ffmpeg` | system | Video encoding (apt-get install) |

**Hardware**: AMD Instinct MI300X, ROCm 6.x, ≥4 GB VRAM

---

## ROCm Adaptations

All fixes below are handled automatically by `fix_and_run.sh`. You only need to apply them manually if you set up the environment yourself.

### 1. Genesis `cuda.bindings` Patch

Genesis calls `from cuda.bindings import runtime` to query GPU shared memory size. This module does not exist on ROCm. The patch wraps the call in a try-except and falls back to the MI300X LDS size (64 KB):

```python
# genesis/engine/solvers/rigid/rigid_solver.py
try:
    from cuda.bindings import runtime
    _, max_shared_mem = runtime.cudaDeviceGetAttribute(...)
    max_shared_mem /= 1024.0
except (ImportError, Exception):
    max_shared_mem = 64.0  # MI300X LDS fallback
```

### 2. numpy / scikit-image ABI Fix

The base Docker image may ship scikit-image compiled against a different numpy version, causing `ValueError: numpy.dtype size changed` at runtime. Force-reinstalling both packages together recompiles the C extensions:

```bash
pip install --force-reinstall --no-cache-dir "scikit-image>=0.22" "numpy==2.1.2"
```

### 3. ROCm-specific Script Flags

```bash
python scripts/01_gen_data.py --no-bbox-detection --no-videos ...
python scripts/02_train_vla.py --num-workers 0 ...
python scripts/03_eval.py --no-bbox-detection ...
```

- `--no-bbox-detection` — bypass bounding box detection compatibility issues on AMD GPUs
- `--num-workers 0` — avoid torchcodec multi-process decoding crashes on ROCm
- `--no-videos` — store images as PNG instead of MP4 (faster on mounted volumes)

### 4. SmolVLA Compatibility

`lerobot>=0.5.0` may have a `dataclass` field ordering issue with `SmolVLAConfig`. If training fails with `TypeError: non-default argument follows default argument`, use `lerobot==0.4.4` or check the [LeRobot releases](https://github.com/huggingface/lerobot/releases).

---

## Reference Results

Verified on AMD MI300X (ROCm 6.x).

### Data Generation

| Scene | Episodes | Success Rate |
|---|---|---|
| Flat (default) | 10 | **100%** |
| Kitchen (custom) | 10 | **100%** |

| Flat Scene | Kitchen Scene |
|:---:|:---:|
| ![flat](./images/cube_scatter_flat.png) | ![kitchen](./images/cube_scatter_kitchen.png) |

### Training (reference: 100 episodes)

| Metric | Value |
|---|---|
| Training episodes | 100 |
| Steps / batch | 2000 / 4 |
| Loss (start → end) | 0.346 → 0.022 |
| Wall time | ~78 min |
| Peak VRAM | 2.2 GB |

### Evaluation (reference: 100 episodes)

| Eval Set | Success Rate |
|---|---|
| Unseen positions (seed=99) | **4/10 = 40%** |
| Training positions (seed=42) | **5/10 = 50%** |

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
