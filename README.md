[English](README.md) | [中文](README-cn.md)

# Robot Synthetic Data Generation Workshop

End-to-end pipeline for robot manipulation on **AMD GPUs (ROCm)**: **Synthetic Data Generation → VLA Training → Simulation Evaluation**.

Verified on **CDNA3 (MI300/MI325 series)** and **RDNA4 (Radeon AI PRO R9700)**.

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

**Workshop routing**: the workshop is run on a **CDNA3 (MI300/MI325 series)** node for training and evaluation. Data generation is pre-done on RDNA4 and pulled from HuggingFace — you do not need to generate data during the session (a 2-3 episode demo is included in the notebook for illustration). For benchmark-quality evaluation numbers, an RDNA4 node is preferred because MI300/MI325 falls back to CPU rasterization and introduces a systematic ~20 pt success-rate bias (see [Appendix A](#appendix-a-rendering-backend--cdna3-vs-rdna4)).

---

## Dataset

The training dataset is pre-generated on RDNA4 and published on [HuggingFace](https://huggingface.co/datasets/lidavidsh/franka-pick-kitchen-up-wrist-100ep-genesis). The pre-built Docker image already caches it:

| Item | Value |
|---|---|
| Scene | Rustic kitchen GLB + Franka Panda picking a red cube |
| Camera layout | `up` (overhead) + `side` (**wrist-mounted, eye-in-hand**), 640×480 |
| Episodes / Frames | 100 / 13,500 |
| Size | ~200 MB (AV1 video, LeRobot v3.0) |
| Action space | 9-DoF joint position (7 arm + 2 finger) |
| Generated on | RDNA4 (Radeon AI PRO R9700), Genesis 0.4.5, `seed=42` |

> ⚠️ The tensor key `observation.images.side` stores the **wrist (eye-in-hand) camera**, not a world-fixed side view. Do not mix with the legacy `up+world-side` dataset — key names collide but semantics differ.

---

## Teacher / Admin Setup

> **This section is for instructors and cluster admins only.** Students skip straight to [Student Quick Start](#student-quick-start).

On each GPU node, run once before the workshop:

```bash
git clone git@github.com:<org>/Robot_synthetic_data_generation_workshop.git
cd Robot_synthetic_data_generation_workshop

# 1. Build the Docker image (~40 min, includes deps + Taichi cache + model/dataset)
bash docker/build.sh                    # → workshop-genesis:latest

# 2. (Optional) Pre-download HF assets to host — useful if bind-mounts shadow the image cache
export HF_CACHE=/data/hf_cache          # local SSD or shared NFS mount
mkdir -p $HF_CACHE
HF_HOME=$HF_CACHE python -c "
from huggingface_hub import snapshot_download
snapshot_download('lerobot/smolvla_base')
snapshot_download('lidavidsh/franka-pick-kitchen-up-wrist-100ep-genesis', repo_type='dataset')
"
```

The Docker image bakes in all Python deps, torchcodec (CPU-only), SmolVLA base model, HF dataset, and pre-compiled Taichi kernels. Step 2 is only needed when bind-mounts shadow the image's internal cache. For multi-node clusters, repeat on each node or point `HF_CACHE` at a shared NFS/Lustre path.

### Launch the container

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  --network=host \
  -v $(pwd):/workspace/workshop \
  -v ${HF_CACHE}:/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 \
  -w /workspace/workshop \
  workshop-genesis:latest \
  bash
```

> On CDNA3 (MI325/MI300) nodes, add `-e HSA_OVERRIDE_GFX_VERSION=9.4.2`. RDNA4 does not need this.

### Start Jupyter

```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Students access the notebook via [notebooks.amd.com](https://notebooks.amd.com). All outputs are written to `output/` under the workshop directory, visible in the Jupyter file browser.

<details>
<summary><b>Alternative base images</b></summary>

```bash
# ROCm 6.4.3 base (if nodes only have ROCm 6.x driver)
BASE_IMAGE=rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0 \
  bash docker/build.sh workshop-mi300:latest
```

See [`docker/Dockerfile.workshop`](docker/Dockerfile.workshop) for what's baked in.

</details>

<details>
<summary><b>Manual setup (without pre-built image)</b></summary>

If the pre-built image is not available, launch a base ROCm container:

```bash
# CDNA3 (MI325/MI300)
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  --network=host \
  -e PYOPENGL_PLATFORM=egl -e HSA_OVERRIDE_GFX_VERSION=9.4.2 \
  -v $(pwd):/workspace/workshop \
  -w /workspace/workshop \
  rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0 bash

# RDNA4 — use rocm7.2 image and drop HSA_OVERRIDE_GFX_VERSION
```

Then install dependencies manually:

```bash
pip install -q git+https://github.com/Genesis-Embodied-AI/Genesis.git@main \
  lerobot==0.4.4 transformers accelerate safetensors \
  matplotlib Pillow jupyter ipykernel num2words
pip install --force-reinstall --no-cache-dir -q "scikit-image>=0.22" "numpy==2.1.2"
apt-get update -qq && apt-get install -y -qq xvfb ffmpeg > /dev/null 2>&1
bash setup_torchcodec.sh   # ~3-5 min, CPU-only torchcodec (required for AV1 dataset)
```

Or run `bash fix_and_run.sh` to do everything in one shot.

</details>

---

## Student Quick Start

1. Go to [notebooks.amd.com](https://notebooks.amd.com) and log in
2. Open `workshop_pipeline.ipynb`
3. Run cells in order — everything is pre-installed

All outputs (checkpoints, plots, eval videos) are written to `output/` in the file browser.

---

## Notebook Overview

| Section | Content | Output |
|---------|---------|--------|
| **0. Environment Setup** | GPU detection, dependency check, kitchen asset download, HF dataset pull | Environment ready, dataset cached |
| **1. Data Generation (demo)** | 2-3 episode IK trajectory demo to illustrate the data pipeline (not a full run — the 100-episode dataset is pulled from HuggingFace) | Sample dataset + camera / trajectory visualizations |
| **2. VLA Training** | SmolVLA post-training on the HF `kitchen-up-wrist` dataset, frozen vision encoder | Checkpoint + loss curve |
| **3. Evaluation** | Closed-loop sim eval in the kitchen scene (see [Appendix A](#appendix-a-rendering-backend--cdna3-vs-rdna4) for CPU-render bias) | Success rate + videos |
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
├── setup_torchcodec.sh              ← build torchcodec v0.10.0 CPU-only for ROCm
├── docker/
│   ├── Dockerfile.workshop          ← pre-built image (all deps + Taichi cache + models)
│   ├── build.sh                     ← build helper script
│   └── warmup_cache.py              ← Taichi kernel pre-compilation for Docker build
├── images/                          ← pre-generated visualizations (referenced by notebook)
│   ├── ep0_camera_views.png
│   ├── ep0_joint_trajectory.png
│   ├── cube_scatter_kitchen.png
│   └── kitchen_wrist/              ← kitchen scene up + wrist camera sample frames
├── scenes/
│   └── rustic_kitchen.json          ← kitchen scene config (anchors, mesh refs)
├── output/                          ← all runtime outputs (created by notebook / scripts)
│   ├── data/                        ← data generation sidecars (gen_summary.json, etc.)
│   │   ├── franka_gen_pick/
│   │   └── custom_scene_gen/
│   ├── train/                       ← training checkpoints + metrics
│   │   └── smolvla_kitchen_wrist/   ← final/, checkpoint_*, train_summary.json
│   └── eval/                        ← evaluation results + videos
│       └── kitchen_eval/            ← eval_summary.json, videos/
└── scripts/
    ├── 00_download_kitchen.py       ← download kitchen GLB assets
    ├── 01_gen_data.py               ← data generation (flat scene, legacy / reference)
    ├── 02_gen_data_custom_scene.py  ← data generation (custom 3D scene + up/wrist camera layouts)
    ├── 02_train_vla.py              ← SmolVLA post-training
    ├── 03_eval.py                   ← closed-loop eval (flat scene, legacy / reference)
    ├── 04_eval_custom_scene.py      ← closed-loop eval (custom scene — workshop main path)
    ├── genesis_scene_utils.py       ← Genesis utility functions
    ├── pick_common.py               ← scene-agnostic pick task builder (camera layout factory)
    └── scene_placement.py           ← robot-local coordinate utilities
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `genesis-world` | main (`pip install git+...@main`) | Physics simulation + rendering (Taichi backend, ROCm native). Install from main to avoid the `cuda.bindings` issue in PyPI 0.4.5. |
| `lerobot` | ≥0.4.4 | Dataset format + SmolVLA model |
| `torch` | ≥2.1 (ROCm) | Training and inference |
| `transformers` | ≥4.40 | SmolVLA backbone (Idefics3) |
| `accelerate` | latest | HuggingFace model loading |
| `num2words` | latest | Required by `transformers` SmolVLM processor |
| `numpy` | ==2.1.2 | Required by Genesis; must match scikit-image C extension ABI |
| `scikit-image` | ≥0.22 | Must be recompiled against numpy==2.1.2 |
| `xvfb` | system | Headless rendering (apt-get install) |
| `ffmpeg` | system | Video encoding (apt-get install) |

**Hardware**: CDNA3 (AMD Instinct MI300/MI325 series, ROCm 6.x) **or** RDNA4 (AMD Radeon AI PRO R9700, ROCm 7.2); ≥4 GB VRAM on either.

---

## Script Quick Reference

<details>
<summary><b>Training (02_train_vla.py)</b></summary>

```bash
python scripts/02_train_vla.py \
  --dataset-id lidavidsh/franka-pick-kitchen-up-wrist-100ep-genesis \
  --n-steps 4000 --batch-size 4 --num-workers 4 \
  --run-name smolvla_kitchen_wrist
```

- AMP BF16 + PyTorch SDPA auto-dispatch (AOTriton flash on AMD) are **auto-enabled** when CUDA is available — no flags needed.
- `--num-workers 4` requires the CPU-only torchcodec build (included in the pre-built image; see manual setup for details).

</details>

<details>
<summary><b>Evaluation (04_eval_custom_scene.py)</b></summary>

```bash
# CDNA3 (MI300/MI325) — must use --render-cpu (no GPU graphics driver)
python scripts/04_eval_custom_scene.py \
  --checkpoint output/train/smolvla_kitchen_wrist/final \
  --dataset-id lidavidsh/franka-pick-kitchen-up-wrist-100ep-genesis \
  --scene rustic_kitchen --anchor floor_origin \
  --camera-layout up_wrist --render-cpu \
  --n-episodes 20 --seed 99 --record-video

# RDNA4 — omit --render-cpu, uses GPU radeonsi
python scripts/04_eval_custom_scene.py \
  --checkpoint output/train/smolvla_kitchen_wrist/final \
  --dataset-id lidavidsh/franka-pick-kitchen-up-wrist-100ep-genesis \
  --scene rustic_kitchen --anchor floor_origin \
  --camera-layout up_wrist \
  --n-episodes 20 --seed 99 --record-video
```

- **Download kitchen assets first**: `python scripts/00_download_kitchen.py` (one-time, ~130 MB). The notebook does this automatically in Section 0; for CLI usage, run it manually before eval.
- `--camera-layout up_wrist` **must** match the dataset; omitting it loads a world-fixed side camera.
- `--render-cpu` (CDNA3 only) forces CPU llvmpipe; introduces ~20 pt success-rate bias vs GPU rendering — see [Appendix A](#appendix-a-rendering-backend-cdna3-vs-rdna4).
- First-time Genesis CPU compilation (`scene.build()`) takes **20-30 min** on MI300/MI325; subsequent runs reuse the Taichi kernel cache. The pre-built Docker image (Option A) already includes this cache — no wait.

</details>

<details>
<summary><b>Data Generation (02_gen_data_custom_scene.py) — optional, RDNA4 preferred</b></summary>

```bash
python scripts/02_gen_data_custom_scene.py \
  --scene rustic_kitchen --anchor floor_origin \
  --camera-layout up_wrist \
  --n-episodes 100 --seed 42 \
  --repo-id local/kitchen-pick
```

</details>

---

## Reference Results

All numbers below are from the workshop main path (`kitchen-up-wrist`, 100 ep dataset).

### Data Generation

| Architecture | Success Rate | Per-episode |
|---|:---:|:---:|
| RDNA4 (R9700) | 100/100 | **~14 s/ep** (Genesis compile + video encode included, ~12 s/ep steady state) |
| CDNA3 (MI300/MI325) | 100/100 | ~4× slower due to CPU rasterization |

100-episode wall clock on RDNA4: **~23 min** end-to-end including Genesis scene compile and SVT-AV1 encode.

### Training (100 ep, 4000 steps, batch 4 — default recipe)

Default recipe = Video dataset + `num-workers=4` + AMP BF16 + PyTorch SDPA auto (AOTriton flash on AMD). AMP + SDPA are auto-enabled by `02_train_vla.py` when CUDA is available.

| Metric | RDNA4 (R9700, ROCm 7.2) | CDNA3 (MI300/MI325, ROCm 6.4.3) |
|---|:---:|:---:|
| Wall time | **~7.4 min** (444 s) | **~10.6 min** (637 s) |
| Per-step | 0.111 s | 0.159 s |
| Peak VRAM | 2.33 GB | 2.24 GB |
| Loss (start → end) | 0.671 → 0.0161 | 0.671 → 0.0162 |

> ⚠️ Per-step values cannot be used to rank RDNA4 vs CDNA3 raw compute: SmolVLA 450M + batch=4 + short sequences severely under-utilize CDNA3 (VRAM usage only ~1.2%), so per-step is kernel-launch-bound rather than compute-bound. This table is for workshop reproducibility, not a chip benchmark.

### Evaluation (kitchen+wrist, 5 eval seeds × 20 trials)

Success rate is sensitive both to the training stack and to the evaluation renderer. The 2×2 matrix below isolates both axes using the same 100-episode dataset and the same default training recipe:

| train \ eval render | **MI300/MI325 CPU (llvmpipe)** | **RDNA4 GPU (radeonsi)** |
|---|:---:|:---:|
| CDNA3 (ROCm 6.4 + PyTorch 2.6) | 25.0 % | 45.0 % |
| RDNA4 (ROCm 7.2 + PyTorch 2.9) | — (not tested) | **48.0 %** |

Key takeaways:

1. **Eval renderer dominates**: same checkpoint, CPU → GPU eval lifts success rate ~20 pt (25 % → 45 %). MI300/MI325 evaluation numbers are systematically low; for benchmark-quality results, evaluate on RDNA4 or another GPU-render node.
2. **Training stack is equivalent**: CDNA3-trained and RDNA4-trained checkpoints score 45 % vs 48 % under matched GPU eval — within eval standard deviation (~10 pt). MI300/MI325 is a fully usable training node.
3. **Per-seed spread is large**: individual eval seeds span 35-60 %; always report pooled or mean ± std over ≥3 eval seeds.

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

## Appendix A: Rendering Backend — CDNA3 vs RDNA4

| Architecture | EGL Renderer | Type |
|---|---|---|
| CDNA3 (MI300/MI325 series) | llvmpipe | CPU software rasterization |
| RDNA4 (R9700) | radeonsi | **GPU hardware rasterization** |

CDNA3 has no graphics pipeline — Genesis falls back to CPU `llvmpipe` for camera rendering. RDNA4 has a full graphics pipeline (`radeonsi`, hardware-accelerated), which is the primary source of the 3-4× data-generation speedup and eliminates the render-gap bias at evaluation.

**CPU-render evaluation bias (MI300/MI325)**: CPU and GPU rasterizers produce visually different frames. A policy trained on GPU-rendered data but evaluated with CPU rendering shows a systematic **~20 pt lower success rate**. On the kitchen+wrist main path: RDNA4 GPU eval pooled ~45-48 % vs MI300/MI325 CPU eval pooled ~25 %. This is expected behaviour, not a bug. For benchmark-quality numbers, evaluate on an RDNA4 or another GPU-render node.

## Appendix B: Known Compatibility Notes

| Issue | Fix | Auto-handled by |
|---|---|---|
| Genesis PyPI 0.4.5 imports `cuda.bindings` (missing on ROCm) | Install from `main` branch (fixed in [`e807698`](https://github.com/Genesis-Embodied-AI/Genesis/commit/e807698b8aa773fad3a6dfb4556889b251c30924)) | Dockerfile / `fix_and_run.sh` |
| numpy / scikit-image ABI mismatch (`numpy.dtype size changed`) | `pip install --force-reinstall "scikit-image>=0.22" "numpy==2.1.2"` | Dockerfile / `fix_and_run.sh` |
| torchcodec pip wheel links CUDA libs, fails on ROCm | `bash setup_torchcodec.sh` (CPU-only build) | Dockerfile / `fix_and_run.sh` |
| `lerobot>=0.5.0` dataclass ordering error with SmolVLAConfig | Pin `lerobot==0.4.4` | Dockerfile / `fix_and_run.sh` |

---

## References

- [LeRobot](https://github.com/huggingface/lerobot) — Robot learning framework (dataset + policies)
- [Genesis](https://genesis-embodied-ai.github.io/) — GPU-accelerated physics simulation (ROCm native via Taichi)
- [SmolVLA](https://huggingface.co/blog/smolvla) — Vision-Language-Action model
- [World Labs Marble](https://marble.worldlabs.ai/) — 3D scene generation for custom environments
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
