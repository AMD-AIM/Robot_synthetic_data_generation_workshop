[English](README.md) | [中文](README-cn.md)

# Robot Synthetic Data Generation Workshop

End-to-end pipeline for robot manipulation on **AMD GPUs (ROCm)**: **Synthetic Data Generation → VLA Training → Simulation Evaluation**.

Verified on **CDNA3 (MI300/MI325 series)**, **RDNA4 (Radeon AI PRO R9700)**, and **RDNA3.5 (Radeon PRO W7900)**.

```
┌──────────────────────────────┐  ┌─────────────────────┐  ┌──────────────────────────┐
│ 02_gen_data_custom_scene.py   │  │  02_train_vla.py     │  │  04_eval_custom_scene.py  │
│   kitchen GLB + floor_origin  │  │                      │  │                           │
│   up + wrist cameras          │─▶│  SmolVLA fine-tune   │─▶│  Closed-loop eval         │
│   100 episodes, GPU render    │  │  on LeRobot dataset  │  │  in Genesis kitchen sim   │
│                               │  │  HF checkpoint out   │  │  success rate + video     │
└──────────────────────────────┘  └─────────────────────┘  └──────────────────────────┘
     Franka 7-DOF                      lerobot/smolvla_base       render → VLA → PD
     pick red cube                     freeze vision encoder      action chunking
     2 cameras (up/wrist)              train expert + state_proj  randomized cube pos
```

---

## Workshop Routing — Two Paths

| | Path A: CDNA3 (MI300/MI325) | Path B: RDNA4/3.5 (R9700 / W7900) |
|---|---|---|
| **Data Generation** | Skip — use pre-built HuggingFace dataset | **End-to-end** `02_gen_data_custom_scene.py` kitchen scene (100 episodes) |
| **Training** | Local training | Local training |
| **Evaluation** | CPU render (llvmpipe), ~20 pt lower success rate | GPU render (radeonsi), benchmark-quality |
| **Total time** | ~20 min (train + eval) | **~30 min** (gen + train + eval) |

---

## Path A: CDNA3 (MI300/MI325) — Pre-built Dataset

For MI300/MI325 nodes without GPU graphics pipeline. The 100-episode dataset is pre-generated on RDNA4 and pulled from HuggingFace.

### Student Quick Start

1. Go to [notebooks.amd.com](https://notebooks.amd.com) and log in
2. Open `workshop_cdna3.ipynb`
3. Run cells in order — everything is pre-installed

---

## Path B: RDNA4/3.5 (R9700 / W7900) — End-to-End

For RDNA nodes with GPU graphics pipeline. **All three steps run live during the workshop** — no external dataset needed.

### Model Weights

SmolVLA models are downloaded from [ModelScope](https://modelscope.cn) (China-friendly mirror):

| Model | ModelScope URL |
|---|---|
| SmolVLA base (450M) | https://modelscope.cn/models/lerobot/smolvla_base |
| SmolVLM2-500M backbone | https://modelscope.cn/models/HuggingFaceTB/SmolVLM2-500M-Video-Instruct |

> The pre-built image `workshop-genesis:rocm7.2_w7900_ready` already caches both models. Students do not need to download manually.

### Student Quick Start

```bash
docker exec -it workshop-genesis bash

# Step 0: Download kitchen scene assets (~130 MB, first time only)
python scripts/00_download_kitchen.py --mesh-only

# Step 1: Data Generation — kitchen scene + up/wrist cameras (~15-24 min)
python scripts/02_gen_data_custom_scene.py \
  --scene rustic_kitchen --anchor floor_origin \
  --camera-layout up_wrist \
  --n-episodes 100 --seed 42 \
  --repo-id local/franka-kitchen-wrist-100ep

# Step 2: Training (~7-11 min)
python scripts/02_train_vla.py \
  --dataset-id local/franka-kitchen-wrist-100ep \
  --pretrained lerobot/smolvla_base \
  --n-steps 4000 --batch-size 4 --num-workers 4 \
  --run-name smolvla_kitchen_wrist

# Step 3: Evaluation — kitchen scene (~4 min, GPU render)
python scripts/04_eval_custom_scene.py \
  --checkpoint output/train/smolvla_kitchen_wrist/final \
  --dataset-id local/franka-kitchen-wrist-100ep \
  --scene rustic_kitchen --anchor floor_origin \
  --camera-layout up_wrist \
  --n-episodes 20 --seed 99 --record-video
```

### Reference Timings

| Step | R9700 (RDNA4) | W7900D (RDNA3.5) |
|---|---|---|
| Data Gen (100 ep, kitchen) | ~23 min | ~24 min |
| Training (4000 steps, batch 4) | ~7.4 min (0.11 s/step) | ~10.6 min (0.15 s/step) |
| Evaluation (20 ep, GPU render) | ~4 min | ~4 min |
| **Total** | **~35 min** | **~39 min** |

### Reference Results (kitchen+wrist scene, directly comparable)

| Metric | R9700 (RDNA4) | W7900D (RDNA3.5) |
|---|:---:|:---:|
| Data gen success | 100% | 100% |
| Loss (start → end) | 0.671 → 0.016 | 0.67 → 0.014 |
| Peak VRAM | 2.33 GB | 2.27 GB |
| Eval success rate (GPU render, kitchen) | **~48%** | **~12%** (3 seeds pooled) |

> Note: Both GPUs use the same kitchen+wrist scene — results are directly comparable. W7900 eval success rate is significantly lower than R9700 despite similar training loss convergence. The gap likely stems from ROCm driver version differences (7.0.2 vs 7.2) or RDNA3.5/RDNA4 rendering differences.

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

> **This section is for instructors and cluster admins only.** Students skip to [Path A](#path-a-cdna3-mi300mi325--pre-built-dataset) or [Path B](#path-b-rdna435-r9700--w7900--end-to-end).

### Common Steps

```bash
git clone git@github.com:<org>/Robot_synthetic_data_generation_workshop.git
cd Robot_synthetic_data_generation_workshop
bash docker/build.sh    # → workshop-genesis:latest (~40 min)
```

### MI300/MI325 (CDNA3) Container

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  --network=host \
  -v $(pwd):/workspace/workshop \
  -v ${HF_CACHE}:/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e HSA_OVERRIDE_GFX_VERSION=9.4.2 \
  -w /workspace/workshop \
  workshop-genesis:latest bash
```

### RDNA4/3.5 (R9700 / W7900) Container

```bash
IMAGE=workshop-genesis:rocm7.2_w7900_ready

docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  --network=host \
  -v $(pwd):/workspace/workshop \
  -w /workspace/workshop \
  $IMAGE bash
```

> For nodes with unstable external network, download model weights via [ModelScope](https://modelscope.cn):
> ```bash
> pip install modelscope
> python -c "
> from modelscope import snapshot_download
> snapshot_download('lerobot/smolvla_base', cache_dir='/root/.cache/huggingface/hub')
> snapshot_download('HuggingFaceTB/SmolVLM2-500M-Video-Instruct', cache_dir='/root/.cache/huggingface/hub')
> "
> ```

### Start Jupyter

```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Students access the notebook via [notebooks.amd.com](https://notebooks.amd.com). All outputs are written to `output/` under the workshop directory.

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

## Notebooks

| Notebook | Target Hardware | Description |
|----------|----------------|-------------|
| `workshop_cdna3.ipynb` | MI300/MI325 (CDNA3) | Uses HF pre-built dataset, CPU render eval |
| `workshop_rdna.ipynb` | R9700 / W7900 (RDNA4/3.5) | End-to-end pipeline, GPU render |

### Content Overview

| Section | Path A: `workshop_cdna3.ipynb` | Path B: `workshop_rdna.ipynb` |
|---------|------|------|
| **0. Environment Setup** | GPU detection + HF dataset pull | GPU detection + kitchen GLB download |
| **1. Data Generation** | 2-3 ep demo (illustrative only) | **100 ep kitchen scene generation (~15-24 min)** |
| **2. VLA Training** | SmolVLA post-training (~10 min) | SmolVLA post-training (~7-11 min) |
| **3. Evaluation** | CPU-render closed-loop eval (~10 min) | GPU-render closed-loop eval (~4 min) |
| **4. Summary** | PNG / MP4 / JSON | PNG / MP4 / JSON |

All outputs (checkpoints, plots, eval videos) are written to `output/` in the file browser.

---

## File Structure

```
robot_synthetic_data_generation_workshop/
├── README.md                        ← this file (English)
├── README-cn.md                     ← Chinese version
├── workshop_cdna3.ipynb             ← Jupyter Notebook (CDNA3 path: MI300/MI325)
├── workshop_rdna.ipynb              ← ★ Jupyter Notebook (RDNA path: R9700/W7900, end-to-end)
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
