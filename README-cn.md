[English](README.md) | [中文](README-cn.md)

# 机器人合成数据生成 Workshop

基于 **AMD GPU (ROCm)** 的机器人操作全流程：**合成数据生成 → VLA 训练 → 仿真评估**。

已在 **CDNA3 (MI300/MI325)**、**RDNA4 (R9700)** 和 **RDNA3.5 (W7900)** 上验证。

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

## Workshop 两条路径

本 Workshop 根据硬件分为两条执行路径：

| | 路径 A: CDNA3 (MI300/MI325) | 路径 B: RDNA4/3.5 (R9700 / W7900) |
|---|---|---|
| **数据生成** | 跳过，使用 HuggingFace 预构建数据集 | **端到端执行** `02_gen_data_custom_scene.py` kitchen 场景生成 100 集 |
| **训练** | 本地训练 | 本地训练 |
| **评估** | CPU 渲染 (llvmpipe)，成功率偏低 ~20pt | GPU 渲染 (radeonsi)，benchmark 级结果 |
| **总耗时** | ~20 min (训练 + 评估) | **~30 min** (数据生成 + 训练 + 评估) |

---

## 路径 A: CDNA3 (MI300/MI325) — 使用预构建数据集

适用于没有 GPU 图形流水线的 MI300/MI325 节点。数据集已预生成并发布到 HuggingFace。

### 学员快速开始

1. 打开 [notebooks.amd.com](https://notebooks.amd.com) 并登录
2. 打开 `workshop_pipeline.ipynb`
3. 按顺序执行 cell——所有依赖已预装

### 脚本方式执行

```bash
# 进入预构建容器（镜像已含数据集 + 模型缓存）
docker exec -it workshop-genesis bash

# Step 1: 数据生成 — 跳过（仅 2-3 集 demo 用于讲解）
# 100 集数据集自动从 HF 拉取: lidavidsh/franka-pick-kitchen-up-wrist-100ep-genesis

# Step 2: 训练 (~10 min)
python scripts/02_train_vla.py \
  --dataset-id lidavidsh/franka-pick-kitchen-up-wrist-100ep-genesis \
  --n-steps 4000 --batch-size 4 --num-workers 4 \
  --run-name smolvla_kitchen_wrist

# Step 3: 评估 (~10 min, CPU 渲染)
python scripts/04_eval_custom_scene.py \
  --checkpoint output/train/smolvla_kitchen_wrist/final \
  --dataset-id lidavidsh/franka-pick-kitchen-up-wrist-100ep-genesis \
  --scene rustic_kitchen --anchor floor_origin \
  --camera-layout up_wrist --render-cpu \
  --n-episodes 20 --seed 99 --record-video
```

### 参考耗时 (MI300/MI325)

| Step | 耗时 | 备注 |
|---|---|---|
| 训练 (4000 steps, batch 4) | ~10.6 min | 0.159 s/step, peak VRAM 2.24 GB |
| 评估 (20 ep, CPU render) | ~10 min | 含 Taichi 首次编译 |
| **合计** | **~20 min** | 不含数据生成 |

### 参考结果

- **Loss**: 0.671 → 0.016
- **Eval 成功率**: ~25% (CPU 渲染, llvmpipe)

> MI300/MI325 首次 Genesis 编译需 20-30 min；预构建镜像已包含 Taichi 缓存，无需等待。

---

## 路径 B: RDNA4/3.5 (R9700 / W7900) — 端到端全流程

适用于有 GPU 图形流水线的 RDNA 节点。**全部三步在 workshop 期间实时执行**，无需外部数据集。

### 模型权重

SmolVLA 模型通过 [ModelScope（魔塔社区）](https://modelscope.cn) 下载（国内网络友好）：

| 模型 | ModelScope 地址 |
|---|---|
| SmolVLA base (450M) | https://modelscope.cn/models/lerobot/smolvla_base |
| SmolVLM2-500M backbone | https://modelscope.cn/models/HuggingFaceTB/SmolVLM2-500M-Video-Instruct |

> 预构建镜像 `workshop-genesis:rocm7.2_w7900_ready` 已缓存上述模型，学员无需手动下载。

### 学员快速开始

```bash
docker exec -it workshop-genesis bash

# Step 0: 下载厨房场景资源 (~130 MB, 首次运行)
python scripts/00_download_kitchen.py --mesh-only

# Step 1: 数据生成 — 厨房场景 + up/wrist 相机 (~15-24 min)
python scripts/02_gen_data_custom_scene.py \
  --scene rustic_kitchen --anchor floor_origin \
  --camera-layout up_wrist \
  --n-episodes 100 --seed 42 \
  --repo-id local/franka-kitchen-wrist-100ep

# Step 2: 训练 (~7-11 min)
python scripts/02_train_vla.py \
  --dataset-id local/franka-kitchen-wrist-100ep \
  --pretrained lerobot/smolvla_base \
  --n-steps 4000 --batch-size 4 --num-workers 4 \
  --run-name smolvla_kitchen_wrist

# Step 3: 评估 — 厨房场景 (~4 min, GPU 渲染)
python scripts/04_eval_custom_scene.py \
  --checkpoint output/train/smolvla_kitchen_wrist/final \
  --dataset-id local/franka-kitchen-wrist-100ep \
  --scene rustic_kitchen --anchor floor_origin \
  --camera-layout up_wrist \
  --n-episodes 20 --seed 99 --record-video
```

### 参考耗时

| Step | R9700 (RDNA4) | W7900D (RDNA3.5) |
|---|---|---|
| 数据生成 (100 ep, kitchen) | ~23 min | ~24 min |
| 训练 (4000 steps, batch 4) | ~7.4 min (0.11 s/step) | ~10.6 min (0.15 s/step) |
| 评估 (20 ep, GPU render) | ~4 min | ~4 min |
| **合计** | **~35 min** | **~39 min** |

### 参考结果 (kitchen+wrist 场景, 可直接对比)

| 指标 | R9700 (RDNA4) | W7900D (RDNA3.5) |
|---|:---:|:---:|
| 数据生成成功率 | 100% | 100% |
| Loss (start → end) | 0.671 → 0.016 | 0.67 → 0.014 |
| Peak VRAM | 2.33 GB | 2.27 GB |
| Eval 成功率 (GPU render, kitchen) | **~48%** | **~12%** (3 seeds pooled) |

> Note: 两卡均使用 kitchen+wrist 场景，结果可直接对比。W7900 eval 成功率显著低于 R9700，训练 loss 收敛一致，差异可能来自 ROCm driver 版本 (7.0.2 vs 7.2) 或 RDNA3.5/RDNA4 渲染差异。

---

## 数据集

| 项目 | 值 |
|---|---|
| 场景 | Rustic Kitchen (GLB mesh) + Franka Panda 抓取红色方块 |
| 相机配置 | `up`（俯视）+ `wrist`（腕部 eye-in-hand），640×480 |
| 集数 / 帧数 | 100 / 13,500 |
| 大小 | ~200 MB（AV1 视频，LeRobot v3.0） |
| 动作空间 | 9-DoF 关节位置（7 臂 + 2 指） |

路径 A 从 HuggingFace 下载预构建数据集；路径 B 使用 `02_gen_data_custom_scene.py` 在 kitchen 场景中实时生成。

---

## 教师 / 管理员配置

> **本节面向讲师和集群管理员。** 学员无需关注。

### 通用步骤

```bash
git clone git@github.com:<org>/Robot_synthetic_data_generation_workshop.git
cd Robot_synthetic_data_generation_workshop
bash docker/build.sh    # → workshop-genesis:latest (~40 min)
```

### MI300/MI325 (CDNA3) 容器启动

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

### RDNA4/3.5 (R9700 / W7900) 容器启动

```bash
# 预构建镜像 (W7900 已保存)
IMAGE=workshop-genesis:rocm7.2_w7900_ready

docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  --network=host \
  -v $(pwd):/workspace/workshop \
  -w /workspace/workshop \
  $IMAGE bash
```

> W7900 外网不稳定时，模型权重通过 ModelScope（魔塔社区）下载：
> ```bash
> pip install modelscope
> python -c "
> from modelscope import snapshot_download
> snapshot_download('lerobot/smolvla_base', cache_dir='/root/.cache/huggingface/hub')
> snapshot_download('HuggingFaceTB/SmolVLM2-500M-Video-Instruct', cache_dir='/root/.cache/huggingface/hub')
> "
> ```

### 启动 Jupyter

```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

---

## Notebook 文件

| Notebook | 目标硬件 | 说明 |
|----------|---------|------|
| `workshop_cdna3.ipynb` | MI300/MI325 (CDNA3) | 使用 HF 预构建数据集，CPU 渲染评估 |
| `workshop_rdna.ipynb` | R9700 / W7900 (RDNA4/3.5) | 端到端全流程，GPU 渲染 |

### 内容概览

| 章节 | 路径 A: `workshop_cdna3.ipynb` | 路径 B: `workshop_rdna.ipynb` |
|------|------|------|
| **0. 环境配置** | GPU 检测 + HF 数据集拉取 | GPU 检测 + 下载厨房 GLB 资源 |
| **1. 数据生成** | 2-3 集 demo（展示数据结构） | **100 集 kitchen 场景生成 (~15-24 min)** |
| **2. VLA 训练** | SmolVLA post-training (~10 min) | SmolVLA post-training (~7-11 min) |
| **3. 仿真评估** | CPU 渲染闭环评估 (~10 min) | GPU 渲染闭环评估 (~4 min) |
| **4. 结果汇总** | PNG / MP4 / JSON | PNG / MP4 / JSON |

---

## 文件结构

```
robot_synthetic_data_generation_workshop/
├── README.md                        ← 英文说明
├── README-cn.md                     ← 本文件（中文说明）
├── workshop_cdna3.ipynb             ← Jupyter Notebook（CDNA3 路径: MI300/MI325）
├── workshop_rdna.ipynb              ← ★ Jupyter Notebook（RDNA 路径: R9700/W7900, 端到端）
├── fix_and_run.sh                   ← 一键执行：安装依赖 + ROCm 补丁 + 运行 notebook
├── setup_torchcodec.sh              ← 构建 torchcodec v0.10.0 CPU-only（ROCm 用）
├── docker/
│   ├── Dockerfile.workshop          ← 预构建镜像（全部依赖 + Taichi 缓存 + 模型）
│   ├── build.sh                     ← 构建辅助脚本
│   └── warmup_cache.py              ← Docker 构建时 Taichi 内核预编译
├── images/                          ← 预生成的可视化（notebook 内引用）
├── scenes/
│   └── rustic_kitchen.json          ← 厨房场景配置（锚点、Mesh 引用）
├── output/                          ← 所有运行时产出
│   ├── data/                        ← 数据生成产出
│   ├── train/                       ← 训练 checkpoint + 指标
│   └── eval/                        ← 评估结果 + 视频
└── scripts/
    ├── 00_download_kitchen.py       ← 下载厨房 GLB 资源
    ├── 01_gen_data.py               ← 数据生成（平面场景）
    ├── 02_gen_data_custom_scene.py  ← 数据生成（厨房场景 + up/wrist 相机）
    ├── 02_train_vla.py              ← SmolVLA 后训练
    ├── 03_eval.py                   ← 闭环评估（平面场景）
    ├── 04_eval_custom_scene.py      ← 闭环评估（厨房场景）
    ├── genesis_scene_utils.py       ← Genesis 工具函数
    ├── pick_common.py               ← 抓取任务构建器
    └── scene_placement.py           ← 机器人坐标系工具
```

---

## 依赖

| 包名 | 版本 | 用途 |
|---|---|---|
| `genesis-world` | main (`pip install git+...@main`) | 物理仿真 + 渲染（Taichi 后端，ROCm 原生） |
| `lerobot` | ==0.4.4 | 数据集格式 + SmolVLA 模型 |
| `torch` | ≥2.1 (ROCm) | 训练与推理 |
| `transformers` | ≥4.40 | SmolVLA 骨干 (SmolVLM2) |
| `accelerate` | latest | 模型加载 |
| `numpy` | ==2.1.2 | Genesis 依赖，需匹配 scikit-image ABI |
| `xvfb` / `ffmpeg` | 系统包 | 无头渲染 / 视频编码 |

**硬件要求**：CDNA3 (MI300/MI325, ROCm 6.x) 或 RDNA4/3.5 (R9700/W7900, ROCm 7.x)，≥4 GB VRAM。

---

## 数据流

```
Genesis 仿真场景                  LeRobot 数据集                SmolVLA
┌──────────────┐                ┌──────────────┐              ┌──────────────┐
│ Franka Panda │                │ observation   │              │ 视觉编码器    │
│ 红色方块      │──IK 规划──────▶│  .state [9D]  │──训练───────▶│ (冻结)       │
│ 双相机        │   关节插值      │  .images.up   │              │              │
│              │   渲染          │  .images.side │              │ Expert       │
│ 物理引擎      │                │ action [9D]   │              │ 层（可训练）   │
│ (Genesis)    │                │ task (文本)    │              │              │
└──────────────┘                └──────────────┘              │ → 动作分块    │
                                                              │   [50步]     │
评估循环：                                                      └──────────────┘
  渲染 → 推理 → PD 控制 → scene.step()
```

---

## 附录 A：渲染后端对比

| 架构 | EGL 渲染器 | 类型 | 评估 bias |
|---|---|---|---|
| CDNA3 (MI300/MI325) | llvmpipe | CPU 软件光栅化 | 成功率低 ~20 pt |
| RDNA4 (R9700) | radeonsi | GPU 硬件光栅化 | 无 bias |
| RDNA3.5 (W7900) | radeonsi | GPU 硬件光栅化 | 无 bias |

CDNA3 没有图形流水线，Genesis 回退到 CPU llvmpipe。RDNA4/3.5 有完整图形流水线，数据生成快 3-4×，评估无 render-gap bias。

## 附录 B：已知兼容性问题

| 问题 | 修复 |
|---|---|
| Genesis PyPI 0.4.5 导入 `cuda.bindings` | 从 main 分支安装 |
| numpy / scikit-image ABI 不兼容 | `pip install --force-reinstall "scikit-image>=0.22" "numpy==2.1.2"` |
| torchcodec pip wheel 链接 CUDA 库 | `bash setup_torchcodec.sh`（CPU-only 构建） |
| `lerobot>=0.5.0` dataclass 排序错误 | 锁定 `lerobot==0.4.4` |

---

## 参考资料

- [LeRobot](https://github.com/huggingface/lerobot) — 机器人学习框架
- [Genesis](https://genesis-embodied-ai.github.io/) — GPU 加速物理仿真
- [SmolVLA](https://huggingface.co/blog/smolvla) — 视觉-语言-动作模型
- [AMD ROCm 文档](https://rocm.docs.amd.com/)
