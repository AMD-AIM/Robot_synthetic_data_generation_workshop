[English](README.md) | [中文](README-cn.md)

# 机器人合成数据生成 Workshop

基于 **AMD GPU (ROCm)** 的机器人操作全流程：**合成数据生成 → VLA 训练 → 仿真评估**。

已在 **CDNA3 (MI300/MI325 系列)** 和 **RDNA4 (Radeon AI PRO R9700)** 上验证。

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

**Workshop 运行链路**：workshop 在 **CDNA3 (MI300/MI325 系列)** 节点上进行训练和评估。数据生成已预先在 RDNA4 上完成并发布到 HuggingFace，学员**无需在 workshop 期间生成数据**（notebook 中保留了 2-3 集 demo 用于讲解数据结构）。若需要 benchmark 质量的评估数字，优先使用 RDNA4 节点——MI300/MI325 走 CPU 软件光栅化，评估成功率会系统性低约 20 pt（见[附录 A](#附录-a渲染后端--cdna3-vs-rdna4)）。

---

## 数据集

训练数据集已预生成于 RDNA4 并发布到 [HuggingFace](https://huggingface.co/datasets/lidavidsh/franka-pick-kitchen-up-wrist-100ep-genesis)。预构建 Docker 镜像已包含缓存：

| 项目 | 值 |
|---|---|
| 场景 | Rustic kitchen GLB + Franka Panda 抓取红色方块 |
| 相机配置 | `up`（俯视）+ `side`（**腕部固定 / 眼在手上**），640×480 |
| 集数 / 帧数 | 100 / 13,500 |
| 大小 | ~200 MB（AV1 视频，LeRobot v3.0） |
| 动作空间 | 9-DoF 关节位置（7 臂 + 2 指） |
| 生成环境 | RDNA4 (Radeon AI PRO R9700)，Genesis 0.4.5，`seed=42` |

> ⚠️ 该数据集的 `observation.images.side` 存储的是**腕部相机（眼在手上）**，不是世界固定侧视。不要与旧的 `up+world-side` 数据集混用——key 名相同但语义冲突。

---

## 教师 / 管理员配置

> **本节面向讲师和集群管理员。** 学员请直接跳到[学员快速开始](#学员快速开始)。

在每个 GPU 节点上，workshop 前执行一次：

```bash
git clone git@github.com:<org>/Robot_synthetic_data_generation_workshop.git
cd Robot_synthetic_data_generation_workshop

# 1. 构建 Docker 镜像（~40 min，含依赖 + Taichi 缓存 + 模型/数据集）
bash docker/build.sh                    # → workshop-genesis:latest

# 2.（可选）将 HF 资源预下载到宿主机——bind-mount 覆盖镜像内缓存时需要
export HF_CACHE=/data/hf_cache          # 本地 SSD 或共享 NFS 挂载点
mkdir -p $HF_CACHE
HF_HOME=$HF_CACHE python -c "
from huggingface_hub import snapshot_download
snapshot_download('lerobot/smolvla_base')
snapshot_download('lidavidsh/franka-pick-kitchen-up-wrist-100ep-genesis', repo_type='dataset')
"
```

Docker 镜像内置全部 Python 依赖、torchcodec（CPU-only）、SmolVLA 基础模型、HF 数据集缓存及预编译 Taichi 内核。步骤 2 仅在 bind-mount 覆盖镜像内部缓存时需要。多节点集群可在每节点重复，或将 `HF_CACHE` 指向共享 NFS/Lustre 路径。

### 启动容器

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

> CDNA3 (MI325/MI300) 节点需额外添加 `-e HSA_OVERRIDE_GFX_VERSION=9.4.2`；RDNA4 不需要。

### 启动 Jupyter

```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

学员通过 [notebooks.amd.com](https://notebooks.amd.com) 访问 notebook。所有产出写入 workshop 目录下的 `output/`，在 Jupyter 文件浏览器中可见。

<details>
<summary><b>备选基础镜像</b></summary>

```bash
# ROCm 6.4.3 基础镜像（节点仅有 ROCm 6.x 驱动时使用）
BASE_IMAGE=rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0 \
  bash docker/build.sh workshop-mi300:latest
```

详见 [`docker/Dockerfile.workshop`](docker/Dockerfile.workshop)。

</details>

<details>
<summary><b>手动配置（无预构建镜像）</b></summary>

如果预构建镜像不可用，启动基础 ROCm 容器：

```bash
# CDNA3 (MI325/MI300)
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  --network=host \
  -e PYOPENGL_PLATFORM=egl -e HSA_OVERRIDE_GFX_VERSION=9.4.2 \
  -v $(pwd):/workspace/workshop \
  -w /workspace/workshop \
  rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0 bash

# RDNA4 — 使用 rocm7.2 镜像，不设 HSA_OVERRIDE_GFX_VERSION
```

然后手动安装依赖：

```bash
pip install -q git+https://github.com/Genesis-Embodied-AI/Genesis.git@main \
  lerobot==0.4.4 transformers accelerate safetensors \
  matplotlib Pillow jupyter ipykernel num2words
pip install --force-reinstall --no-cache-dir -q "scikit-image>=0.22" "numpy==2.1.2"
apt-get update -qq && apt-get install -y -qq xvfb ffmpeg > /dev/null 2>&1
bash setup_torchcodec.sh   # ~3-5 min，CPU-only torchcodec（AV1 数据集必需）
```

或执行 `bash fix_and_run.sh` 一键完成。

</details>

---

## 学员快速开始

1. 打开 [notebooks.amd.com](https://notebooks.amd.com) 并登录
2. 打开 `workshop_pipeline.ipynb`
3. 按顺序执行 cell——所有依赖已预装

所有产出（checkpoint、图表、评估视频）写入文件浏览器中的 `output/` 目录。

---

## Notebook 内容概览

| 章节 | 内容 | 产出 |
|------|------|------|
| **0. 环境配置** | GPU 检测、依赖验证、厨房资源下载、HF 数据集缓存 | 环境就绪，数据集本地缓存 |
| **1. 数据生成（演示）** | 2-3 集 IK 轨迹演示，讲解数据结构（不做 100 集全量——主数据集从 HuggingFace 拉取） | 示例数据集 + 相机/轨迹可视化 |
| **2. VLA 训练** | 基于 HF `kitchen-up-wrist` 数据集的 SmolVLA 后训练（冻结视觉编码器） | Checkpoint + Loss 曲线 |
| **3. 仿真评估** | 厨房场景闭环评估（MI300/MI325 CPU 渲染 bias 见[附录 A](#附录-a渲染后端--cdna3-vs-rdna4)） | 成功率 + 评估视频 |
| **4. 结果汇总** | 全部产出物收集与展示 | PNG / MP4 / JSON |

每节包含：
- **背景知识** — 本步骤意义及技术原理
- **可执行代码** — 直接运行 cell
- **嵌入式可视化** — 预生成图片 + 运行时 matplotlib 实时绘图

---

## 文件结构

```
robot_synthetic_data_generation_workshop/
├── README.md                        ← 英文说明
├── README-cn.md                     ← 本文件（中文说明）
├── workshop_pipeline.ipynb          ← ★ Jupyter Notebook（Workshop 主体）
├── fix_and_run.sh                   ← 一键执行：安装依赖 + ROCm 补丁 + 运行 notebook
├── setup_torchcodec.sh              ← 构建 torchcodec v0.10.0 CPU-only（ROCm 用）
├── docker/
│   ├── Dockerfile.workshop          ← 预构建镜像（全部依赖 + Taichi 缓存 + 模型）
│   ├── build.sh                     ← 构建辅助脚本
│   └── warmup_cache.py              ← Docker 构建时 Taichi 内核预编译
├── images/                          ← 预生成的可视化（notebook 内引用）
│   ├── ep0_camera_views.png
│   ├── ep0_joint_trajectory.png
│   ├── cube_scatter_kitchen.png
│   └── kitchen_wrist/              ← 厨房场景 up + wrist 相机示例帧
├── scenes/
│   └── rustic_kitchen.json          ← 厨房场景配置（锚点、Mesh 引用）
├── output/                          ← 所有运行时产出（由 notebook / 脚本创建）
│   ├── data/                        ← 数据生成 sidecar（gen_summary.json 等）
│   │   ├── franka_gen_pick/
│   │   └── custom_scene_gen/
│   ├── train/                       ← 训练 checkpoint + 指标
│   │   └── smolvla_kitchen_wrist/   ← final/、checkpoint_*、train_summary.json
│   └── eval/                        ← 评估结果 + 视频
│       └── kitchen_eval/            ← eval_summary.json、videos/
└── scripts/
    ├── 00_download_kitchen.py       ← 下载厨房 GLB 资源
    ├── 01_gen_data.py               ← 数据生成（平面场景，历史 / 参考用）
    ├── 02_gen_data_custom_scene.py  ← 数据生成（自定义 3D 场景 + up/wrist 相机配置）
    ├── 02_train_vla.py              ← SmolVLA 后训练
    ├── 03_eval.py                   ← 闭环评估（平面场景，历史 / 参考用）
    ├── 04_eval_custom_scene.py      ← 闭环评估（自定义场景 — workshop 主路径）
    ├── genesis_scene_utils.py       ← Genesis 工具函数
    ├── pick_common.py               ← 场景无关的抓取任务构建器（含相机配置工厂）
    └── scene_placement.py           ← 机器人局部坐标系工具
```

---

## 依赖

| 包名 | 版本 | 用途 |
|---|---|---|
| `genesis-world` | main（`pip install git+...@main`） | 物理仿真 + 渲染（Taichi 后端，原生支持 ROCm）。从 main 安装以避免 PyPI 0.4.5 的 `cuda.bindings` 问题。 |
| `lerobot` | ≥0.4.4 | 数据集格式 + SmolVLA 模型 |
| `torch` | ≥2.1 (ROCm) | 训练与推理 |
| `transformers` | ≥4.40 | SmolVLA 骨干网络 (Idefics3) |
| `accelerate` | 最新 | HuggingFace 模型加载 |
| `num2words` | 最新 | `transformers` SmolVLM 处理器依赖 |
| `numpy` | ==2.1.2 | Genesis 依赖，需与 scikit-image 的 C 扩展 ABI 匹配 |
| `scikit-image` | ≥0.22 | 需在 numpy==2.1.2 下重新编译 |
| `xvfb` | 系统包 | 无头渲染（apt-get 安装） |
| `ffmpeg` | 系统包 | 视频编码（apt-get 安装） |

**硬件要求**：CDNA3 (AMD Instinct MI300/MI325 系列, ROCm 6.x) **或** RDNA4 (AMD Radeon AI PRO R9700, ROCm 7.2)，任一架构显存 ≥4 GB。

---

## 脚本速查

<details>
<summary><b>训练（02_train_vla.py）</b></summary>

```bash
python scripts/02_train_vla.py \
  --dataset-id lidavidsh/franka-pick-kitchen-up-wrist-100ep-genesis \
  --n-steps 4000 --batch-size 4 --num-workers 4 \
  --run-name smolvla_kitchen_wrist
```

- AMP BF16 + PyTorch SDPA auto-dispatch（AMD 上走 AOTriton flash）在 CUDA 可用时**自动开启**，无需额外参数。
- `--num-workers 4` 需要 CPU-only torchcodec 构建（预构建镜像已包含；手动配置见上文）。

</details>

<details>
<summary><b>评估（04_eval_custom_scene.py）</b></summary>

```bash
# CDNA3 (MI300/MI325) — 必须加 --render-cpu（无 GPU 图形驱动）
python scripts/04_eval_custom_scene.py \
  --checkpoint output/train/smolvla_kitchen_wrist/final \
  --dataset-id lidavidsh/franka-pick-kitchen-up-wrist-100ep-genesis \
  --scene rustic_kitchen --anchor floor_origin \
  --camera-layout up_wrist --render-cpu \
  --n-episodes 20 --seed 99 --record-video

# RDNA4 — 不加 --render-cpu，默认走 GPU radeonsi
python scripts/04_eval_custom_scene.py \
  --checkpoint output/train/smolvla_kitchen_wrist/final \
  --dataset-id lidavidsh/franka-pick-kitchen-up-wrist-100ep-genesis \
  --scene rustic_kitchen --anchor floor_origin \
  --camera-layout up_wrist \
  --n-episodes 20 --seed 99 --record-video
```

- **先下载厨房资源**：`python scripts/00_download_kitchen.py`（仅首次，~130 MB）。notebook 第 0 节会自动执行；CLI 方式需手动运行。
- `--camera-layout up_wrist` **必须**与数据集匹配；不传会加载世界固定侧视相机。
- `--render-cpu`（仅 CDNA3）强制 CPU llvmpipe 路径，成功率会比 GPU 渲染低约 20 pt——详见[附录 A](#附录-a渲染后端--cdna3-vs-rdna4)。
- MI300/MI325 首次 Genesis CPU 编译（`scene.build()`）需 **20-30 分钟**；后续运行复用 Taichi 内核缓存。预构建 Docker 镜像已包含此缓存，无需等待。

</details>

<details>
<summary><b>数据生成（02_gen_data_custom_scene.py）— 可选，优先 RDNA4</b></summary>

```bash
python scripts/02_gen_data_custom_scene.py \
  --scene rustic_kitchen --anchor floor_origin \
  --camera-layout up_wrist \
  --n-episodes 100 --seed 42 \
  --repo-id local/kitchen-pick
```

</details>

---

## 参考结果

以下数据均来自 workshop 主路径（`kitchen-up-wrist`，100 集数据集）。

### 数据生成

| 架构 | 成功率 | 每集耗时 |
|---|:---:|:---:|
| RDNA4 (R9700) | 100/100 | **~14 s/ep**（含 Genesis 编译 + 视频编码，稳态 ~12 s/ep） |
| CDNA3 (MI300/MI325) | 100/100 | ~4× 慢，因走 CPU 软件光栅化 |

100 集端到端 wall clock（RDNA4）：**~23 min**，含 Genesis 场景编译和 SVT-AV1 编码。

### 训练（100 集，4000 步，batch 4 — 默认配方）

默认配方 = Video 格式 + `num-workers=4` + AMP BF16 + PyTorch SDPA auto（AMD 上走 AOTriton flash）。AMP/SDPA 在 `02_train_vla.py` 里 CUDA 可用时自动开启。

| 指标 | RDNA4 (R9700, ROCm 7.2) | CDNA3 (MI300/MI325, ROCm 6.4.3) |
|---|:---:|:---:|
| 训练耗时 | **~7.4 min** (444 s) | **~10.6 min** (637 s) |
| 每步耗时 | 0.111 s | 0.159 s |
| 峰值显存 | 2.33 GB | 2.24 GB |
| Loss（起始 → 结束） | 0.671 → 0.0161 | 0.671 → 0.0162 |

> ⚠️ per-step 数值不能用来横向比较 RDNA4 vs CDNA3 算力：SmolVLA 450M + batch=4 + 短序列严重 under-utilize CDNA3（显存仅用 ~1.2 %），per-step 被 kernel launch 而非 compute 主导。本表只用于 workshop 复现，不是芯片 benchmark。

### 评估（kitchen+wrist，5 个评估 seed × 20 trials）

成功率同时依赖训练栈和评估渲染后端。下面 2×2 矩阵用同一份 100 集数据集 + 同一套默认训练配方分离这两个维度：

| 训练栈 \ 评估渲染 | **MI300/MI325 CPU (llvmpipe)** | **RDNA4 GPU (radeonsi)** |
|---|:---:|:---:|
| CDNA3 (ROCm 6.4 + PyTorch 2.6) | 25.0 % | 45.0 % |
| RDNA4 (ROCm 7.2 + PyTorch 2.9) | — (未测) | **48.0 %** |

核心结论：

1. **评估渲染后端主导性能**：同一 checkpoint，CPU → GPU 评估可提升 ~20 pt（25 % → 45 %）。MI300/MI325 上的评估数字会系统性偏低；如需 benchmark 级结果，请在 RDNA4 或其它支持 GPU 渲染的节点上评估。
2. **训练栈等效**：CDNA3 训练和 RDNA4 训练的 checkpoint 在同等 GPU 评估条件下成绩 45 % vs 48 %，差距在评估自身标准差（~10 pt）之内。MI300/MI325 完全可作训练节点。
3. **单 seed 方差大**：不同评估 seed 成功率在 35-60 % 区间波动；建议报告 pooled 或 mean ± std（≥3 seeds）。

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
  ▲ 场景来源：                                                 │   [50步]     │
  │ (a) 平面 (01)                                             │              │
  │ (b) 厨房 GLB (02)           相同 LeRobot 格式              └──────────────┘

评估循环：
  渲染 ─────────────────────────────────── 推理 ──────────────▶ 动作分块
  读取关节状态 ──────────────────────────── 预测 ──────────────▶ 目标关节角
  执行 action[0] ──────── PD 控制 ──────── scene.step()
```

---

## 附录 A：渲染后端 — CDNA3 vs RDNA4

| 架构 | EGL 渲染器 | 类型 |
|---|---|---|
| CDNA3 (MI300/MI325 系列) | llvmpipe | CPU 软件光栅化 |
| RDNA4 (R9700) | radeonsi | **GPU 硬件光栅化** |

CDNA3 没有图形流水线，Genesis 相机渲染回退到 CPU `llvmpipe`。RDNA4 有完整图形流水线（`radeonsi` 硬件加速），这是数据生成快 3-4× 且评估不受 render-gap bias 影响的根本原因。

**MI300/MI325 CPU 渲染评估 bias**：CPU 和 GPU 光栅化器输出的画面在视觉分布上不同。策略在 GPU 渲染数据上训练、用 CPU 渲染评估时，成功率会系统性低约 **20 pt**。kitchen+wrist 主路径实测：RDNA4 GPU 评估 pooled ~45-48 % vs MI300/MI325 CPU 评估 pooled ~25 %。这是预期行为，不是 bug。需要 benchmark 级结果请在 RDNA4 或其它带 GPU 渲染的节点上评估。

## 附录 B：已知兼容性问题

| 问题 | 修复 | 自动处理于 |
|---|---|---|
| Genesis PyPI 0.4.5 导入 `cuda.bindings`（ROCm 上不存在） | 从 `main` 分支安装（已在 [`e807698`](https://github.com/Genesis-Embodied-AI/Genesis/commit/e807698b8aa773fad3a6dfb4556889b251c30924) 修复） | Dockerfile / `fix_and_run.sh` |
| numpy / scikit-image ABI 不兼容（`numpy.dtype size changed`） | `pip install --force-reinstall "scikit-image>=0.22" "numpy==2.1.2"` | Dockerfile / `fix_and_run.sh` |
| torchcodec pip wheel 链接 CUDA 库，ROCm 上无法导入 | `bash setup_torchcodec.sh`（CPU-only 构建） | Dockerfile / `fix_and_run.sh` |
| `lerobot>=0.5.0` SmolVLAConfig dataclass 字段排序错误 | 锁定 `lerobot==0.4.4` | Dockerfile / `fix_and_run.sh` |

---

## 参考资料

- [LeRobot](https://github.com/huggingface/lerobot) — 机器人学习框架（数据集 + 策略模型）
- [Genesis](https://genesis-embodied-ai.github.io/) — GPU 加速物理仿真（通过 Taichi 原生支持 ROCm）
- [SmolVLA](https://huggingface.co/blog/smolvla) — 视觉-语言-动作模型
- [World Labs Marble](https://marble.worldlabs.ai/) — 3D 场景生成，用于自定义仿真环境
- [AMD ROCm 文档](https://rocm.docs.amd.com/)
