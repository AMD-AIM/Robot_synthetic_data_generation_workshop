[English](README.md) | [中文](README-cn.md)

# 机器人合成数据生成 Workshop

基于 **AMD GPU (ROCm)** 的机器人操作全流程：**合成数据生成 → VLA 训练 → 仿真评估**。

已在 **CDNA3 (MI300 系列)** 和 **RDNA4 (Radeon AI PRO R9700)** 上验证。

```
                 预生成于 RDNA4，发布到 HuggingFace
┌──────────────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
│ HF 数据集 (kitchen+wrist) │─▶  │ 02_train_vla.py      │ ─▶ │ 04_eval_custom_scene │
│ 100 集 · 13,500 帧        │    │ SmolVLA 后训练        │    │ 闭环仿真评估          │
│ 双相机：俯视 + 腕部        │    │ 冻结视觉编码器,       │    │ 成功率 + 视频         │
│ AV1 视频                  │    │ 仅训 expert + proj   │    │ 渲染 → VLA → PD 控制   │
└──────────────────────────┘     └─────────────────────┘     └──────────────────────┘

     可选 ────┐
     02_gen_data_custom_scene.py （重新生成数据 — 优先 RDNA4，3-4× 加速）
```

**Workshop 运行链路**：workshop 在 **CDNA3 (MI300 系列)** 节点上进行训练和评估。数据生成已预先在 RDNA4 上完成并发布到 HuggingFace，学员**无需在 workshop 期间生成数据**（notebook 中保留了 2-3 集 demo 用于讲解数据结构）。若需要 benchmark 质量的评估数字，优先使用 RDNA4 节点——MI300 走 CPU 软件光栅化，评估成功率会系统性低约 20 pt（见[附录 A](#附录-a渲染后端--cdna3-vs-rdna4)）。

---

## 数据集（快速路径）

Workshop 主路径的数据集已预生成于 RDNA4 并发布到 HuggingFace。**Workshop 期间无需运行数据生成**，直接拉数据进入训练：

```bash
pip install lerobot==0.4.4 torchcodec

python scripts/02_train_vla.py \
  --dataset-id lidavidsh/franka-pick-kitchen-up-wrist-100ep-genesis \
  --n-steps 4000 --batch-size 4 --num-workers 4 \
  --save-dir outputs/smolvla_kitchen_wrist
```

或在 Python 中直接加载：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset("lidavidsh/franka-pick-kitchen-up-wrist-100ep-genesis")
print(f"Episodes: {dataset.meta.total_episodes}, Frames: {len(dataset)}")
```

| 项目 | 值 |
|---|---|
| 场景 | Rustic kitchen GLB + Franka Panda 抓取红色方块 |
| 相机配置 | `up`（俯视）+ `side`（**腕部固定 / 眼在手上**），640×480 |
| 集数 / 帧数 | 100 / 13,500 |
| 大小 | ~200 MB（AV1 视频，LeRobot v3.0） |
| 动作空间 | 9-DoF 关节位置（7 臂 + 2 指） |
| 生成环境 | RDNA4 (Radeon AI PRO R9700)，Genesis 0.4.5，`seed=42` |
| HF 仓库 | [`lidavidsh/franka-pick-kitchen-up-wrist-100ep-genesis`](https://huggingface.co/datasets/lidavidsh/franka-pick-kitchen-up-wrist-100ep-genesis) |

> ⚠️ 该数据集的 `observation.images.side` 存储的是**腕部相机（眼在手上）**，不是世界固定侧视。不要与旧的 `lidavidsh/franka-pick-kitchen-100ep-genesis`（up + 世界侧视）做 concat / 混合训练——key 名相同但语义冲突。

---

## 快速开始（完整流程）

整个 Workshop 通过 `workshop_pipeline.ipynb` 进行，在远端 AMD GPU 节点的 Docker 容器内运行。

### 第 1 步 — 登录 GPU 节点

```bash
ssh -A <你的用户名>@<gpu节点>
```

### 第 2 步 — 克隆仓库

```bash
git clone git@github.com:<组织>/Robot_synthetic_data_generation_workshop.git
cd Robot_synthetic_data_generation_workshop
```

### 第 3 步 — 启动 Docker 容器

<details open>
<summary><b>CDNA3 (MI300 系列) — ROCm 6.x — workshop 主节点</b></summary>

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

这是 workshop 期间学员运行训练和评估的容器。

</details>

<details>
<summary><b>RDNA4 (R9700) — ROCm 7.2 — 优先用于数据生成 / benchmark 评估</b></summary>

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

> RDNA4 不需要 `HSA_OVERRIDE_GFX_VERSION`——ROCm 7.2 原生支持 gfx1201。RDNA4 有完整图形流水线（radeonsi 硬件光栅化），因此数据生成快 3-4×，评估成功率也不会受 MI300 的 CPU 渲染 bias 影响。

</details>

> `-it` 参数用于进入交互式 shell，后续步骤都在容器内执行。

### 第 4 步 — 安装依赖（容器内）

```bash
# Python 依赖
pip install -q git+https://github.com/Genesis-Embodied-AI/Genesis.git@main \
  lerobot==0.4.4 transformers accelerate safetensors \
  matplotlib Pillow jupyter ipykernel

# 修复 numpy / scikit-image ABI 不兼容（Genesis 要求 numpy==2.1.2）
pip install --force-reinstall --no-cache-dir -q "scikit-image>=0.22" "numpy==2.1.2"

# 系统依赖：无头渲染 + 视频编码
apt-get update -qq && apt-get install -y -qq xvfb ffmpeg > /dev/null 2>&1
```

#### 第 4b 步 — 构建 torchcodec（CPU-only）— 必做

HF 数据集用 AV1 视频存储观测，训练时靠 `torchcodec` 解码。pip wheel 链接 NVIDIA CUDA 库，ROCm 下无法 import。`torchcodec` 的 GPU 解码路径仅支持 NVDEC——AMD 有对等硬件（VCN）但上游尚无 VA-API 后端，所以只能走 CPU `libavcodec`。实际不影响训练性能（视频 I/O 远小于 GPU 前向/反向耗时）。

```bash
bash setup_torchcodec.sh   # ~3-5 min，克隆 v0.10.0 + CPU-only 构建
```

跳过此步会导致 HF 数据集加载（第 1c 节）和训练（第 2 节）crash。

> 也可以执行 `bash fix_and_run.sh` 一键完成上述所有步骤并自动跑完 notebook。但建议手动启动 Jupyter，逐 cell 执行以理解 pipeline。

### 第 5 步 — 启动 Jupyter Notebook

```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

在本地终端打开 SSH 隧道：

```bash
ssh -L 8888:localhost:8888 <你的用户名>@<gpu节点>
```

浏览器打开 `http://localhost:8888`，进入 `workshop_pipeline.ipynb`，按顺序执行。

---

## Notebook 内容概览

| 章节 | 内容 | 产出 |
|------|------|------|
| **0. 环境配置** | GPU 检测、依赖验证、厨房资源下载、HF 数据集缓存 | 环境就绪，数据集本地缓存 |
| **1. 数据生成（演示）** | 2-3 集 IK 轨迹演示，讲解数据结构（不做 100 集全量——主数据集从 HuggingFace 拉取） | 示例数据集 + 相机/轨迹可视化 |
| **2. VLA 训练** | 基于 HF `kitchen-up-wrist` 数据集的 SmolVLA 后训练（冻结视觉编码器） | 模型 Checkpoint + Loss 曲线 |
| **3. 仿真评估** | 厨房场景闭环评估（MI300 CPU 渲染 bias 见[附录 A](#附录-a渲染后端--cdna3-vs-rdna4)） | 成功率 + 评估视频 |
| **4. 结果汇总** | 全部产出物收集与展示 | PNG / MP4 / JSON |

---

## 文件结构

```
robot_synthetic_data_generation_workshop/
├── README.md                        ← 英文说明
├── README-cn.md                     ← 本文件（中文说明）
├── workshop_pipeline.ipynb          ← ★ Jupyter Notebook（Workshop 主体）
├── fix_and_run.sh                   ← 一键执行：安装依赖 + ROCm 补丁 + 运行 notebook
├── setup_torchcodec.sh              ← 构建 torchcodec v0.10.0 CPU-only（ROCm 用）
├── images/                          ← 预生成的可视化（notebook 内引用）
│   ├── ep0_camera_views.png         ← Franka 抓取过程双相机视角
│   ├── ep0_joint_trajectory.png     ← 9 自由度关节轨迹曲线
│   ├── cube_scatter_kitchen.png     ← 方块生成位置散点图（厨房场景）
│   └── kitchen_wrist/              ← 厨房场景 up + wrist 相机示例帧
├── scenes/
│   └── rustic_kitchen.json          ← 厨房场景配置（锚点、Mesh 引用）
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
| `numpy` | ==2.1.2 | Genesis 依赖，需与 scikit-image 的 C 扩展 ABI 匹配 |
| `scikit-image` | ≥0.22 | 需在 numpy==2.1.2 下重新编译 |
| `xvfb` | 系统包 | 无头渲染（apt-get 安装） |
| `ffmpeg` | 系统包 | 视频编码（apt-get 安装） |

**硬件要求**：CDNA3 (AMD Instinct MI300 系列, ROCm 6.x) **或** RDNA4 (AMD Radeon AI PRO R9700, ROCm 7.2)，任一架构显存 ≥4 GB。

---

## 脚本速查

<details>
<summary><b>训练（02_train_vla.py）</b></summary>

```bash
python scripts/02_train_vla.py \
  --dataset-id lidavidsh/franka-pick-kitchen-up-wrist-100ep-genesis \
  --n-steps 4000 --batch-size 4 --num-workers 4 \
  --save-dir /output/outputs/workshop_smolvla_kitchen_wrist
```

- AMP BF16 + PyTorch SDPA auto-dispatch（AMD 上走 AOTriton flash）在 CUDA 可用时**自动开启**，无需额外参数。
- `--num-workers 4` 需要先完成 CPU-only torchcodec 构建（见[第 4b 步](#第-4b-步--构建-torchcodeccpu-only-必做)）。

</details>

<details>
<summary><b>评估（04_eval_custom_scene.py）</b></summary>

```bash
# CDNA3 (MI300) — 必须加 --render-cpu（无 GPU 图形驱动）
python scripts/04_eval_custom_scene.py \
  --checkpoint /output/outputs/workshop_smolvla_kitchen_wrist/final \
  --dataset-id lidavidsh/franka-pick-kitchen-up-wrist-100ep-genesis \
  --scene rustic_kitchen --anchor floor_origin \
  --camera-layout up_wrist --render-cpu \
  --n-episodes 20 --seed 99 --record-video \
  --save /output/eval_kitchen_wrist

# RDNA4 — 不加 --render-cpu，默认走 GPU radeonsi
python scripts/04_eval_custom_scene.py \
  --checkpoint ... --dataset-id ... \
  --scene rustic_kitchen --anchor floor_origin \
  --camera-layout up_wrist \
  --n-episodes 20 --seed 99 --record-video \
  --save /output/eval_kitchen_wrist
```

- `--camera-layout up_wrist` **必须**与数据集匹配；不传会加载世界固定侧视相机。
- `--render-cpu`（仅 CDNA3）强制 CPU llvmpipe 路径，成功率会比 GPU 渲染低约 20 pt——详见[附录 A](#附录-a渲染后端--cdna3-vs-rdna4)。

</details>

<details>
<summary><b>数据生成（02_gen_data_custom_scene.py）— 可选，优先 RDNA4</b></summary>

```bash
python scripts/02_gen_data_custom_scene.py \
  --scene rustic_kitchen --anchor floor_origin \
  --camera-layout up_wrist \
  --n-episodes 100 --seed 42 \
  --repo-id local/kitchen-pick \
  --save /output
```

</details>

---

## 参考结果

以下数据均来自 workshop 主路径（`kitchen-up-wrist`，100 集数据集）。

### 数据生成

| 架构 | 成功率 | 每集耗时 |
|---|:---:|:---:|
| RDNA4 (R9700) | 100/100 | **~14 s/ep**（含 Genesis 编译 + 视频编码，稳态 ~12 s/ep） |
| CDNA3 (MI300) | 100/100 | ~4× 慢，因走 CPU 软件光栅化 |

100 集端到端 wall clock（RDNA4）：**~23 min**，含 Genesis 场景编译和 SVT-AV1 编码。

### 训练（100 集，4000 步，batch 4 — 默认配方）

默认配方 = Video 格式 + `num-workers=4` + AMP BF16 + PyTorch SDPA auto（AMD 上走 AOTriton flash）。AMP/SDPA 在 `02_train_vla.py` 里 CUDA 可用时自动开启。

| 指标 | RDNA4 (R9700, ROCm 7.2) | CDNA3 (MI300, ROCm 6.4.3) |
|---|:---:|:---:|
| 训练耗时 | **~7.4 min** (444 s) | **~10.6 min** (637 s) |
| 每步耗时 | 0.111 s | 0.159 s |
| 峰值显存 | 2.33 GB | 2.24 GB |
| Loss（起始 → 结束） | 0.671 → 0.0161 | 0.671 → 0.0162 |

> ⚠️ per-step 数值不能用来横向比较 RDNA4 vs CDNA3 算力：SmolVLA 450M + batch=4 + 短序列严重 under-utilize CDNA3（显存仅用 ~1.2 %），per-step 被 kernel launch 而非 compute 主导。本表只用于 workshop 复现，不是芯片 benchmark。

### 评估（kitchen+wrist，5 个评估 seed × 20 trials）

成功率同时依赖训练栈和评估渲染后端。下面 2×2 矩阵用同一份 100 集数据集 + 同一套默认训练配方分离这两个维度：

| 训练栈 \ 评估渲染 | **MI300 CPU (llvmpipe)** | **RDNA4 GPU (radeonsi)** |
|---|:---:|:---:|
| CDNA3 (ROCm 6.4 + PyTorch 2.6) | 25.0 % | 45.0 % |
| RDNA4 (ROCm 7.2 + PyTorch 2.9) | — (未测) | **48.0 %** |

核心结论：

1. **评估渲染后端主导性能**：同一 checkpoint，CPU → GPU 评估可提升 ~20 pt（25 % → 45 %）。MI300 上的评估数字会系统性偏低；如需 benchmark 级结果，请在 RDNA4 或其它支持 GPU 渲染的节点上评估。
2. **训练栈等效**：CDNA3 训练和 RDNA4 训练的 checkpoint 在同等 GPU 评估条件下成绩 45 % vs 48 %，差距在评估自身标准差（~10 pt）之内。MI300 完全可作训练节点。
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
| CDNA3 (MI300 系列) | llvmpipe | CPU 软件光栅化 |
| RDNA4 (R9700) | radeonsi | **GPU 硬件光栅化** |

CDNA3 没有图形流水线，Genesis 相机渲染回退到 CPU `llvmpipe`。RDNA4 有完整图形流水线（`radeonsi` 硬件加速），这是数据生成快 3-4× 且评估不受 render-gap bias 影响的根本原因。

**MI300 CPU 渲染评估 bias**：CPU 和 GPU 光栅化器输出的画面在视觉分布上不同。策略在 GPU 渲染数据上训练、用 CPU 渲染评估时，成功率会系统性低约 **20 pt**。kitchen+wrist 主路径实测：RDNA4 GPU 评估 pooled ~45-48 % vs MI300 CPU 评估 pooled ~25 %。这是预期行为，不是 bug。需要 benchmark 级结果请在 RDNA4 或其它带 GPU 渲染的节点上评估。

## 附录 B：已知兼容性问题

| 问题 | 修复 | 自动处理于 |
|---|---|---|
| Genesis PyPI 0.4.5 导入 `cuda.bindings`（ROCm 上不存在） | 从 `main` 分支安装（已在 [`e807698`](https://github.com/Genesis-Embodied-AI/Genesis/commit/e807698b8aa773fad3a6dfb4556889b251c30924) 修复） | 第 4 步 / `fix_and_run.sh` |
| numpy / scikit-image ABI 不兼容（`numpy.dtype size changed`） | `pip install --force-reinstall "scikit-image>=0.22" "numpy==2.1.2"` | 第 4 步 / `fix_and_run.sh` |
| torchcodec pip wheel 链接 CUDA 库，ROCm 上无法导入 | `bash setup_torchcodec.sh`（CPU-only 构建） | 第 4b 步 / `fix_and_run.sh` |
| `lerobot>=0.5.0` SmolVLAConfig dataclass 字段排序错误 | 锁定 `lerobot==0.4.4` | 第 4 步 / `fix_and_run.sh` |

---

## 参考资料

- [LeRobot](https://github.com/huggingface/lerobot) — 机器人学习框架（数据集 + 策略模型）
- [Genesis](https://genesis-embodied-ai.github.io/) — GPU 加速物理仿真（通过 Taichi 原生支持 ROCm）
- [SmolVLA](https://huggingface.co/blog/smolvla) — 视觉-语言-动作模型
- [World Labs Marble](https://marble.worldlabs.ai/) — 3D 场景生成，用于自定义仿真环境
- [AMD ROCm 文档](https://rocm.docs.amd.com/)
