[English](README.md) | [中文](README-cn.md)

# 机器人合成数据生成 Workshop

基于 **AMD MI300X (ROCm)** 的机器人操作全流程：**合成数据生成 → VLA 训练 → 仿真评估**。

```
┌──────────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│ 01_gen_data.py (默认场景)  │     │  02_train_vla.py     │     │  03_eval.py          │
│   平面 + 红色方块          │     │                      │     │                      │
│ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│────▶│  SmolVLA 后训练       │────▶│  闭环仿真评估         │
│ 02_gen_data_custom_scene │     │  基于 LeRobot 数据集   │     │  在 Genesis 中评估     │
│   厨房 GLB + 锚点         │     │  输出 HF checkpoint   │     │  成功率 + 视频         │
└──────────────────────────┘     └─────────────────────┘     └─────────────────────┘
     Franka 7自由度                   lerobot/smolvla_base       渲染 → VLA → PD 控制
     抓取红色方块                      冻结视觉编码器              动作分块预测
     双相机 (俯视 + 侧视)              仅训练 expert + state_proj  随机化方块位置
```

---

## 快速开始

整个 Workshop 通过 `workshop_pipeline.ipynb` 进行，在远端 AMD MI300X GPU 节点的 Docker 容器内运行。

Notebook 已内嵌了预生成的可视化图片（`images/` 目录），即使不运行也可以直接阅读理解整个流程。

### 第 1 步 — 登录 GPU 节点

```bash
ssh -A <你的用户名>@<mi300x节点>
```

### 第 2 步 — 克隆仓库

```bash
git clone git@github.com:<组织>/Robot_synthetic_data_generation_workshop.git
cd Robot_synthetic_data_generation_workshop
```

### 第 3 步 — 启动 Docker 容器

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

> `-it` 参数用于进入交互式 shell，后续步骤都在容器内执行。

### 第 4 步 — 安装依赖（容器内）

```bash
# Python 依赖
pip install -q genesis-world lerobot transformers accelerate safetensors \
  matplotlib Pillow jupyter ipykernel

# 修复 numpy / scikit-image ABI 不兼容（Genesis 要求 numpy==2.1.2）
pip install --force-reinstall --no-cache-dir -q "scikit-image>=0.22" "numpy==2.1.2"

# 系统依赖：无头渲染 + 视频编码
apt-get update -qq && apt-get install -y -qq xvfb ffmpeg > /dev/null 2>&1

# 应用 Genesis ROCm 补丁（详见下方「ROCm 适配」章节）
# fix_and_run.sh 的第 3 步会自动处理
```

> 也可以执行 `bash fix_and_run.sh` 一键完成上述所有步骤并自动跑完 notebook。但建议手动启动 Jupyter，逐 cell 执行以理解 pipeline。

### 第 5 步 — 启动 Jupyter Notebook

```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

在本地终端打开 SSH 隧道：

```bash
ssh -L 8888:localhost:8888 <你的用户名>@<mi300x节点>
```

浏览器打开 `http://localhost:8888`，进入 `workshop_pipeline.ipynb`，按顺序执行。

---

## Notebook 内容概览

| 章节 | 内容 | 产出 |
|------|------|------|
| **0. 环境配置** | GPU 检测、依赖验证、厨房场景资源下载 | 环境就绪 |
| **1. 数据生成** | 平面场景 + 厨房场景 IK 轨迹生成 | LeRobot 数据集 + 可视化 |
| **2. VLA 训练** | SmolVLA 后训练（冻结视觉编码器） | 模型 Checkpoint + Loss 曲线 |
| **3. 仿真评估** | 闭环评估（渲染→推理→执行→物理更新） | 成功率 + 评估视频 |
| **4. 结果汇总** | 全部产出物收集与展示 | PNG / MP4 / JSON |

每个章节都包含：
- **背景说明** — 为什么要做这一步、技术原理是什么
- **可执行代码** — 直接运行即可
- **内嵌可视化** — 预生成图片 + 运行时 matplotlib 实时绘图

---

## 文件结构

```
robot_synthetic_data_generation_workshop/
├── README.md                        ← 英文说明
├── README-cn.md                     ← 本文件（中文说明）
├── workshop_pipeline.ipynb          ← ★ Jupyter Notebook（Workshop 主体）
├── fix_and_run.sh                   ← 一键执行：安装依赖 + ROCm 补丁 + 运行 notebook
├── run_pipeline.sh                  ← Shell 一键管线（无 notebook）
├── images/                          ← 预生成的可视化（notebook 内引用）
│   ├── ep0_camera_views.png         ← Franka 抓取过程双相机视角
│   ├── ep0_joint_trajectory.png     ← 9 自由度关节轨迹曲线
│   ├── cube_scatter_flat.png        ← 方块生成位置散点图（平面场景）
│   └── cube_scatter_kitchen.png     ← 方块生成位置散点图（厨房场景）
├── scenes/
│   └── rustic_kitchen.json          ← 厨房场景配置（锚点、Mesh 引用）
└── scripts/
    ├── 00_download_kitchen.py       ← 下载厨房 GLB 资源
    ├── 01_gen_data.py               ← 数据生成（平面场景）
    ├── 02_gen_data_custom_scene.py  ← 数据生成（自定义 3D 场景）
    ├── 02_train_vla.py              ← SmolVLA 后训练
    ├── 03_eval.py                   ← 闭环评估（平面场景）
    ├── 04_eval_custom_scene.py      ← 闭环评估（自定义场景）
    ├── genesis_scene_utils.py       ← Genesis 工具函数
    ├── pick_common.py               ← 场景无关的抓取任务构建器
    └── scene_placement.py           ← 机器人局部坐标系工具
```

---

## 依赖

| 包名 | 版本 | 用途 |
|---|---|---|
| `genesis-world` | ≥0.4.1 | 物理仿真 + 渲染（Taichi 后端，原生支持 ROCm） |
| `lerobot` | ≥0.4.4 | 数据集格式 + SmolVLA 模型 |
| `torch` | ≥2.1 (ROCm) | 训练与推理 |
| `transformers` | ≥4.40 | SmolVLA 骨干网络 (Idefics3) |
| `accelerate` | 最新 | HuggingFace 模型加载 |
| `numpy` | ==2.1.2 | Genesis 依赖，需与 scikit-image 的 C 扩展 ABI 匹配 |
| `scikit-image` | ≥0.22 | 需在 numpy==2.1.2 下重新编译 |
| `xvfb` | 系统包 | 无头渲染（apt-get 安装） |
| `ffmpeg` | 系统包 | 视频编码（apt-get 安装） |

**硬件要求**：AMD Instinct MI300X，ROCm 6.x，显存 ≥4 GB

---

## ROCm 适配

以下修复均由 `fix_and_run.sh` 自动处理。仅在手动配置环境时需要关注。

### 1. Genesis `cuda.bindings` 补丁

Genesis 内部调用 `from cuda.bindings import runtime` 查询 GPU 共享内存大小，该模块在 ROCm 上不存在。补丁将其包裹在 try-except 中，使用 MI300X 的 LDS 大小（64 KB）作为回退值：

```python
# genesis/engine/solvers/rigid/rigid_solver.py
try:
    from cuda.bindings import runtime
    _, max_shared_mem = runtime.cudaDeviceGetAttribute(...)
    max_shared_mem /= 1024.0
except (ImportError, Exception):
    max_shared_mem = 64.0  # MI300X LDS 回退值
```

### 2. numpy / scikit-image ABI 修复

Docker 基础镜像中预装的 scikit-image 可能是基于不同版本的 numpy 编译的，运行时会报 `ValueError: numpy.dtype size changed`。需要在固定 numpy 版本的同时强制重新编译 scikit-image 的 C 扩展：

```bash
pip install --force-reinstall --no-cache-dir "scikit-image>=0.22" "numpy==2.1.2"
```

### 3. ROCm 相关脚本参数

```bash
python scripts/01_gen_data.py --no-bbox-detection --no-videos ...
python scripts/02_train_vla.py --num-workers 0 ...
python scripts/03_eval.py --no-bbox-detection ...
```

- `--no-bbox-detection` — 绕过 AMD GPU 上的边界框检测兼容性问题
- `--num-workers 0` — 避免 ROCm 下 torchcodec 多进程解码崩溃
- `--no-videos` — 以 PNG 存储图像（在挂载卷上速度更快）

### 4. SmolVLA 兼容性

`lerobot>=0.5.0` 中 `SmolVLAConfig` 可能存在 `dataclass` 字段排序问题。如遇 `TypeError: non-default argument follows default argument`，请使用 `lerobot==0.4.4` 或查看 [LeRobot 发布页](https://github.com/huggingface/lerobot/releases)。

---

## 参考结果

以下结果在 AMD MI300X (ROCm 6.x) 上验证。

### 数据生成

| 场景 | 集数 | 成功率 |
|---|---|---|
| 平面（默认） | 10 | **100%** |
| 厨房（自定义） | 10 | **100%** |

| 平面场景 | 厨房场景 |
|:---:|:---:|
| ![平面](./images/cube_scatter_flat.png) | ![厨房](./images/cube_scatter_kitchen.png) |

### 训练（100 集参考值）

| 指标 | 值 |
|---|---|
| 训练集数 | 100 |
| 步数 / 批量 | 2000 / 4 |
| Loss（起始 → 结束） | 0.346 → 0.022 |
| 耗时 | 约 78 分钟 |
| 峰值显存 | 2.2 GB |

### 评估（100 集参考值）

| 评估集 | 成功率 |
|---|---|
| 未见位置 (seed=99) | **4/10 = 40%** |
| 训练位置 (seed=42) | **5/10 = 50%** |

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

## 参考资料

- [LeRobot](https://github.com/huggingface/lerobot) — 机器人学习框架（数据集 + 策略模型）
- [Genesis](https://genesis-embodied-ai.github.io/) — GPU 加速物理仿真（通过 Taichi 原生支持 ROCm）
- [SmolVLA](https://huggingface.co/blog/smolvla) — 视觉-语言-动作模型
- [World Labs Marble](https://marble.worldlabs.ai/) — 3D 场景生成，用于自定义仿真环境
- [AMD ROCm 文档](https://rocm.docs.amd.com/)
