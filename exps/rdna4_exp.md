# RDNA4 (RX 9070) Workshop 复现实验

> 目标：在 RDNA4 架构上复现 Robot Synthetic Data Generation Workshop 全流程，
> 重点关注 **渲染性能** 和 **成功率**，与 CDNA3 (MI308X) 结果做对比。

---

## 实验总览

| Exp | 假设 | 状态 | 关键结果 | 结论 |
|-----|------|------|---------|------|
| R1-flat | RDNA4 渲染加速 data gen (flat) | ✅ done | 10/10=100%, 8.3s/ep | **3.4× faster** vs CDNA3 28s/ep |
| R1-kitchen | RDNA4 渲染加速 data gen (kitchen) | ✅ done | 10/10=100%, 200s/10ep | Kitchen GLB 场景也正常 |
| R2 | 全流程 PNG nw=0 (gen→train→eval) | ✅ done | unseen 10%, train 10% | 全流程跑通，与 CDNA3 B0 一致 |
| R3 | Video nw=4 训练加速 | ✅ done | 训练 24.5 min (3.0×) | 训练 3.0× 加速，GPU 利用率 96% |
| R4 | 100ep seed variance | ✅ done | mean=30%, std=18.5% | 100ep ceiling 不足，variance 极大 |
| **R5** | **200ep episode scaling** | ✅ done | **mean=52.4%, std=21.2%** | **mean +22pp，variance 仍高** |
| **O1** | **AMP BF16 降低 per-step latency** | 🔄 running | — | — |
| **O2** | **SDPA auto (AOTriton flash) 加速 attention** | ⏳ planned | — | — |

> **O 系列**：训练 Latency 优化实验（单卡 RDNA4），后续需针对 CDNA3 单独验证。

---

## 环境

| 项目 | 值 |
|------|-----|
| GPU | AMD Radeon AI PRO R9700 × 4 (RDNA4, gfx1201, 16GB GDDR6) |
| CPU | AMD Ryzen Threadripper PRO 9995WX 96-Cores @ 5460 MHz |
| ROCm | 7.2.0 |
| Docker 基础镜像 | `rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1` |
| Mesa | 25.0.7 (radeonsi + llvmpipe) |
| Genesis | 0.4.5 |
| LeRobot | 0.4.4 |
| transformers | 4.57.6 |
| PyTorch | 2.9.1+rocm7.2.0 |
| Python | 3.12.3 |
| Node | 10.161.176.9 |

### RDNA4 节点存储拓扑（2026-04-11 探测）

| 设备 | 容量 | 挂载点 | 当前状态 |
|------|:---:|--------|----------|
| `nvme2n1p2` | 1.8T | `/` (系统盘) | 100% 已满（不再放实验输出） |
| `nvme0n1p1` | 3.6T | `/dc1` | 可用约 1.8T |
| `nvme1n1p1` | 3.6T | `/dc2` | 可用约 1.6T |

备注：

- 当前节点不存在 `/dc`、`/data` 独立挂载（`/mnt` 存在但非大容量 NVMe 挂载点）。
- 后续实验数据、checkpoint、视频输出 **优先落盘到 `/dc1` 或 `/dc2`**，避免继续占用系统盘 `/`。
- 建议统一输出根目录：`/dc2/rdna4_output`（或 `/dc1/rdna4_output`），并在 Docker 中挂载为 `/output`。

---

## 背景与动机

### CDNA3 渲染瓶颈

在 MI308X (CDNA3) 上，Genesis 相机渲染走 Mesa EGL → **llvmpipe (CPU 软件光栅化)**，
导致 data gen 每 episode ~28s，这是全流程主要瓶颈。

### 为什么选 RDNA4

- RDNA4 有完整图形流水线（OpenGL 4.6, Vulkan 1.4, 硬件光栅化）
- Genesis Rasterizer 默认路径 = EGL + radeonsi → **零代码改动**即可 GPU 硬件加速渲染
- ROCm 7.2 原生支持 RDNA4

---

## 渲染后端验证

### EGL Device 探测结果

通过 EGL device enumeration 确认 RDNA4 硬件渲染可用：

```
Found 5 EGL devices
  Device 0: AMD Radeon AI PRO R9700 (radeonsi, gfx1201, LLVM 20.1.2, DRM 3.64)
  Device 1: AMD Radeon AI PRO R9700 (radeonsi, gfx1201, ...)
  Device 2: AMD Radeon AI PRO R9700 (radeonsi, gfx1201, ...)
  Device 3: AMD Radeon AI PRO R9700 (radeonsi, gfx1201, ...)
  Device 4: llvmpipe (LLVM 20.1.2, 256 bits)
```

- 4 个 GPU 设备使用 **radeonsi** 硬件渲染驱动
- 第 5 个是 llvmpipe (CPU fallback)
- Genesis EGL 平台代码通过 `eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, ...)` 自动选择 GPU 设备

### GLX vs EGL 的坑

| 方式 | 渲染器 | 说明 |
|------|--------|------|
| `xvfb-run` + GLX | **llvmpipe (CPU)** | Xvfb 虚拟 framebuffer 强制软件渲染 |
| EGL (PYOPENGL_PLATFORM=egl) | **radeonsi (GPU)** | Genesis 默认路径，零改动 |

**注意**：workshop 脚本 (`01_gen_data.py`, `03_eval.py`) 在无 DISPLAY 时自动启动 Xvfb，
但 Genesis EGL 平台代码在初始化时临时移除 DISPLAY 环境变量，因此仍走 EGL → radeonsi。

---

## Exp-R1: Data Generation 性能

### Flat Scene (10 episodes)

| 指标 | RDNA4 (R9700) | CDNA3 (MI308X) | 加速比 |
|------|:--:|:--:|:--:|
| 成功率 | **10/10 = 100%** | 100% | - |
| 总耗时 | **83s** | ~280s | **3.4×** |
| 每 episode | **~8.3s** | ~28s | **3.4×** |

### Flat Scene (100 episodes, PNG mode)

| 指标 | RDNA4 (R9700) | CDNA3 (MI308X) |
|------|:--:|:--:|
| 成功率 | **100/100 = 100%** | 100/100 = 100% |
| 总耗时 (video) | **638s** (~10.6 min) | ~2800s (~47 min) |
| 总耗时 (PNG) | **771s** (~12.9 min) | ~3080s (~51 min) |
| 每 ep (video) | **~6.4s** | ~28s |
| 每 ep (PNG) | **~7.7s** | ~30.8s |
| 加速比 | **3.4-4.4×** | baseline |

### Kitchen Scene (10 episodes)

| 指标 | RDNA4 (R9700) |
|------|:--:|
| 成功率 | **10/10 = 100%** |
| 总耗时 | **200s** (含 CoACD 分解 + mesh 加载) |
| 每 ep (稳态) | ~18-20s |

### 分析

- **RDNA4 data gen 加速 3.4-4.4×**，从 CDNA3 的 ~28s/ep 降到 ~6.4-8.3s/ep
- 加速主要来自 **EGL radeonsi GPU 硬件渲染**（vs CDNA3 llvmpipe CPU 软件渲染）
- 未达到理论预期的 5-10× 加速，可能因为：
  1. 物理仿真（Quadrants HIP kernel）仍有显著开销
  2. Python 循环 + IK 计算 + 数据保存等 CPU 开销
  3. video/PNG 编码也消耗时间
- Kitchen 场景因 mesh 复杂度高（126.9 MB GLB），渲染开销更大

---

## Exp-R2: 全流程复现 (100 ep + train + eval)

### 训练

```bash
python scripts/02_train_vla.py \
  --dataset-id local/rdna4-workshop-png-100ep \
  --n-steps 2000 --batch-size 4 --num-workers 0 \
  --save-dir /output/outputs/rdna4_smolvla
```

| 指标 | RDNA4 (R9700) | CDNA3 (MI308X) B0 | README 参考 (MI300X) |
|------|:--:|:--:|:--:|
| 训练时间 | **4463s (~74 min)** | ~4862s (~81 min) | ~78 min |
| GPU 计算 | 0.70-0.71 s/step | 0.23 s/step | - |
| 有效速度 (含 data loading) | ~2.23 s/step | ~2.43 s/step | - |
| Loss (start → end) | 0.323 → 0.015 | 0.416 → 0.016 | 0.346 → 0.022 |
| Peak VRAM | **2.33 GB** | ~4 GB | 2.2 GB |
| Epochs | ~0.6 | ~0.6 | - |

**分析**：
- RDNA4 GPU 计算速度 (0.70 s/step) 比 MI308X (0.23 s/step) 慢 **~3×**——
  MI308X 是数据中心卡（192GB HBM3 + 304 CU），算力远强于消费级 RDNA4（16GB GDDR6 + 32 CU）
- 但总训练时间反而接近（74 vs 81 min），原因是 **两边都严重 data-loading-bound**：
  - CDNA3：GPU 仅占 per-step 时间的 **9.5%**（0.23s / 2.43s），90% 时间在等 PNG 读取
  - RDNA4：GPU 占 per-step 时间的 **31%**（0.70s / 2.23s），69% 时间在等 PNG 读取
  - RDNA4 节点的 CPU 更快（Threadripper PRO 9995WX 96 核 @ 5460 MHz），PNG 解码速度更快（1.53s vs 2.20s/step）
- 结论：**3× 的 GPU 计算差距被 data loading 瓶颈稀释了**，RDNA4 更快的 CPU 反而部分弥补了 GPU 劣势
- 若切换到 Video 格式 + num_workers=4，data loading 瓶颈将大幅降低，届时 GPU 计算差距会更显著
- Peak VRAM 仅 2.33 GB，16GB RDNA4 完全够用

### 评估

| Seed | RDNA4 | CDNA3 B0 | README 参考 |
|:----:|:--:|:--:|:--:|
| 99 (unseen) | **1/10 = 10%** | 1/10 = 10% | 4/10 = 40% |
| 42 (training) | **1/10 = 10%** | 0/10 = 0% | 5/10 = 50% |

| 指标 | 值 |
|------|-----|
| Eval 时间 (unseen) | 94s |
| Eval 时间 (training) | 92s |
| 总 eval 时间 | 186s |

### 分析

1. **全流程跑通**：RDNA4 上 data gen → train → eval 完整可复现
2. **Eval 成功率与 CDNA3 B0 一致**（unseen 10%），远低于 README 参考值（40%）
3. **原因是 under-training**：2000 steps × batch 4 = 8000 samples seen，数据集 13500 frames → 仅 **0.6 epochs**
4. 这是已知问题（stage1_exp.md B0 分析一致），不是 RDNA4 特有的

---

## Exp-R3: Video 格式 + num_workers=4 训练加速

### torchcodec ROCm 修复

pip 安装的 torchcodec 二进制链接了 CUDA 库（libnvrtc.so.13, libcudart.so.13 等），
在 ROCm 上无法 import。解决方案：**从源码构建 torchcodec 0.10.0 CPU-only 版本**。

```bash
git clone --depth 1 --branch v0.10.0 https://github.com/pytorch/torchcodec.git
pip install pybind11
# 直接 cmake 构建，ENABLE_CUDA= 留空 → 不链接 CUDA
cmake ... -DENABLE_CUDA= -DTORCHCODEC_DISABLE_COMPILE_WARNING_AS_ERROR=ON
cmake --build . && cmake --install .
```

构建后的 .so 只依赖 FFmpeg + libtorch（无 CUDA），`import torchcodec` 和 `VideoDecoder` 均正常。

### Data Generation (100 ep, Video 格式)

```bash
python scripts/01_gen_data.py --n-episodes 100 --repo-id local/rdna4-video-100ep \
  --fps 30 --no-bbox-detection
```

| 指标 | Video 格式 | PNG 格式 (之前) |
|------|:--:|:--:|
| 成功率 | **100/100 = 100%** | 100/100 |
| 总耗时 | **629s** (~10.5 min) | 771s (~12.9 min) |
| 每 ep | **~6.3s** | ~7.7s |
| 磁盘占用 | ~160 MB | ~4.6 GB |

### 训练 (2000 steps, batch 4, num_workers=4)

```bash
python scripts/02_train_vla.py --dataset-id local/rdna4-video-100ep \
  --n-steps 2000 --batch-size 4 --num-workers 4 \
  --save-dir /output/outputs/rdna4_smolvla_video
```

| 指标 | Video nw=4 | PNG nw=0 (之前) | 加速比 |
|------|:--:|:--:|:--:|
| **训练时间** | **1468s (~24.5 min)** | 4463s (~74 min) | **3.0×** |
| per-step 有效速度 | **~0.73 s/step** | ~2.23 s/step | **3.0×** |
| GPU 计算 | 0.70 s/step | 0.70 s/step | 不变 |
| Data loading 开销 | **~0.03 s/step** | ~1.53 s/step | **51× 降低** |
| GPU 利用率 | **96%** | 31% | - |
| Loss (start → end) | 0.535 → 0.053 | 0.323 → 0.015 | - |
| Peak VRAM | 2.38 GB | 2.33 GB | - |

**分析**：
- Video + nw=4 将 data loading 开销从 1.53s 压缩到 ~0.03s，**训练从 data-loading-bound 变为 GPU-bound**
- GPU 利用率从 31% 提升到 **96%**，几乎逼近 GPU 计算理论下限（0.70s × 2000 = 1400s = 23.3 min）
- **3.0× 加速**与 CDNA3 上 PNG→Video 加速比一致（stage1_exp.md: 521s → 190s = 2.7×，nw=0 下）

### 评估

| Seed | Video nw=4 | PNG nw=0 (之前) | README 参考 |
|:----:|:--:|:--:|:--:|
| 99 (unseen) | **8/10 = 80%** | 1/10 = 10% | 4/10 = 40% |
| 42 (training) | **6/10 = 60%** | 1/10 = 10% | 5/10 = 50% |

**⚠️ Eval 提升与 Video/nw=4 无关，是训练随机性导致**：

两个数据集的 cube 位置**完全一致**（验证：两次 data gen 均使用默认 `--seed 42`，
`random.Random(42)` 产生的 100 个 cube (x,y) 逐一 MATCH）。

| | PNG nw=0 | Video nw=4 |
|--|:--:|:--:|
| 数据集 cube 位置 | seed=42, **完全相同** | seed=42, **完全相同** |
| Final loss | **0.015** (更低) | 0.053 |
| Eval unseen | 1/10 = 10% | 8/10 = 80% |

eval 差异的真正原因：
- 两次训练的 **PyTorch 随机种子不同**（脚本未固定 `torch.manual_seed`）
- 2000 steps × batch 4 = 8000 samples，数据集 13500 frames → 仅看了 **59% 的数据**
- **哪 59% 被采样到**决定了策略质量，这是纯随机的
- nw=0 vs nw=4 也改变了 dataloader 的采样/prefetch 行为
- **结论**：在 0.6 epoch 的低训练量下，eval 方差极大，80% 和 10% 都不代表稳定水平

### 4000 steps 对照实验（~1.2 epochs，覆盖全部数据）

**动机**：2000 steps 仅覆盖 59% 数据，eval 主要反映采样运气。4000 steps × batch 4 = 16000 samples，
数据集 13500 frames → **~1.2 epochs，所有数据至少过一遍**，eval 更有统计意义。

```bash
python scripts/02_train_vla.py --dataset-id local/rdna4-video-100ep \
  --n-steps 4000 --batch-size 4 --num-workers 4 \
  --save-dir /output/outputs/rdna4_smolvla_video_4k
```

**训练结果**：

| 指标 | 4000 steps | 2000 steps (对照) |
|------|:--:|:--:|
| 训练时间 | 2891s (~48.2 min) | 1468s (~24.5 min) |
| per-step 速度 | ~0.72 s/step | ~0.73 s/step |
| Epochs | ~1.2 | ~0.6 |
| Final loss | **0.033** | 0.053 |
| Loss 起始 | 0.722 | 0.535 |
| Peak VRAM | 2.38 GB | 2.38 GB |
| GPU 利用率 | ~100% | ~96% |

**评估结果（50-trial 定量，两次独立训练 run，未固定 seed）**：

| | 2k-run (0.6 epoch) | 4k-run (1.2 epochs) |
|--|:--:|:--:|
| **Unseen (seed=99)** | **36/50 = 72%** | 16/50 = 32% |
| **Training (seed=42)** | **40/50 = 80%** | 16/50 = 32% |
| Final loss | 0.053 | 0.033 |

> eval 需 ≥50 trials 才可信（n=10 方差极大：2k unseen 从 80%→50%→72%）。

---

## Checkpoint Curve：排除 Early Stopping 假说

对 4k-run 的 1k/2k/3k/4k checkpoints 跑 50-trial eval，验证 eval 下降是否因训练过久：

| Steps | Epochs | Unseen (seed=99, 50 trials) |
|:---:|:---:|:---:|
| 1000 | 0.3 | 15/50 = **30%** |
| 2000 | 0.6 | 14/50 = **28%** |
| 3000 | 0.9 | 14/50 = **28%** |
| 4000 (final) | 1.2 | 16/50 = **32%** |

曲线完全平坦 (~30%)，而独立 2k-run 在相同 2000 steps 时是 **72%**。

**结论**：2k-run vs 4k-run 的差距 (72% vs 32%) **100% 来自训练随机性**（未固定 seed），
不是 early stopping / overfitting。当前系统的主导问题是 **高 run-to-run variance**。

---

## Seed Variance 实验：100ep × 2k steps × 5 seeds

> 核心问题：**训练 run-to-run variance 有多大？100ep 数据是否足够稳定？**

使用 `train_with_seed.py` 固定 `torch.manual_seed` + `random.seed` + `np.random.seed`，
在同一 100ep 数据上跑 5 次独立训练（seed=0,1,2,3,4），每次 50-trial eval。

| Seed | Steps | Eval (50 trials, seed=99) | 训练时间 |
|:---:|:---:|:---:|:---:|
| 0 | 2000 | 12/50 = **24%** | 1433s |
| 1 | 2000 | 29/50 = **58%** | 1423s |
| 2 | 2000 | 4/50 = **8%** | 1421s |
| 3 | 2000 | 12/50 = **24%** | 1435s |
| 4 | 2000 | 18/50 = **36%** | 1422s |
| **mean ± std** | | **30.0% ± 18.5%** | ~24 min |

### 结论

- **mean = 30%**，**std = 18.5%**（range: 8% ~ 58%）
- 符合判断标准中 `mean < 50%` 情景：**100ep 数据 ceiling 不足**
- 即使固定 seed，5 次训练的 run-to-run variance 仍极大（CV = 62%）
- Seed=1 的 58% 是最优结果，但 seed=2 仅 8%——随机种子对结果的影响远大于超参调优
- 之前未固定 seed 的 72% (独立 2k-run) 本身就是一次 lucky run，不具代表性

---

## Episode Scaling 实验：200ep × 4k steps × 5 seeds

> 核心问题：**episode 翻倍能否提升 mean 并降低 variance？**

200ep data gen: 200/200 success, 1334s (~22 min), ~6.7s/ep (Video 格式)。

200ep × 4k steps = 16,000 samples / 27,000 frames ≈ **0.59 epochs**（与 100ep × 2k 的 epoch 数匹配）。

| Seed | Steps | Eval (50 trials, seed=99) | 训练时间 |
|:---:|:---:|:---:|:---:|
| 0 | 4000 | 35/50 = **70%** | 2844s |
| 1 | 4000 | 12/50 = **24%** | 2844s |
| 2 | 4000 | 37/50 = **74%** | 2852s |
| 3 | 4000 | 28/50 = **56%** | 2856s |
| 4 | 4000 | 19/50 = **38%** | 2863s |
| **mean ± std** | | **52.4% ± 21.2%** | ~48 min |

### 100ep vs 200ep 对比

| | 100ep × 2k (0.6 epoch) | 200ep × 4k (0.6 epoch) | 变化 |
|--|:--:|:--:|:--:|
| **mean** | 30.0% | **52.4%** | **+22.4pp** |
| **std** | 18.5% | 21.2% | +2.7pp |
| **CV** | 62% | **40%** | -22pp |
| **range** | 8% ~ 58% | 24% ~ 74% | 下限 +16pp |
| **best seed** | 58% (seed=1) | **74%** (seed=2) | +16pp |
| **worst seed** | 8% (seed=2) | 24% (seed=1) | +16pp |
| Data gen 时间 | ~11 min | ~22 min | 2× |
| 训练时间 | ~24 min | ~48 min | 2× |

### 结论

关键指标按重要性排序：

1. **worst-case ↑ (8% → 24%)**：完全失败的 run 基本消失，数据已足够避免灾难性结果
2. **best-case ↑ (58% → 74%)**：数据 unlock 了更好的策略，200ep 的 coverage 确实有用
3. **CV ↓ (62% → 40%)**：相对稳定性有改善，这才是"稳定性改善"的正确信号

### Regime 转变分析

| | 100ep (阶段 1) | 200ep (阶段 2) |
|--|--|--|
| **瓶颈** | data-limited | optimization-limited |
| **特征** | 大多数解都差，variance 来自"随机撞对" | 好策略已存在 (74%)，但能否 consistently 进入好 basin 取决于优化 |
| **结论** | 需要更多数据 | **需要更稳定的训练** |

**关键转变**：瓶颈从"数据不够"→"训练不稳"。继续加数据 (300ep/500ep) 收益递减——
因为 good basin 已经存在，问题是能不能稳定进去。

---

## 调试记录

| 轮次 | 问题 | 修复 | 结果 |
|------|------|------|------|
| r1 | Genesis `cuda.bindings` 在 ROCm 上 ImportError | 应用 rigid_solver.py try-except patch | OK |
| r2 | lerobot 0.5.1 API 不兼容（`lerobot.common` 移动） | 降级到 lerobot==0.4.4 | OK |
| r3 | transformers 5.x 与 lerobot 0.4.4 hub 版本冲突 | 安装 transformers 4.57.6 | OK |
| r4 | SmolVLM processor 缺 num2words | `pip install num2words` | OK |
| r5 | torchcodec 需要 libnvrtc.so.13 (CUDA) | 使用 `--no-videos` PNG 格式避免 torchcodec | OK |
| r6 | stdout 缓冲导致训练日志不输出 | 添加 `PYTHONUNBUFFERED=1` | OK |
| r7 | torchcodec pip wheel 链接 CUDA (.so) | 从源码构建 torchcodec 0.10.0 (CPU-only) | OK |

---

## ROCm 7.2 + RDNA4 关键发现

### 正面

1. **EGL 硬件渲染开箱即用**：Mesa 25.0.7 radeonsi 驱动，Genesis EGL 平台自动选择 GPU
2. **ROCm 7.2 原生支持 gfx1201**：无需 `HSA_OVERRIDE_GFX_VERSION`
3. **物理仿真 + 渲染 + 训练均正常**：全流程无 LLVM ISel 等编译问题
4. **16GB VRAM 完全够用**：SmolVLA 训练仅需 2.33 GB

### 需注意

1. **torchcodec CUDA 依赖**：pip 安装的 torchcodec 二进制链接了 CUDA 库，
   在 ROCm 上无法使用 → **从源码构建 torchcodec 0.10.0 CPU-only 版本**解决
2. **渲染加速有上限**：理论 20-60× 加速未实现，实际 3.4-4.4×，因为物理仿真、IK、Python 开销占比大
3. **lerobot 版本敏感**：0.5.x API 不兼容 workshop 脚本，需 0.4.4

---

## 性能对比总结

| 环节 | RDNA4 PNG nw=0 | RDNA4 Video nw=4 | CDNA3 (MI308X) | RDNA4 最优加速比 |
|------|:--:|:--:|:--:|:--:|
| **Data gen (per ep)** | ~7.7s | **~6.3s** | ~28s | **4.4×** |
| **Training (2K steps)** | ~74 min | **~24.5 min** | ~81 min | **3.3×** |
| **GPU 利用率** | 31% | **96%** | 9.5% | - |
| **渲染后端** | radeonsi (GPU) | radeonsi (GPU) | llvmpipe (CPU) | - |
| **Peak VRAM** | 2.33 GB | 2.38 GB | ~4 GB | - |
| **Eval success (unseen)** | 1/10 = 10% | 8/10 = 80% ⚠️ | 1/10 = 10% | ⚠️ 不同数据集，不可直接比 |

---

## 结论与 Next Step

### 结论

1. **RDNA4 是可行的 data gen 加速方案**：EGL 硬件渲染开箱即用，data gen **4.4× 加速**
2. **全流程可复现**：workshop pipeline 在 RDNA4 + ROCm 7.2 上完整跑通
3. **Video + nw=4 训练 3.0× 加速**：从 74 min 降到 24.5 min，GPU 利用率从 31% 提升到 96%
4. **torchcodec 已解决**：从源码构建 0.10.0 CPU-only 版本，Video 格式在 ROCm 上完全可用
5. **Episode scaling 有效但不够**：100ep→200ep mean 30%→52%，worst 8%→24%，best 58%→74%
6. **Regime 转变**：瓶颈从"数据不够"→"训练不稳"，继续加数据收益递减
7. **run-to-run variance 是当前主导瓶颈**：同数据同 epoch 数，仅换 seed 即可从 24% 到 74%

### Next Step

- [x] ~~**torchcodec ROCm 适配**~~：已通过源码构建 0.10.0 解决
- [x] ~~**Seed variance 实验 (100ep)**~~：5 seeds × 50 trials，mean=30%, std=18.5%
- [x] ~~**Episode scaling (200ep)**~~：mean 52.4%, std 21.2%，瓶颈从 data → optimization
- [x] ~~**R6: 200ep Checkpoint Curve**~~：已完成趋势版（基于已执行结果），确认存在明显 seed-specific peak，支持 checkpoint selection
- [ ] **R7: 降低 Optimization Variance**：详见下方方案总览
- [ ] **R8: DART 闭环数据**：等 decent policy 稳定后，验证 DART 能否进一步提升 mean 和鲁棒性

---

## R6：200ep Checkpoint Curve（趋势版，基于已完成结果）

> 说明：按你的要求不跑满 35 组，仅使用当前已完成的 50-trial 结果做趋势判断。
> 已覆盖：`seed3/seed4` 的 1k/2k/3k/4k；以及 `seed0~4` 的 final(4k)。

### 已完成中间 checkpoint 曲线（50 trials）

| Seed | 1000 | 2000 | 3000 | 4000(final) | peak |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 3 | 18/50 = **36%** | 11/50 = **22%** | 17/50 = **34%** | 28/50 = **56%** | **4000** |
| 4 | 11/50 = **22%** | 30/50 = **60%** | 24/50 = **48%** | 19/50 = **38%** | **2000** |

### 5 seeds final (4k) 参考

| Seed | Final(4k) |
|:---:|:---:|
| 0 | 35/50 = **70%** |
| 1 | 12/50 = **24%** |
| 2 | 37/50 = **74%** |
| 3 | 28/50 = **56%** |
| 4 | 19/50 = **38%** |

### 趋势判断

1. **不存在统一最优 step**：不同 seed 的 peak 出现在不同 checkpoint（seed3 在 final，seed4 在 2k）。
2. **中间 checkpoint 可能显著优于 final**：seed4 的 2k=60%，比 final=38% 高 **22pp**，说明 checkpoint selection 有现实收益。
3. **也存在 final 最优的 seed**：seed3 从 2k=22% 回升到 final=56%，不能一刀切 early stop。
4. **结论**：R6 支持“每个 seed 选本 seed 最佳 checkpoint”的策略；这是一种**零额外训练成本**的提 floor 手段。

### 注记

- 本轮按“趋势优先”策略停止继续补跑（未覆盖的 seed0/1/2 中间 checkpoint 暂不追加）。
- 若后续需要精确量化全量 ceiling，可在磁盘空间恢复后再补齐 5 seeds × 7 中间 checkpoints。

---

## R7 方案总览：降低 Optimization Variance

### 问题定义

200ep 数据已足够支撑好策略（best seed=74%），但 SGD 优化轨迹高度依赖随机种子，
不同 seed 落入不同 basin，导致 24%~74% 的巨大 spread。

目标：**提高 worst-case（floor），缩小 seed 间差距**。

### 现实预期

SmolVLA 的 loss landscape 是高度非凸的，200ep ≈ 27k frames 对 VLA 模型仍是极小数据集。
在小数据 regime 下 basin 间差异大，仅靠超参调整**无法完全消除 variance**。
现实预期：worst-case 从 24% → 40-50%，std 从 21% → 10-15%。
要让 any seed 都达到 70%+ 需要更多数据或方法改进（如 DART）。

### 候选手段

| 手段 | 机制 | 预期效果 | 局限 | 成本 |
|------|------|---------|------|------|
| **降 LR** | 梯度步更小，轨迹更平滑 | 减少跳过好 basin 的概率 | 可能收敛太慢，卡在更差的位置 | 低 |
| **增大 batch** | 梯度估计更准确，噪声更小 | 轨迹更确定性，减少随机性 | VRAM 限制；噪声有时反而帮助逃离差 basin | 低 |
| **EMA** | 对权重做滑动平均 | 平滑 final checkpoint，避免尾部震荡 | 不改变轨迹本身，只是选更好的终点 | 低 |
| **Warmup + cosine decay** | 前期大步探索，后期精细收敛 | 更好的探索-收敛平衡 | 小数据下效果不确定 | 低 |
| **Gradient clipping** | 限制单步更新幅度 | 避免极端梯度导致的轨迹偏移 | 可能已经默认开启 | 低 |

### 实际策略（按优先级）

1. **Checkpoint selection**（R6 结果驱动）：如果 R6 发现某些 seed 在中间 step 有明显 peak，
   则直接选 best checkpoint 即可提升 floor，零训练成本
2. **Best-of-N**：跑 3 个 seed，选最好的——实际部署中的常见做法，接受 variance 存在
3. **EMA + 降 LR**：如果 R6 显示曲线尾部震荡，EMA 是最直接的修复；降 LR 可同时尝试
4. **增大 batch**：如果 VRAM 允许（当前仅用 2.4/16 GB），batch 从 4→8 可直接降低梯度噪声

---

## R9：Overhead + Wrist 视角

> 假设：`overhead + wrist` 比 `overhead + side` 互补性更强（wrist 提供抓取局部接触信息）。

### 关键发现（必读）

> **Franka MJCF `hand` link-local 坐标系：z 轴朝下（朝桌面）。**
> 这意味着 `pos_z > 0` 把相机放到 hand **下方**（仰视），`pos_z < 0` 才是 hand **上方**（俯视）。
> 之前所有正 `pos_z` 的配置（M3 等）都是**仰视 gripper**，这是 0/10 success 的根本原因。

### 正确的 Wrist Cam 配置（D040）

```
mount:   hand link
pos:     (0.05, 0.00, -0.08)    # hand 前方 5cm，上方 8cm
lookat:  (0.00, 0.00,  0.10)    # 看向 hand 下方 10cm（指尖/cube）
up:      (0.00, 0.00, -1.00)    # hand-local -z = world 上方
fov:     65
```

**视角解读**（参考 `images/camera_views_compare/r9d_top1_D000.png`）：

这是标准 **eye-in-hand 俯视视角** — 相机从两指中间上方俯视抓取区域：

```
         camera 📷 (pos_z < 0 = hand 上方)
            │
            ▼  俯视
   ┌─ finger_L ─┐   ┌─ finger_R ─┐
   │             │   │             │
   │      ┌──────┴───┴──────┐      │
   │      │    ★ CUBE ★     │      │
   │      └─────────────────┘      │
   └─────────────────────────────┘
```

- `pos_y = 0`：左右居中 → 恰好在两个平行指之间
- `pos_x = 0.05`：hand 前方 5cm → 略偏向指尖方向
- `pos_z = -0.08`：hand 上方 8cm → 从上往下看
- 画面中两指对称出现在左右两侧，cube 在画面中央
- CNN 直接学到 finger-cube 对齐关系，无需隐式空间映射

设计原则：
- **overhead cam** 负责全局定位（world frame 俯视）
- **wrist cam** 负责接触控制（hand frame 俯视，grasp 阶段看到 fingers + cube）
- approach 阶段 wrist cam 看不到 cube 是正常的（overhead cam 覆盖）
- 两者**都是俯视**，但 reference frame 不同（world vs hand-local）

### 搜索历程（R9a → R9d）

| 阶段 | 做了什么 | 结论 |
|------|---------|------|
| R9a/b | 近距离 mount + lookat 扫描 | hand 近距离严重 self-occlusion |
| R9b-M3 | 大位移几何重置（0.28, 0, 0.22） | cube 可见但**方向反转**（仰视），E2E 0/10 |
| R9c | 自动化 sweep（192 候选 hand + link7） | link7 不可用；hand pos_z>0 全是仰视 |
| **R9d** | **方向修正**（pos_z<0 + lookat_z>0 + up=-z） | **72/81 viable，100% grasp_vis，方向正确** |

关键排错：
- `link7` link frame 与 `hand` 完全不同 → **只能用 `hand`**
- `pos_z > 0`（hand-local）= 相机在桌面仰视 → **CNN 学到反向空间关系，policy 学习难度 ↑**
- `pos_z < 0`（hand-local）= 相机在 hand 上方俯视 → **正确 eye-in-hand 视角**

### R9d Sweep 结果摘要

81 候选 × 1 ep，239s。搜索范围：px=[0.02,0.05,0.08], pz=[-0.05,-0.08,-0.12], lz=[0.05,0.10,0.15], fov=[55,65,80]

| pz（hand 上方） | g_area 范围 | grasp_vis | 适合 policy learning |
|-----------------|------------|-----------|---------------------|
| -0.05（5cm） | 6300-7300 | 100% | 偏近（cube 过大，motion sensitivity 高） |
| **-0.08（8cm）** | **3100-4700** | **100%** | **最佳平衡（detail + context）** |
| -0.12（12cm） | 1300-2700 | 100%* | 偏远（cube 较小）；px=0.02 穿入 arm |

选择 D040（px=0.05, pz=-0.08, lz=0.10, fov=65）：g_area=3145，适中距离。

图像存档（已清理）：
- `up_side/ep{0,1,2}` — side cam baseline
- `up_wrist/ep0` — 原始 self-occlusion 参考
- `wrist_dircal/m3/ep0` — M3 仰视（错误方向参考）
- `r9d_top1_D000.png` — **D040 俯视（正确方向）**

### R9e：E2E 实验（进行中）

D040 参数 → E2E 全流程，与 up_side 对标。

| 步骤 | 配置 | 状态 |
|------|------|------|
| 数据生成 | 100ep, seed=42, `local/franka-pick-up_wrist-d040-100ep` | 100/100 OK |
| 训练 | 2k steps, batch=4, 5 seeds (0-4) | 进行中 |
| Eval | 50 trials/seed, eval_seed=99, `--camera-layout up_wrist` | 待训练完成 |

脚本：`scripts/run_wrist_d040_e2e.sh`（训练 + eval 一键执行）

对标 baseline（up_side, R4）：

| seed | up_side success_rate |
|------|---------------------|
| 0 | 4/50 (8%) |
| 1 | 6/50 (12%) |
| 2 | 8/50 (16%) |
| 3 | 6/50 (12%) |
| 4 | 6/50 (12%) |
| **avg** | **6/50 (12%)** |

#### R9e 结果

5 seeds × 2k steps 训练 + 50 trials eval，总耗时 ~2.4h。

| seed | up_wrist (D040) | up_side (baseline) | 提升 |
|------|:---------------:|:------------------:|:----:|
| 0 | **5/50 (10%)** | 4/50 (8%) | +2% |
| 1 | **13/50 (26%)** | 6/50 (12%) | +14% |
| 2 | 1/50 (2%) | **8/50 (16%)** | -14% |
| 3 | **23/50 (46%)** | 6/50 (12%) | +34% |
| 4 | **21/50 (42%)** | 6/50 (12%) | +30% |
| **avg** | **12.6/50 (25.2%)** | **6/50 (12%)** | **+13.2%** |
| **median** | **13/50 (26%)** | **6/50 (12%)** | **+14%** |
| **max** | **23/50 (46%)** | **8/50 (16%)** | **+30%** |

#### R9e 分析

1. **up_wrist 平均 success rate 是 up_side 的 2.1×**（25.2% vs 12%）
2. **seed 敏感性仍然高**：最好 46%，最差 2%（方差比 up_side 更大）
3. **3/5 seeds 明显优于 up_side**（seed 1,3,4），1 seed 持平（seed 0），1 seed 明显更差（seed 2）
4. **方向修正是关键**：之前仰视的 M3 配置 0/10，修正后平均 25%
5. **100ep + 2k steps 仍然不够**：最佳 seed 也只有 46%，需要更多数据或更长训练

**Wrist 高 variance 的原因分析**（详见 `camera_layout_analysis.md`）：

- Side cam：信息弱但稳定（固定视角，帧间变化小）→ **容易学，low variance, low ceiling**
- Wrist cam：信息强但非平稳（随手移动，approach 阶段≈噪声）→ **难学，high variance, high ceiling**
- 2k steps < 1 epoch（3375 steps/epoch），policy 还没学完一遍数据，wrist feature 的复杂性尚未被充分学习

### R9f：延长训练（100ep + 4k steps）

> **假设**：高 variance 的根因是训练不足，延长到 4k steps（≈1.2 epoch）能降低 seed 敏感性并提升平均 success rate。

| 配置 | R9e（对照） | R9f |
|------|-----------|-----|
| 数据 | 100ep (不变) | 100ep (不变，复用 `local/franka-pick-up_wrist-d040-100ep`) |
| 训练 | 2k steps | **4k steps** |
| Seeds | 0-4 | 0-4 |
| Eval | 50 trials, seed=99 | 50 trials, seed=99 |
| save-every | 500 (默认) | **2000**（节省磁盘） |

预期结果：

| 场景 | 指标变化 | 解读 |
|------|---------|------|
| **假设成立** | avg ≥ 35%, std ↓, seed 2 回升 | 训练不足是主因，继续 200ep+4k |
| **部分成立** | avg ~30%, std 略降 | 训练有帮助但数据多样性也是瓶颈 |
| **假设不成立** | avg ≤ 25%, std 不变 | 可能过拟合 100ep，需 200ep |

**磁盘管理**：
- 系统盘 `/` 仅剩 66G (1.8T, 97%)，已清理 R9e 所有 checkpoints（保留 eval 结果）
- `--save-every 2000`：5 seeds × 3 ckpts × 1.2G ≈ 18G，安全
- `/dc2/david/` 有 1.6T 空闲，后续 10k+ 实验需重建容器挂载此盘
- 脚本：`scripts/run_wrist_d040_4k.sh`

#### R9f 结果

| Seed | up_wrist 4k | up_wrist 2k (R9e) | up_side 2k | 4k vs 2k |
|------|:-----------:|:-----------:|:----------:|:--------:|
| 0 | **22/50 (44%)** | 5/50 (10%) | 4/50 (8%) | +34% |
| 1 | **26/50 (52%)** | 13/50 (26%) | 6/50 (12%) | +26% |
| 2 | **13/50 (26%)** | 1/50 (2%) | 8/50 (16%) | +24% |
| 3 | **36/50 (72%)** | 23/50 (46%) | 6/50 (12%) | +26% |
| 4 | 10/50 (20%) | 21/50 (42%) | 6/50 (12%) | -22% |
| **avg** | **21.4/50 (42.8%)** | 12.6/50 (25.2%) | 6/50 (12%) | **+17.6%** |
| **median** | **22/50 (44%)** | 13/50 (26%) | 6/50 (12%) | **+18%** |
| **std** | 9.5 (19%) | 9.2 (18.4%) | 1.4 (2.8%) | — |

#### R9f 分析

1. **假设成立**：avg 42.8% >> 35% 阈值，4k 训练显著提升 wrist cam 性能
2. **4/5 seeds 大幅提升**：seed 0 (+34%), seed 1 (+26%), seed 2 (+24%), seed 3 (+26%)
3. **seed 4 异常回退** (42% → 20%)：可能过拟合或不幸 eval 随机性，需关注
4. **最佳 seed 达 72%** (seed 3)，远超 up_side 最佳 16%
5. **Variance 未显著下降** (std 19% vs 18.4%)：延长训练提升了均值但未降方差
6. **up_wrist 4k 是 up_side 2k 的 3.6×**（42.8% vs 12%）

**结论与 Next Steps**：
- 训练不足确实是主因（均值 +17.6%），但方差仍高 → 数据多样性也是瓶颈
- 建议：**200ep + 4k steps** 同时增加数据量和训练，预期 avg > 50%, std ↓
- Seed 4 回退需要诊断（检查 loss curve 是否异常）

---

## R10：Custom Scene（Kitchen）up+side 视角数据补齐

### R10a：Kitchen 相机视角图（基于 `02_gen_data_custom_scene.py`）

目标：将 camera views 从 flat scene 对齐到 workshop 主路径的 custom scene（`rustic_kitchen`），并确认 `up+side` 视角可用。

执行：

```bash
python scripts/02_gen_data_custom_scene.py \
  --scene rustic_kitchen --anchor floor_origin \
  --n-episodes 3 --repo-id local/kitchen-up-side-views-3ep \
  --save /output --seed 42 --no-videos

python scripts/05_visualize_camera_views.py \
  --dataset-id local/kitchen-up-side-views-3ep \
  --out-dir /output/camera_views/kitchen_up_side \
  --episodes 0 1 2 --n-frames 6
```

产物（已回传到仓库）：

- `images/camera_views_compare/kitchen_up_side/ep0_camera_views.png`
- `images/camera_views_compare/kitchen_up_side/ep1_camera_views.png`
- `images/camera_views_compare/kitchen_up_side/ep2_camera_views.png`

结论：

- `up` 相机持续覆盖抓取工作区，`side` 相机可观测抬升过程，适合 custom scene 下的 pick 任务记录与诊断。

### R10b：RDNA4 生成 Kitchen 100ep 数据

执行环境：`10.161.176.9`（RDNA4 R9700），容器 `rdna4_workshop`。

执行：

```bash
python scripts/02_gen_data_custom_scene.py \
  --scene rustic_kitchen --anchor floor_origin \
  --n-episodes 100 \
  --repo-id local/franka-pick-kitchen-up-side-100ep \
  --save /output --seed 42
```

结果（`custom_scene_gen/gen_summary.json`）：

| 指标 | 值 |
|------|-----|
| dataset id | `local/franka-pick-kitchen-up-side-100ep` |
| scene / anchor | `rustic_kitchen` / `floor_origin` |
| episodes | 100 |
| frames / ep | 135 |
| total frames | 13,500 |
| success rate | **100/100 = 100%** |
| action space | 9-DoF joint position |

关键日志：

- `[gen] ep 100/100 [OK] ...`
- `[gen] dataset saved: /root/.cache/huggingface/lerobot/local/franka-pick-kitchen-up-side-100ep`
- `[gen] success_rate: 100/100 = 100%`

结论：

- custom scene（kitchen）下 `up+side` 的数据采集链路在 RDNA4 上稳定可用，已完成 100ep 数据集构建，可直接用于后续 kitchen 侧训练/评估。

---

## 训练 Latency 优化

### SmolVLA Attention 调用链分析

SmolVLA 的 attention 并不是 LeRobot 自己实现的，而是层层委托到 PyTorch/Transformers 底层：

```
02_train_vla.py
  └─ SmolVLAPolicy.forward(batch)                         # LeRobot policy
       └─ self.vlm_with_expert.forward(attention_mask=...) # LeRobot SmolVLMWithExpertModel
            ├─ self.vlm (SmolVLMForConditionalGeneration)  # HuggingFace Transformers
            │    └─ SmolVLMDecoderLayer.self_attn(Q, K, V) # Transformer Layer
            │         └─ F.scaled_dot_product_attention()   # PyTorch SDPA 统一 API
            │              └─ PyTorch runtime 自动 dispatch:
            │                   ├─ flash_sdp    → AOTriton (AMD) / FlashAttention (NVIDIA)
            │                   ├─ mem_efficient_sdp → xFormers-style fused kernel
            │                   └─ math_sdp     → 纯 matmul（最慢，当前 fallback）
            └─ self.lm_expert (action expert, 同架构 transformer)
                 └─ 同上调用路径
```

**关键点**：
- SmolVLM 声明 `_supports_sdpa = True`, `_supports_flash_attn = True`
- LeRobot / SmolVLA 代码**不关心具体 kernel**——只调用 `F.scaled_dot_product_attention`
- backend 选择发生在 **PyTorch runtime 层**，由 `torch.backends.cuda.enable_flash_sdp()` 全局开关控制
- HuggingFace Transformers 另支持 `attn_implementation="flash_attention_2"` 参数，
  可绕过 PyTorch SDPA，直接调用 Tri Dao `flash-attn` 包的 `flash_attn_func`

#### RDNA4 上可用的 Attention Backend 对比

数据来源：[flash-attention PR #2400](https://github.com/Dao-AILab/flash-attention/pull/2400) benchmark（gfx1200, SeqLen=4096）

| Backend | 调用路径 | 安装方式 | RDNA4 gfx12 支持 | 延迟 | TFLOPS | vs SDPA math |
|---------|---------|---------|:--:|:--:|:--:|:--:|
| **SDPA math** | `F.sdpa` → math kernel | PyTorch 内置 | ✅ | 5.09 ms | 27 | 1.0x (baseline) |
| **SDPA flash (AOTriton)** | `F.sdpa` → flash_sdp | PyTorch 内置 (AOTriton 0.10b+) | ✅ official | 4.18 ms | 33 | **1.2x** |
| **FA2 CK (Tri Dao)** | `flash_attn_func` | `pip install flash-attn` 源码编译 | ✅ (PR #2400, 2026-03-26) | **2.62 ms** | **52** | **1.9x** |

> **注意**：FA2 CK 在 gfx12 上 forward + backward 均可用（仅 deterministic backward 不支持），
> 适用于训练。gfx11 (RDNA3) backward 被禁用，仅限推理。
> 参考：[rocm-lib-compat SKILL](https://github.com/ZJLi2013/rocm3d/blob/main/.cursor/skills/rocm-lib-compat/SKILL.md)

#### 三层递进策略

| 层级 | 方案 | 改动量 | 预期加速 |
|:--:|------|:--:|:--:|
| L0 | 重新启用 SDPA auto（`--attn-backend auto`）→ AOTriton flash | 1 行 CLI flag | ~1.2x attn |
| L1 | 安装 `flash-attn` CK + `attn_implementation="flash_attention_2"` | 安装 + 代码改造 | **~1.9x attn** |
| L2 | 安装 `aiter` CK（ROCm 7.2 auto-select CK path） | `pip install aiter` | ~1.9x attn (同 CK) |

当前实验先走 **L0**（零安装成本验证 AOTriton），若瓶颈仍在 attention 再升级到 **L1/L2**。

### 现状分析

当前训练配置（`02_train_vla.py`）为手写训练循环，**未使用 LeRobot 官方训练管线中的任何加速特性**：

| 维度 | 当前状态 | 优化空间 |
|------|---------|---------|
| SDPA backend | flash + mem_efficient **已禁用**（AOTriton hang workaround for MI300+ROCm6.4） | L0: AOTriton ~1.2x / L1: FA2 CK ~1.9x |
| 精度 | **纯 FP32** | AMP BF16 autocast，加速所有 op 含 attention |
| LR scheduler | 常数 LR | CosineDecayWithWarmup（官方默认），更快收敛 |
| Batch size | **4**（VRAM 仅用 2.38/16 GB = 15%） | 可提升至 16-32 |
| torch.compile | 未启用 | SmolVLA 原生支持 `compile_model=True` |

> **约束**：Workshop 现场只提供单卡，不走 DDP。优化全部在单 GPU 上完成。

瓶颈已是 GPU-bound（data loading 仅 0.03s/step），SmolVLA 是 transformer-based VLA，
**attention 是计算大头**，math kernel fallback（最慢路径）是全局性能 bottleneck。

### 优化方案：按收益排序

在 `02_train_vla.py` 中新增 CLI flags，逐层叠加验证：

| 优先级 | Flag | 功能 | 预期收益 |
|:--:|------|------|---------|
| P0 | `--amp` | BF16 mixed precision（`torch.autocast`） | 全局 1.5-2x，最安全 |
| P1 | `--attn-backend auto` | 重新启用 flash/efficient SDPA | attention 2-4x（叠加 AMP 效果更大） |
| P2 | `--lr-scheduler cosine` | CosineDecay with warmup | 更快收敛 → 更少步数 |
| P3 | `--batch-size 16` | 增大 batch | 相同数据覆盖量步数 /4 |
| P4 | `--compile` | `torch.compile(policy)` | 额外 10-30% kernel fusion |

> **P0 + P1 是互补的**：AMP 让每个 op 的 dtype 更轻，fast SDPA 让最重的 attention op 算法更优，
> 两者叠加 = bf16 flash attention，预期单步最大化加速。

### Baseline（优化前）

```bash
python scripts/02_train_vla.py --dataset-id local/rdna4-video-100ep \
  --n-steps 2000 --batch-size 4 --num-workers 4 \
  --save-dir /output/outputs/rdna4_smolvla_video
```

| 指标 | 值 |
|------|:--:|
| per-step | ~0.73 s/step |
| 训练时间 (2k steps) | ~24.5 min |
| Peak VRAM | 2.38 GB / 16 GB |
| GPU 利用率 | 96% |
| 精度 | FP32 |
| SDPA backend | math (flash + efficient 已禁用) |

### Exp O1: AMP BF16

#### Phase 0 确认
- 观测完备性: 与 R3 baseline 相同（image + joint state），仅改变训练精度
- 随机化变量: 无新增，cube 位置 seed=42 固定

#### 假设
BF16 mixed-precision (`torch.autocast`) 在 RDNA4 (gfx1201) + ROCm 7.2 + PyTorch 2.9.1 下
可正常运行，per-step 从 ~0.73s 降至 ~0.4-0.5s（~1.5-2x 加速），loss 收敛行为与 FP32 baseline 一致。

#### 实验方案
- 验证环境: `10.161.176.9`（RDNA4 R9700），容器 `rdna4_workshop`
- 后续需在 CDNA3 (MI308X) 上单独验证
- 脚本: `scripts/02_train_vla.py`
- 数据集: `local/rdna4-video-100ep`（100ep, flat scene, seed=42）
- 对照组 (R3 baseline): `--n-steps 2000 --batch-size 4 --num-workers 4` → 0.73 s/step, 24.5 min
- 变量: 仅新增 `--amp`

```bash
# smoke test (10 steps)
python scripts/02_train_vla.py --dataset-id local/rdna4-video-100ep \
  --n-steps 10 --batch-size 4 --num-workers 4 --amp \
  --save-dir /output/outputs/rdna4_opt_amp_smoke

# 全量
python scripts/02_train_vla.py --dataset-id local/rdna4-video-100ep \
  --n-steps 2000 --batch-size 4 --num-workers 4 --amp \
  --save-dir /output/outputs/rdna4_opt_amp
```

#### 预期
- 假设成立: per-step ~0.4-0.5s, 训练 ~13-17 min, loss 趋势同 baseline, Peak VRAM 下降
- 假设不成立: RuntimeError (dtype mismatch) 或 loss 发散 → fallback FP32

#### 结果
（待实验）

#### 分析
（待实验）

---

### Exp O2: AMP + SDPA auto（重新启用 AOTriton flash）

#### Phase 0 确认
- 同 O1

#### 假设
RDNA4 + ROCm 7.2 的 AOTriton (0.10b+) 已官方支持 gfx1201，重新启用 flash/efficient SDPA
不会 hang。叠加 AMP BF16，attention 走 bf16 flash kernel，per-step 进一步降低。

#### 实验方案
- 验证环境: 同 O1
- 对照组: O1（AMP + math SDPA）
- 变量: `--attn-backend auto`（重新启用 flash + efficient SDPA）

```bash
# smoke test (10 steps, 验证不 hang)
python scripts/02_train_vla.py --dataset-id local/rdna4-video-100ep \
  --n-steps 10 --batch-size 4 --num-workers 4 --amp --attn-backend auto \
  --save-dir /output/outputs/rdna4_opt_amp_sdpa_smoke

# 全量
python scripts/02_train_vla.py --dataset-id local/rdna4-video-100ep \
  --n-steps 2000 --batch-size 4 --num-workers 4 --amp --attn-backend auto \
  --save-dir /output/outputs/rdna4_opt_amp_sdpa
```

#### 预期
- 假设成立: per-step < O1（AOTriton flash ~1.2x vs math），无 hang/crash
- 假设不成立: 进程 hang（AOTriton 仍有问题）→ kill, fallback `--attn-backend math`，记录 failure
- 风险: AOTriton 0.9.2b known issue "FA kernel for 9070XT may segfault on certain conditions"

#### 结果
（待实验）

#### 分析
（待实验）

---

### Exp O3/O4（待 O1/O2 结论后设计）

O3: AMP + SDPA + cosine LR + batch 16，O4: + torch.compile。
具体参数待 O1/O2 结果确定最优 attention backend 后再定。

