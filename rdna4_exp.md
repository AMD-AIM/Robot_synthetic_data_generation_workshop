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

### 实现

- `01_gen_data.py`：新增 `--camera-layout up_wrist`，wrist cam attach 到 `hand` link，link-local 外参通过 `--wrist-cam-{pos,lookat,fov}` 控制
- `05_visualize_camera_views.py`：从 dataset 导出 `ep{i}_camera_views.png`，用于视觉验证
- Genesis 机制：`camera.attach(rigid_link=..., offset_T=...)`，每步渲染前自动 `move_to_attach()`（[Issue #1016](https://github.com/Genesis-Embodied-AI/Genesis/issues/1016), [PR #611](https://github.com/Genesis-Embodied-AI/Genesis/pull/611)）

### 参数搜索历程（R9a/R9b）

Franka hand 近距离 mount 存在严重 self-occlusion，经 3 轮搜索才找到可见候选：

| 轮次 | 候选 | 策略 | 结果 |
|:---:|------|------|------|
| A | W0, C1-C3 | 近距离 pos 小扰动 | **全部自遮挡**，cube 不可见 |
| B | D1-D6, U1-U3 | 固定 pos，扫 lookat/up 方向 | **仍自遮挡** |
| C | G1-G4 | **大位移几何重置**（远离 hand） | cube+gripper 均可见 |
| D | M1-M3 | 围绕 G1 微调 | **M3 最稳**，进入 E2E |

图像存档（已清理，仅保留关键节点）：
- `up_side/ep{0,1,2}` — baseline
- `up_wrist/ep0` — 原始失败参考（self-occlusion）
- `wrist_dircal/g1/ep0` — 几何重置突破点
- `wrist_dircal/m3/ep0` — 当前最佳候选

### M3 参数与图像评估

```
M3: pos=(0.28, 0.00, 0.22)  lookat=(0.04, 0.00, -0.06)  up=(0,0,1)  fov=95
```

对比 `up_side/ep{0,1,2}` 与 `wrist_dircal/m3/ep0`，M3 有 3 个硬伤：

| 问题 | 说明 |
|------|------|
| **approach 阶段 cube 不可见** | 手臂远离 cube 时，wrist cam 视野内 cube 极小或出视野 |
| **fov=95 过大** | 广角畸变，物体缩小，丧失 wrist cam 近距离细节优势 |
| **offset 过远 (36cm)** | 实质是"前臂俯瞰相机"而非 wrist cam |

**结论：M3 可见性成立但信息密度不如 side。需先优化参数至 ≥ side 水平，再做 E2E 对比。**

### R9c：几何重构 + 自动化搜索（进行中）

> **核心洞察**：M3 的根本问题不是 FOV 或 offset 微调，而是**相机 optical axis 没对准 grasp interaction zone**。
> M3 本质上仍是"远距离外部相机"（offset 36cm，向下看桌面），不是"接触感知传感器"。

#### 设计原则修正

| 旧思路（R9a/R9b） | 新思路（R9c） |
|---|---|
| wrist cam 应全程看到 cube | wrist cam **只服务最后 5-10cm 的抓取闭环** |
| lookat 朝向桌面/cube | lookat 对准 **gripper 闭合区域 (0,0,0)** |
| offset 15-36cm | offset **5-12cm** |
| 全帧均匀评分 | **grasp 阶段权重 50%**，approach 阶段完全忽略 |
| approach 阶段 cube 不可见 = 失败 | approach 由 overhead cam 负责，**完全可以接受** |

正确分工：

| 阶段 | 距离 | 主要视觉 |
|------|------|---------|
| approach (far) | 10-30cm | overhead cam |
| grasp (near) | 0-10cm | **wrist cam** |

三个必须满足的几何条件：
1. camera → gripper center 距离 ≈ 5-10cm
2. optical axis ≈ 对准夹爪闭合区域
3. grasp 时视野中 fingers 占 30-50%、cube 占 10-30%

#### 搜索空间

基于上述原则重新定义，**不再围绕 M3 微调**：

| 分组 | mount | pos_x | pos_z | lookat_z | fov | 组合数 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| **主力：link7** | panda_link7 | [0.05, 0.08, 0.12] | [0.03, 0.06, 0.10] | [0.00, -0.03] | [55, 65, 80] | 54 |
| **对照：hand** | hand | [0.06, 0.10] | [0.04, 0.08] | [0.00] | [60, 75] | 8 |
| **合计** | | | | | | **62** |

所有候选 lookat 统一对准 gripper 中心附近 `(0, 0, 0~-0.03)`，不再向下看桌面。

#### 评分机制

轨迹分 5 个 phase，**approach 阶段（40 帧）完全跳过**，不渲染不打分：

| Phase | 帧范围 | 权重 | 说明 |
|-------|--------|------|------|
| approach | 0-39 | **0**（跳过） | overhead cam 负责 |
| pre_grasp | 40-69 | 0.15 | 下降接近 |
| **grasp** | **70-89** | **0.50** | 关键：接触感知 |
| lift | 90-119 | 0.25 | 抬升确认 |
| hold | 120-134 | 0.10 | 稳定持有 |

composite score = weighted_vis × 0.6 + min(grasp_area / 2000, 1) × 0.4

#### 工具

`scripts/07_wrist_cam_sweep.py`
- 无 LeRobot 依赖，直接 Genesis 跑轨迹 + 渲染 + 打分
- approach 阶段跳过渲染（加速 ~30%）
- 输出：CSV 排名 + top-K 可视化 PNG + summary JSON

#### 执行

```bash
python scripts/07_wrist_cam_sweep.py \
  --out-dir /output/wrist_sweep \
  --episodes-per-candidate 3 \
  --top-k 5 \
  --no-bbox-detection
```

62 候选 × 3 ep，预计 ~22 min。

#### 通过标准

top-1 候选必须满足：
1. `grasp_vis_rate` ≥ 90%（grasp 阶段 cube 几乎全程可见）
2. `grasp_avg_area` ≥ 500px（cube 在视野中足够大）
3. 人工确认 top-5 PNG：grasp 时 cube 在双指之间、能判断对齐
4. 无严重自遮挡

通过后进入 E2E：100ep × 2k × 5 seeds × 50 trials，与 up_side R4 对标。

#### 预期结果

- **假设成立**：panda_link7 近距离组（5-12cm）中存在 grasp_vis ≥ 90% 的候选，hand 组因 finger 遮挡普遍较差
- **假设不成立**：所有 62 候选 grasp_vis < 70% → 说明 link-local frame 几何理解有误，需回头检查 panda_link7 的坐标系方向
- **中间情况**：top-1 grasp_vis 70-90%，area 偏小 → 在 top-1 附近做细粒度二次搜索

#### 结果

（待实验）

#### 分析与 Next Step

（待实验）

