# RDNA4 (RX 9070) Workshop 复现实验

> 目标：在 RDNA4 架构上复现 Robot Synthetic Data Generation Workshop 全流程，
> 重点关注 **渲染性能** 和 **成功率**，与 CDNA3 (MI300) 结果做对比。

---

## 实验总览

| Exp | 假设 | 状态 | 关键结果 | 结论 |
|-----|------|------|---------|------|
| R1-flat | RDNA4 渲染加速 data gen (flat) | ✅ done | 10/10=100%, 8.3s/ep | **3.4× faster** vs CDNA3 28s/ep |
| R1-kitchen | RDNA4 渲染加速 data gen (kitchen) | ✅ done | 10/10=100%, 200s/10ep | Kitchen GLB 场景也正常 |
| R3 | Video + nw=4 training baseline | ✅ done | 2k step 24.5 min, GPU util 96 % | 当前默认训练配置（含 torchcodec ROCm 修复）|
| R4 | 100ep seed variance | ✅ done | mean=30%, std=18.5% | 100ep ceiling 不足，variance 极大 |
| **R5** | **200ep episode scaling** | ✅ done | **mean=52.4%, std=21.2%** | **mean +22pp，variance 仍高** |
| **O1** | **AMP BF16 降低 per-step latency** | ✅ done | **0.31 s/step, 644s** | **2.3× faster** vs baseline 0.73s |
| **O2** | **AMP + SDPA auto (AOTriton flash)** | ✅ done | **0.09 s/step, 213s** | **8.1× faster** vs baseline, 3.0× vs O1 |
| **K2** | Kitchen + wrist 100ep 数采 | ✅ done | 100/100, 23 min | HF: [`lidavidsh/franka-pick-kitchen-up-wrist-100ep-genesis`](https://huggingface.co/datasets/lidavidsh/franka-pick-kitchen-up-wrist-100ep-genesis) |
| **K3-A** | CDNA3 train + CDNA3 CPU eval (kitchen+wrist) | ✅ done | pooled **25.0 %** (5 eval_seed × 20) | ❌ 低于通过线，由 render gap 导致 |
| **K3-B** | RDNA4 train + RDNA4 GPU eval（render-gap 消融） | ✅ done | pooled **48.0 %**, 3/5 seeds ≥50 % | ✅ 主假设成立，kitchen D040 可用 |
| **K3-C** | CDNA3 train ckpt → RDNA4 GPU eval（交叉消融） | ✅ done | pooled **45.0 %**（≈ K3-B） | ✅ 23 pt 差距 ≈ 纯 render gap，训练 stack 等效 |

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

- 后续实验数据、checkpoint、视频输出 **优先落盘到 `/dc1` 或 `/dc2`**，避免继续占用系统盘 `/`。
- 建议统一输出根目录：`/dc2/rdna4_output`（或 `/dc1/rdna4_output`），并在 Docker 中挂载为 `/output`。

---

## Exp-R1: Data Generation 渲染性能

workshop 脚本（`01_gen_data.py` / `02_gen_data_custom_scene.py` / `04_eval_*.py`）不需要任何额外配置，Genesis 默认走 EGL 自动选择可用后端：

| 架构 | 渲染器 | data gen 速度 |
|---|---|:--:|
| **RDNA4 (R9700)** | radeonsi **GPU 硬件渲染** | ~6–8 s/ep |
| CDNA3 (MI300) | llvmpipe **CPU 软件渲染**（无 graphics driver，自动回落） | ~28 s/ep |

> eval 阶段若在 CDNA3 上运行，`04_eval_custom_scene.py --render-cpu` 会显式用 CPU 后端。注意 CPU 渲染引入约 **20 pt success-rate bias**（见 K3-C），只适合 debug，不能用于 benchmark。

**Flat Scene (100 episodes, PNG mode)**

| 指标 | RDNA4 (R9700) | CDNA3 (MI300) |
|------|:--:|:--:|
| 成功率 | **100/100 = 100%** | 100/100 = 100% |
| 总耗时 (video) | **638s** (~10.6 min) | ~2800s (~47 min) |
| 总耗时 (PNG) | **771s** (~12.9 min) | ~3080s (~51 min) |
| 每 ep (video) | **~6.4s** | ~28s |
| 每 ep (PNG) | **~7.7s** | ~30.8s |
| 加速比 | **3.4-4.4×** | baseline |

**分析**：RDNA4 data gen 加速 **3.4–4.4×**（28 s/ep → 6.4–8.3 s/ep），主要来自 EGL radeonsi GPU 硬件渲染 vs CDNA3 llvmpipe CPU 软件渲染。未达理论 5–10× 因物理仿真、IK、Python loop、video/PNG 编码的 CPU 开销占比不小。Kitchen 场景 mesh 复杂度高（126.9 MB GLB），渲染开销更大。

---

## Exp-R3: Video 格式 + num_workers=4 训练加速

> **torchcodec ROCm 依赖**：pip wheel 链接 CUDA `.so`，ROCm 上 import 失败。workshop 用户按 [`README.md → torchcodec on ROCm`](../README.md#4-torchcodec-on-rocm-video-mode) 从源码构建 0.10.0 CPU-only 版即可（不能走 `pip install` / `requirements.txt`，必须源码构建，建议固化到 Dockerfile）。

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

> **⚠️ 注意**：两次训练 cube 位置完全一致（都是 seed=42），但 eval 差异大（10 % → 80 %）是 **训练未固定 seed 的 run-to-run variance**，与 Video/nw=4 无关。4k 对照实验 (1.2 epoch) 在同一 4k-run 的 1k/2k/3k/4k checkpoint 上 eval 完全平坦 (~30 %)，而另一次独立 2k-run 是 72 %——进一步佐证高 variance 的主导地位。**结论**：低训练量 (< 1 epoch) 下 eval 方差极大，n=10 trial 结果不可信，需系统性 seed variance 实验（见 R4/R5）。

---

## R4 + R5：Seed Variance 与 Episode Scaling

> **核心问题**：seed 间 run-to-run variance 多大？100 → 200 ep 能否改善 mean 和 worst-case？
>
> 两组实验均用 `train_with_seed.py` 固定 torch/random/np seed，5 seeds × 50-trial eval (seed=99)，同为 ≈ 0.6 epoch。训练时间采用当时 FP32 baseline 配置（已由 O2 替换为 AMP+SDPA flash，~8×），此处不再列出；**success rate 分布与精度无关**（见 O1/O2 loss 收敛对照）。

| Seed | 100 ep × 2k step | 200 ep × 4k step |
|:---:|:---:|:---:|
| 0 | 24 % | 70 % |
| 1 | **58 %** | 24 % |
| 2 | 8 % | **74 %** |
| 3 | 24 % | 56 % |
| 4 | 36 % | 38 % |
| **mean ± std** | **30.0 % ± 18.5 %** | **52.4 % ± 21.2 %** |
| range | 8 – 58 % | 24 – 74 % |
| CV | 62 % | **40 %** |

**结论**：

1. **seed variance 极高**：固定 seed 后 5 次训练 CV 仍 40–60 %，seed 影响远大于超参调优。**≥ 50 trial eval 是可信度必须条件**。
2. **100 ep data-limited → 200 ep optimization-limited**：mean +22.4 pp、worst 8 → 24 %、best 58 → 74 %、CV −22 pp。好策略已存在（best 74 %），能否 consistently 进入好 basin 取决于优化；继续加数据（300/500 ep）收益递减。

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

## R7（规划，未执行）：降低 Optimization Variance

200ep 数据已支撑好策略（best seed 74 %），但 SGD 轨迹高度 seed 依赖，24–74 % spread。计划候选：降 LR、增大 batch、EMA、warmup + cosine、grad clip。**实际策略优先级**：R6 checkpoint selection（零成本）→ best-of-N（工程常见）→ EMA + 降 LR → batch 4→8。现实预期：worst 24 % → 40–50 %，std 21 % → 10–15 %。未执行，转入 K 系列 kitchen 场景验证。

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

**视角**：标准 eye-in-hand 俯视，相机从两指中间上方俯视抓取区，画面中两指对称出现在左右两侧、cube 居中；approach 阶段 wrist 看不到 cube 是正常的（由 overhead cam 覆盖）。设计原则：overhead 负责全局定位（world frame），wrist 负责接触控制（hand frame），reference frame 正交互补。

### 搜索历程摘要（R9a–d）

R9a/b 近距 mount 出现严重 self-occlusion；R9b-M3 用 pz>0 得到 cube 可见但**方向反转**（hand-local z 轴朝下，pz>0 = 相机在桌面仰视），E2E 0/10；R9c 自动 sweep (192 候选) 排除 `link7` frame；R9d 方向修正（pz<0 + lookat_z>0 + up=-z）后 72/81 viable。**一条关键经验**：Franka MJCF `hand` link-local z 轴朝下，pz<0 才是正确 eye-in-hand 俯视。

**R9d sweep** (81 候选, 239 s)：pz=-0.08 最佳平衡。最终选 **D040** = (0.05, 0, -0.08) / lookat (0,0,0.10) / up (0,0,-1) / fov 65，g_area=3145。

### D040 选型依据（vs sweep g_area top1 D000）

D000 (pos=(-0.02,0,-0.05), fov=55) 几何排名第一但 E2E 未验证；D040 (pos=(0.05,0,-0.08), fov=65) 已用 R9e 25.2 % 和 R9f 42.8 % 实证。D040 优势：

1. **motion sensitivity 低**（fov 宽 + 距离远，手腕抖动画面变化小）
2. **fov 匹配社区主流 VLA (60–90°)**，RT-1/RT-2 90°、DROID 70°
3. **与 overhead cam 正交互补**（一个 hand-local top-down，一个 world top-down）

参考视图：

![D040 wrist view, R9f flat 100ep dataset ep0](images/camera_views_compare/r9f_flat_d040_ep0.png)

其它图像存档：`up_side/ep{0,1,2}`（side baseline）、`up_wrist/ep0`（原始 self-occlusion）、`wrist_dircal/m3/ep0`（M3 仰视错误）、`r9d_top1_D000.png`（D000 对照）。

### R9e：E2E 2k step × 5 seed

D040 参数，100ep 数据 `local/franka-pick-up_wrist-d040-100ep`，2k steps × 5 seeds × 50 trials eval (seed=99)，`--camera-layout up_wrist`。

| seed | up_wrist D040 | up_side baseline |
|:---:|:---:|:---:|
| 0 | 5/50 (10 %) | 4/50 (8 %) |
| 1 | 13/50 (26 %) | 6/50 (12 %) |
| 2 | 1/50 (2 %) | 8/50 (16 %) |
| 3 | 23/50 (46 %) | 6/50 (12 %) |
| 4 | 21/50 (42 %) | 6/50 (12 %) |
| **avg** | **25.2 %** | **12.0 %** |
| max | 46 % | 16 % |

**结论**：up_wrist 均值是 up_side 的 2.1×，但 seed variance 仍高（2 %–46 %）。100ep + 2k step 不够，2k < 1 epoch (3375 steps/epoch)，wrist 非平稳 feature 未充分学习。

### R9f：100ep × 4k step（延长训练）

同 dataset 和 seeds 0–4，仅 2k → 4k step (≈1.2 epoch)。

| Seed | up_wrist 4k | vs R9e 2k |
|:---:|:---:|:---:|
| 0 | 22/50 (44 %) | +34 |
| 1 | 26/50 (52 %) | +26 |
| 2 | 13/50 (26 %) | +24 |
| 3 | **36/50 (72 %)** | +26 |
| 4 | 10/50 (20 %) | −22 |
| **avg ± std** | **42.8 % ± 19 %** | **+17.6 pp** |

**结论**：训练不足是主因（均值 +17.6 pp），但 std 未降（19 % vs 18.4 %），数据多样性也是瓶颈。4/5 seed 大幅提升，seed 4 异常回退（可能过拟合或 eval 随机性）。最佳 seed 72 %，远超 up_side 最佳 16 %。建议 next：200 ep + 4k 同增数据和训练。

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

## Exp-O 系列：训练 Latency 优化

### O.0 SmolVLA attention 调用链

```
SmolVLAPolicy.forward
  └─ SmolVLMWithExpertModel.forward
       ├─ vlm (SmolVLMForConditionalGeneration)
       │    └─ SmolVLMDecoderLayer.self_attn → F.scaled_dot_product_attention
       │         └─ PyTorch runtime dispatch:
       │              ├─ flash_sdp         → AOTriton (AMD) / FlashAttention (NVIDIA)
       │              ├─ mem_efficient_sdp → xFormers-style fused kernel
       │              └─ math_sdp          → 纯 matmul（最慢 fallback）
       └─ lm_expert（action expert, 同结构, 同路径）
```

LeRobot / SmolVLA **不关心具体 kernel**，只调 `F.scaled_dot_product_attention`；backend 选择由 PyTorch runtime 承担（`--attn-backend auto` 等价 `torch.backends.cuda.enable_flash_sdp(True)`）。HF Transformers 另支持 `attn_implementation="flash_attention_2"` 走 Tri Dao CK backend，但非必须。

### O.1 / O.2：RDNA4 优化（AMP + AOTriton flash SDPA）

**Backend 选型**（参考 [flash-attn PR #2400](https://github.com/Dao-AILab/flash-attention/pull/2400)，gfx1200 / SeqLen=4096）：SDPA math 5.09 ms、**AOTriton flash 4.18 ms (1.2×)**、FA2 CK 2.62 ms (1.9×)。选 **AOTriton auto**：零安装，PyTorch runtime 自动 dispatch；FA2 CK 仅 per-step 再需压缩时考虑。

**结果**（`local/rdna4-video-100ep`, 100 ep flat, 2k step × bs 4 × nw 4）：

| Exp | 配置 | per-step | 2k 总时间 | Peak VRAM | Final loss | 加速 |
|---|---|:---:|:---:|:---:|:---:|:---:|
| BL | FP32 + math SDPA | 0.73 s | 24.5 min | 2.38 GB | 0.053 | 1.0× |
| **O1** | AMP BF16 + math SDPA | 0.31 s | 10.7 min | 2.27 GB | 0.063 | **2.3×** |
| **O2** | AMP BF16 + AOTriton flash | **0.09 s** | **3.6 min** | 2.27 GB | 0.060 | **8.1×** |

```bash
python scripts/02_train_vla.py --dataset-id local/rdna4-video-100ep \
  --n-steps 2000 --batch-size 4 --num-workers 4 \
  --amp --attn-backend auto --save-dir /output/outputs/rdna4_opt_amp_sdpa
```

**结论**：AMP BF16 单独 2.3×（loss 收敛与 BL 一致，精度无损），叠加 AOTriton flash 到 **8.1×**。VRAM 基本不变（SmolVLA seqlen 短，flash 的 O(n) 内存优势不显）。workshop **默认开启 `--amp --attn-backend auto`**。

### O.3：CDNA3 对照（C-O1 / C-O2）

动机：旧 workaround 因 MI300+ROCm 6.4 AOTriton hang 禁用 SDPA，验证 AMP + AOTriton 在 CDNA3 (MI300) + ROCm 6.4.3 是否同样正常。环境 `rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0`，dataset `lidavidsh/franka-pick-100ep-genesis`，2k step × bs 4 × nw 4。

| Exp | 配置 | per-step | 2k 总时间 | Peak VRAM | Final loss | 加速 |
|---|---|:---:|:---:|:---:|:---:|:---:|
| C-BL | FP32 + math SDPA | 0.21 s | 457 s | 2294 MB | 0.012 | 1.0× |
| **C-O1** | AMP BF16 + math SDPA | 0.18 s | 381 s | 2244 MB | 0.028 | **1.20×** |
| **C-O2** | AMP BF16 + AOTriton | **0.14 s** | **330 s** | 2238 MB | 0.012 | **1.39×** |

C-O2 日志 `Using AOTriton backend for Efficient Attention forward...`，无 hang / crash → **历史 SDPA disable workaround 可安全移除**。

### O.4：RDNA4 vs CDNA3 横纵对比

**纵向**（同平台各优化相对 baseline 加速）：

| 平台 | BL → O1（AMP） | O1 → O2（+ AOTriton） | BL → O2（总） |
|---|:---:|:---:|:---:|
| RDNA4 | 2.3× | 3.4× | **8.1×** |
| CDNA3 | 1.17× | 1.16× | **1.39×** |

RDNA4 收益远大于 CDNA3——**因 RDNA4 baseline 被 math SDPA 拖得慢**，CDNA3 FP32 math 已经接近 kernel-launch-bound，继续压缩空间有限。

**横向**（per-step 绝对值）：

| config | RDNA4 | CDNA3 | CDNA3 / RDNA4 |
|---|:---:|:---:|:---:|
| BL (FP32 + math) | 0.73 s | 0.21 s | 0.29× |
| O2 (AMP + AOTriton) | 0.09 s | 0.14 s | **1.56×** |

> ⚠️ **横向绝对值不能代表算力强弱**：
> 1. 本实验使用的 CDNA3 SKU 并非满配 MI300X（CU / HBM 均有裁剪），满配 MI300X 算力显著更高。
> 2. SmolVLA 450 M + bs 4 + 短序列严重 under-utilize CDNA3（VRAM 用 1.2 %），per-step 被 kernel launch 主导而非 compute；该条件下 RDNA4 反而占优。
> 3. stack 版本不同：RDNA4 ROCm 7.2 + PyTorch 2.9.1，CDNA3 ROCm 6.4.3 + PyTorch 2.6.0，AOTriton 成熟度不一致。
>
> **有效结论仅限**："AMP + AOTriton SDPA 在两边都生效，原 workaround 可移除"。batch ≥ 32 / 大模型 / 长序列下横向比较需重新评估。

---

## Kitchen Scene — 眼在手上（wrist+up）数据采集与验证

> 接续 R9d/R9e/R9f：flat 场景下 `up+wrist` D040 配置（pos=(0.05,0,-0.08), lookat=(0,0,0.10), up=(0,0,-1), fov=65）已验证可行，R9f 100ep × 4k step 5-seed eval 得 mean 42.8%（range 20-72%）。本章节把 `wrist` 视角迁移到 custom scene（kitchen），服务 workshop 在 CDNA3 上的多 seed 验证。

### Exp-K2：Kitchen + wrist 100ep 数据采集

```bash
scripts/02_gen_data_custom_scene.py --scene rustic_kitchen --anchor floor_origin \
  --n-episodes 100 --camera-layout up_wrist \
  --repo-id local/franka-pick-kitchen-up-wrist-100ep-genesis --seed 42
```

D040 wrist pose（与 R9f flat 完全一致：pos (0.05,0,-0.08) / lookat (0,0,0.10) / up (0,0,-1) / fov 65），SVT-AV1 video 存储。

**Results**（RDNA4 R9700 / `rdna4_workshop`）：

| 指标 | 值 |
|---|---|
| 成功率 | **100 / 100 = 100 %** |
| 总 frames | 13,500（100 ep × 135 f/ep，30 FPS） |
| 运行耗时 | 23 min 14 s（含 Genesis scene compile + video encode），~12 s/ep 稳态 |
| 数据集大小 | 198 MB（`observation.images.{up,side}` 各 1 mp4） |

**结论**：D040 pose 在 kitchen 场景 depth / occlusion / FOV 均可接受，与 flat D040 (100/100) 同行为 → 数据直接作为 K3 训练输入。数据集地址 [`lidavidsh/franka-pick-kitchen-up-wrist-100ep-genesis`](https://huggingface.co/datasets/lidavidsh/franka-pick-kitchen-up-wrist-100ep-genesis)。

> ⚠️ 与老 repo `lidavidsh/franka-pick-kitchen-100ep-genesis` (up+side baseline) 的 tensor key 完全相同但 `.side` 语义不同（世界侧视 vs 腕视）——**严禁** LeRobotDataset concat / mixed training，混合须先改 pipeline 为 3-cam + 重命名 wrist 为独立 key。

### Exp-K3-A：CDNA3 train + CDNA3 CPU-eval（1 train × 5 eval-seed × 20 trials）

**Config**：CDNA3 `smc300x-clt-r4c11-02` / MI300 × 5（ROCm 6.4.3 + PyTorch 2.6），`cdna3_workshop` 容器。训练 seed=0、4000 step、batch 4、num_workers 4，dataset HF `lidavidsh/franka-pick-kitchen-up-wrist-100ep-genesis`。Eval 5 路并行 pin GPU 0-4，`eval_seeds=99-103 × 20 trials × max_steps=150 × --camera-layout up_wrist × --render-cpu`（CDNA3 节点无 graphics driver，Genesis 走 CPU llvmpipe，policy 仍 CUDA）。

> 脚本侧新增 `02_train_vla.py --seed` 和 `04_eval_custom_scene.py --render-cpu` 两个补丁。

**Results**：

| 指标 | 值 |
|---|---|
| train loss start → end | 0.671 → 0.0162 |
| train elapsed | 637 s（≈10.6 min），peak VRAM 2.24 GB |

| eval_seed | succ / 20 | rate |
|:---:|:---:|:---:|
| 99  | 6/20 | 30 % |
| 100 | 6/20 | 30 % |
| 101 | 4/20 | 20 % |
| 102 | 4/20 | 20 % |
| 103 | 5/20 | 25 % |
| **Pooled** | **25/100** | **25.0 %** |
| mean ± std | — | 25.0 % ± 5.0 pt |

**Wall-clock**：orchestrator 1880 s ≈ 31 min（train 637 s + 5 路并行 eval 21 min）。JSON 存档：`exps/k3_plan_A_results/k3_A_bundle/`。

**分析**：pooled 25 % 低于预设 30 % 通过线、无 eval_seed ≥ 50 %。vs R9f flat D040 @ RDNA4 GPU (42.8 %)：Δ = −17.8 pt。差距候选：(a) kitchen 环境复杂度、(b) 单 train-seed 偶然偏低、(c) CDNA3 CPU-render vs RDNA4 GPU-render 的 train-eval gap。eval-seed std 仅 5 pt 说明单 eval_seed × 20 trials 是可靠点估；差距来源由 **K3-B/K3-C 消融**给出答案（见下）。

---

### Exp-K3-B：RDNA4 train + RDNA4 GPU-eval（render-gap 消融）

**Config**：复用 K3-A 框架，只改节点 → RDNA4 R9700、render → GPU Vulkan、dataset → 本地 K2 产物、eval → 单卡串行。脚本 patch commit `2c077f6`。

**训练**（seed=0）：loss 0.6706 → 0.0161，elapsed **444 s (7.4 min, RDNA4 快 30 %)**，peak VRAM 2.33 GB。vs K3-A loss 曲线几乎重叠（CDNA3 end 0.0162）→ **ckpt 本质相同**。

**评估**：

| eval_seed | K3-B RDNA4 GPU | K3-A CDNA3 CPU | Δ |
|:---:|:---:|:---:|:---:|
| 99  | 10/20 = **50 %** | 6/20 = 30 % | +20 |
| 100 | 7/20 = 35 %     | 6/20 = 30 % | +5 |
| 101 | 8/20 = 40 %     | 4/20 = 20 % | +20 |
| 102 | 12/20 = **60 %** | 4/20 = 20 % | +40 |
| 103 | 11/20 = **55 %** | 5/20 = 25 % | +30 |
| **Pooled** | **48/100 = 48.0 %** | 25/100 = 25.0 % | **+23.0** |
| mean ± std | 48.0 % ± 10.4 pt | 25.0 % ± 5.0 pt | — |

Wall-clock：1288 s ≈ 21.5 min（train 444 + 串行 eval 840）。JSON：`exps/k3b_rdna4_results/k3b_bundle/`。

**分析**：
- Pooled 48 % ≫ 30 % 通过线，3 个 eval_seed ≥ 50 %，**K3 主假设（kitchen + up+wrist 100ep 能训出 demo-quality policy）成立**。
- vs R9f flat baseline 42.8 %（同 GPU render）：kitchen **略优 +5 pt**——kitchen 视觉先验可能起正则作用，不构成额外难度。
- **Render gap 量化：−23 pt**（同 ckpt、同 eval_seed、仅切 render backend）。CDNA3 MI 节点 CPU llvmpipe 的输出与训练用 GPU render 在抗锯齿/HDR/纹理过滤上有系统性偏差。
- Std 上升 (5 → 10.4 pt)：GPU render 下 policy 对 cube 位置更敏感；CPU render 因失败率高反而压缩了方差（伪稳定）。

---

### Exp-K3-C：CDNA3-trained ckpt × RDNA4 GPU-eval（填 2×2 矩阵左下角）

**Config**：K3-B 证明 RDNA4 train+eval 同 pipeline 下 48 %，但 K3-A 25 % 的 23 pt gap 还需拆分 (a) eval render、(b) train stack。把 K3-A `seed0/final` 搬到 RDNA4 eval 即可直接量化两者贡献。

ckpt 传输：ckpt 存档 `lidavidsh/smolvla-kitchen-wrist-k3a-cdna3-seed0`（private）。Eval 参数完全对齐 K3-B。

**Results**（2×2 矩阵）：

| train \ eval render | CDNA3 CPU (llvmpipe) | RDNA4 GPU (Vulkan) |
|:---:|:---:|:---:|
| **CDNA3 (ROCm6.4+Torch2.6)** | K3-A: **25.0 %** | **K3-C: 45.0 %** |
| **RDNA4 (ROCm7.2+Torch2.9)** | —（未做，下方解释） | K3-B: **48.0 %** |

K3-C per-seed：45/35/55/55/35 %，mean 45.0 % ± 10.0 pt，min/max 35 %/55 %。Wall-clock：ckpt pull 35 s + 5 × eval (avg 163 s) = 853 s ≈ 14.2 min。JSON：`exps/k3c_rdna4_results/`。

#### K3 系列结论

1. **Eval 渲染后端主导性能**：CPU llvmpipe 相对 GPU Vulkan 系统性低估 success rate 约 **20 pt**（同 ckpt 在 CPU eval 25 %、GPU eval 45 %）→ **visual policy eval 必须用 GPU 渲染节点**（R9700 或其他），CDNA3 MI 节点上的 `--render-cpu` 仅限 debug。
2. **训练节点/stack 不影响最终策略**：CDNA3 (ROCm 6.4 + Torch 2.6) 与 RDNA4 (ROCm 7.2 + Torch 2.9) 在相同 GPU eval 下 45 % vs 48 %，差距在 eval 自身 std (~10 pt) 之内 → **CDNA3 可当纯训练节点使用**，产出 ckpt 与 RDNA4 等效。
3. **kitchen + up+wrist 100 ep 达到 demo 要求**：正确 eval 下 pooled **45–48 %**、多 seed ≥ 50 %，与 flat baseline 同量级。
4. **HF 是好用的跨节点 ckpt 中转通道**（实测 160 MB/s 上 / 30 MB/s 下），后续跨节点实验沿用。

**Workshop 主路径决策**：
- ✅ `up+wrist + kitchen` 可作**主推高级示例**，数据 `lidavidsh/franka-pick-kitchen-up-wrist-100ep-genesis`（public）、ckpt `lidavidsh/smolvla-kitchen-wrist-k3a-cdna3-seed0`（private）均已归档。
- ⚠️ 文档须显式声明："visual policy eval 请在 R9700 或其它 GPU-render 节点；CDNA3 MI 节点只用于训练。"
- ⚠️ `04_eval_custom_scene.py --render-cpu` help 加警告："CPU rendering introduces ~20 pt success-rate bias vs GPU render; use for debug only, not for benchmarking."

**Next Step**（可选）：
1. `04_eval_custom_scene.py --render-cpu` help 加 known-limitation 警告。
2. README / notebook 高级主题加 kitchen+wrist 路径 + eval 节点要求备注。
3. （未来）demo 要更高 success rate：RDNA4 train-seed 消融 (1–4) + 100→200 ep。

（K3-C 完成 @ 2026-04-17 07:53 UTC, 14.2 min）


