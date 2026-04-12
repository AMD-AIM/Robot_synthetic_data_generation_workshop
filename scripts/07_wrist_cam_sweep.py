"""
Automated wrist camera parameter sweep with visibility scoring.

Generates pick-cube trajectories and scores each wrist camera configuration
by measuring cube (red pixel) visibility across frames. No LeRobot dependency.

Usage:
  python scripts/07_wrist_cam_sweep.py --out-dir /output/wrist_sweep
  python scripts/07_wrist_cam_sweep.py --out-dir /output/wrist_sweep --top-k 5 --episodes-per-candidate 3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---- Franka constants (same as 01_gen_data.py) ----
JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4",
    "joint5", "joint6", "joint7",
    "finger_joint1", "finger_joint2",
]
HOME_QPOS = np.array([0, -0.3, 0, -2.2, 0, 2.0, 0.79, 0.04, 0.04], dtype=np.float32)
KP = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100], dtype=np.float32)
KV = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10], dtype=np.float32)
FORCE_LOWER = np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100], dtype=np.float32)
FORCE_UPPER = np.array([87, 87, 87, 87, 12, 12, 12, 100, 100], dtype=np.float32)
CUBE_SIZE = (0.04, 0.04, 0.04)
GRASP_QUAT = np.array([0, 1, 0, 0], dtype=np.float32)
FINGER_OPEN = 0.04
FINGER_CLOSED = 0.01

CUBE_COLOR = np.array([255, 77, 77], dtype=np.uint8)  # (1.0, 0.3, 0.3) * 255
RED_THRESHOLD = 60  # max color distance to count as "cube pixel"
MIN_CUBE_PIXELS = 50  # minimum pixels to count as "visible"


@dataclass
class CandidateConfig:
    name: str
    mount_link: str  # "hand" or "panda_link7"
    pos: tuple[float, float, float]
    lookat: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov: float = 70.0


@dataclass
class SweepResult:
    name: str
    mount_link: str
    pos: tuple
    lookat: tuple
    fov: float
    grasp_vis_rate: float  # cube visibility in grasp phase (frames 70-89)
    grasp_avg_area: float  # avg cube pixel area in grasp phase
    lift_vis_rate: float   # cube visibility in lift phase (frames 90-119)
    overall_weighted_vis: float  # phase-weighted visibility (approach excluded)
    score: float = 0.0  # composite score


def ensure_display():
    if os.environ.get("DISPLAY"):
        return
    xvfb = subprocess.run(["which", "Xvfb"], capture_output=True)
    if xvfb.returncode != 0:
        return
    proc = subprocess.Popen(
        ["Xvfb", ":99", "-screen", "0", "1280x1024x24", "-ac", "+extension", "GLX"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    os.environ["DISPLAY"] = ":99"
    time.sleep(2)


def to_numpy(t):
    arr = t.cpu().numpy() if hasattr(t, "cpu") else np.array(t)
    return arr[0] if arr.ndim > 1 else arr


def lerp(a, b, n):
    a, b = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
    return [a + (b - a) * (i + 1) / max(n, 1) for i in range(n)]


def render_cam(cam):
    rgb, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
    arr = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.array(rgb)
    if arr.ndim == 4:
        arr = arr[0]
    return arr.astype(np.uint8)


def count_cube_pixels(img: np.ndarray) -> int:
    """Count pixels close to cube color using L1 distance in RGB."""
    diff = np.abs(img.astype(np.int16) - CUBE_COLOR.astype(np.int16))
    dist = diff.sum(axis=-1)
    return int((dist < RED_THRESHOLD).sum())


def build_candidate_grid() -> list[CandidateConfig]:
    """Build parameter grid focused on close-range contact-sensing geometry.

    Design principles (per real wrist-cam practice):
    - Wrist cam serves ONLY the last 5-10cm of grasp, not approach
    - Optical axis must point at gripper closing region, not table
    - Camera 5-12cm from gripper center, not 20-36cm
    - panda_link7 preferred (avoids finger self-occlusion)
    """
    candidates = []
    idx = 0

    # Primary: panda_link7, close-range, gripper-center lookat
    # Distance sweep: 5/8/12 cm, angle via pos_z variation
    for px in [0.05, 0.08, 0.12]:
        for pz in [0.03, 0.06, 0.10]:
            for lz in [0.00, -0.03]:
                for fov in [55, 65, 80]:
                    candidates.append(CandidateConfig(
                        name=f"L{idx:03d}",
                        mount_link="panda_link7",
                        pos=(px, 0.0, pz),
                        lookat=(0.0, 0.0, lz),
                        fov=fov,
                    ))
                    idx += 1

    # Secondary: hand link, close-range (control group, expect some occlusion)
    for px in [0.06, 0.10]:
        for pz in [0.04, 0.08]:
            for fov in [60, 75]:
                candidates.append(CandidateConfig(
                    name=f"H{idx:03d}",
                    mount_link="hand",
                    pos=(px, 0.0, pz),
                    lookat=(0.0, 0.0, 0.0),
                    fov=fov,
                ))
                idx += 1

    return candidates


def evaluate_candidate(
    scene, franka, cube, cam_wrist, candidate: CandidateConfig,
    cube_positions: list[tuple[float, float]],
    gs_module, torch_module,
) -> SweepResult:
    """Run episodes and compute visibility metrics for one candidate."""
    from genesis.utils.geom import pos_lookat_up_to_T

    motors_dof = [franka.get_joint(name).dofs_idx_local[0] for name in JOINT_NAMES]
    n_dofs = len(JOINT_NAMES)
    end_effector = franka.get_link(candidate.mount_link)
    cube_z = CUBE_SIZE[2] / 2.0

    wrist_pos = torch_module.tensor(list(candidate.pos), dtype=gs_module.tc_float, device=gs_module.device)
    wrist_lookat = torch_module.tensor(list(candidate.lookat), dtype=gs_module.tc_float, device=gs_module.device)
    wrist_up = torch_module.tensor(list(candidate.up), dtype=gs_module.tc_float, device=gs_module.device)
    offset_T = pos_lookat_up_to_T(wrist_pos, wrist_lookat, wrist_up)

    try:
        cam_wrist.attach(rigid_link=end_effector, offset_T=offset_T)
    except TypeError:
        cam_wrist.attach(end_effector, offset_T)

    def solve_ik(pos, quat=GRASP_QUAT, finger_pos=FINGER_OPEN):
        qpos = to_numpy(franka.inverse_kinematics(
            link=franka.get_link("hand"), pos=np.array(pos, dtype=np.float32),
            quat=np.array(quat, dtype=np.float32),
        ))
        target = np.zeros(n_dofs, dtype=np.float32)
        target[:7] = qpos[:7]
        target[7] = finger_pos
        target[8] = finger_pos
        return target

    # Phase boundaries within each episode (135 frames total):
    #   approach:  0-39  (overhead cam's job, wrist cam ignored)
    #   pre_grasp: 40-69 (descent toward cube)
    #   grasp:     70-89 (CRITICAL — contact sensing)
    #   lift:      90-119
    #   hold:      120-134
    N_APPROACH, N_PREGRSP, N_GRASP, N_LIFT, N_HOLD = 40, 30, 20, 30, 15

    phase_pixels = {"pre_grasp": [], "grasp": [], "lift": [], "hold": []}

    for cx, cy in cube_positions:
        franka.set_dofs_position(HOME_QPOS, motors_dof)
        franka.control_dofs_position(HOME_QPOS, motors_dof)
        franka.zero_all_dofs_velocity()
        cube.set_pos(torch_module.tensor([cx, cy, cube_z], dtype=torch_module.float32, device=gs_module.device).unsqueeze(0))
        cube.set_quat(torch_module.tensor([1, 0, 0, 0], dtype=torch_module.float32, device=gs_module.device).unsqueeze(0))
        cube.zero_all_dofs_velocity()
        for _ in range(30):
            scene.step()

        q_home = HOME_QPOS.copy()
        q_hover = solve_ik([cx, cy, 0.25], finger_pos=FINGER_OPEN)
        q_grasp = solve_ik([cx, cy, 0.135], finger_pos=FINGER_OPEN)
        q_close = solve_ik([cx, cy, 0.135], finger_pos=FINGER_CLOSED)
        q_lift = solve_ik([cx, cy, 0.30], finger_pos=FINGER_CLOSED)

        traj = (
            lerp(q_home, q_hover, N_APPROACH) +
            lerp(q_hover, q_grasp, N_PREGRSP) +
            lerp(q_grasp, q_close, N_GRASP) +
            lerp(q_close, q_lift, N_LIFT) +
            [q_lift.copy() for _ in range(N_HOLD)]
        )

        for fi, target in enumerate(traj):
            franka.control_dofs_position(np.array(target, dtype=np.float32), motors_dof)
            scene.step()
            if fi < N_APPROACH:
                continue
            img = render_cam(cam_wrist)
            px = count_cube_pixels(img)
            if fi < N_APPROACH + N_PREGRSP:
                phase_pixels["pre_grasp"].append(px)
            elif fi < N_APPROACH + N_PREGRSP + N_GRASP:
                phase_pixels["grasp"].append(px)
            elif fi < N_APPROACH + N_PREGRSP + N_GRASP + N_LIFT:
                phase_pixels["lift"].append(px)
            else:
                phase_pixels["hold"].append(px)

    try:
        cam_wrist.detach()
    except Exception:
        pass

    def _vis_rate(pixels):
        if not pixels:
            return 0.0
        return sum(1 for p in pixels if p >= MIN_CUBE_PIXELS) / len(pixels)

    def _avg_area(pixels):
        visible = [p for p in pixels if p >= MIN_CUBE_PIXELS]
        return float(np.mean(visible)) if visible else 0.0

    grasp_vis = _vis_rate(phase_pixels["grasp"])
    grasp_area = _avg_area(phase_pixels["grasp"])
    lift_vis = _vis_rate(phase_pixels["lift"])
    pregrsp_vis = _vis_rate(phase_pixels["pre_grasp"])

    # Phase-weighted visibility (approach excluded entirely)
    # Grasp is 2x more important than other phases
    weighted_vis = (
        pregrsp_vis * 0.15 +
        grasp_vis * 0.50 +
        lift_vis * 0.25 +
        _vis_rate(phase_pixels["hold"]) * 0.10
    )

    # Composite score: weighted visibility (60%) + grasp area quality (40%)
    score = weighted_vis * 0.6 + min(grasp_area / 2000.0, 1.0) * 0.4

    return SweepResult(
        name=candidate.name,
        mount_link=candidate.mount_link,
        pos=candidate.pos,
        lookat=candidate.lookat,
        fov=candidate.fov,
        grasp_vis_rate=grasp_vis,
        grasp_avg_area=grasp_area,
        lift_vis_rate=lift_vis,
        overall_weighted_vis=weighted_vis,
        score=score,
    )


def save_topk_images(
    scene, franka, cube, cam_wrist, cam_up,
    candidate: CandidateConfig,
    cube_positions: list[tuple[float, float]],
    out_dir: Path, gs_module, torch_module,
):
    """Generate visualization grid for a top candidate (like 05_visualize)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from genesis.utils.geom import pos_lookat_up_to_T

    motors_dof = [franka.get_joint(name).dofs_idx_local[0] for name in JOINT_NAMES]
    n_dofs = len(JOINT_NAMES)
    end_effector = franka.get_link(candidate.mount_link)
    cube_z = CUBE_SIZE[2] / 2.0

    wrist_pos = torch_module.tensor(list(candidate.pos), dtype=gs_module.tc_float, device=gs_module.device)
    wrist_lookat = torch_module.tensor(list(candidate.lookat), dtype=gs_module.tc_float, device=gs_module.device)
    wrist_up = torch_module.tensor(list(candidate.up), dtype=gs_module.tc_float, device=gs_module.device)
    offset_T = pos_lookat_up_to_T(wrist_pos, wrist_lookat, wrist_up)
    try:
        cam_wrist.attach(rigid_link=end_effector, offset_T=offset_T)
    except TypeError:
        cam_wrist.attach(end_effector, offset_T)

    def solve_ik(pos, quat=GRASP_QUAT, finger_pos=FINGER_OPEN):
        qpos = to_numpy(franka.inverse_kinematics(
            link=franka.get_link("hand"), pos=np.array(pos, dtype=np.float32),
            quat=np.array(quat, dtype=np.float32),
        ))
        target = np.zeros(n_dofs, dtype=np.float32)
        target[:7] = qpos[:7]
        target[7] = finger_pos
        target[8] = finger_pos
        return target

    cx, cy = cube_positions[0]
    franka.set_dofs_position(HOME_QPOS, motors_dof)
    franka.control_dofs_position(HOME_QPOS, motors_dof)
    franka.zero_all_dofs_velocity()
    cube.set_pos(torch_module.tensor([cx, cy, cube_z], dtype=torch_module.float32, device=gs_module.device).unsqueeze(0))
    cube.set_quat(torch_module.tensor([1, 0, 0, 0], dtype=torch_module.float32, device=gs_module.device).unsqueeze(0))
    cube.zero_all_dofs_velocity()
    for _ in range(30):
        scene.step()

    q_home = HOME_QPOS.copy()
    q_hover = solve_ik([cx, cy, 0.25], finger_pos=FINGER_OPEN)
    q_grasp = solve_ik([cx, cy, 0.135], finger_pos=FINGER_OPEN)
    q_close = solve_ik([cx, cy, 0.135], finger_pos=FINGER_CLOSED)
    q_lift = solve_ik([cx, cy, 0.30], finger_pos=FINGER_CLOSED)

    traj = (
        lerp(q_home, q_hover, 40) +
        lerp(q_hover, q_grasp, 30) +
        lerp(q_grasp, q_close, 20) +
        lerp(q_close, q_lift, 30) +
        [q_lift.copy() for _ in range(15)]
    )

    # Bias sampling toward grasp-relevant phases:
    # 1 approach, 1 pre-grasp, 2 grasp, 1 lift, 1 hold
    sample_indices = np.array([20, 55, 75, 85, 105, 130], dtype=int)
    sample_labels = ["approach", "pre_grasp", "grasp_start", "grasp_end", "lift", "hold"]
    n_sample = len(sample_indices)
    imgs_up, imgs_wrist = [], []

    for i, target in enumerate(traj):
        franka.control_dofs_position(np.array(target, dtype=np.float32), motors_dof)
        scene.step()
        if i in sample_indices:
            imgs_up.append(render_cam(cam_up))
            imgs_wrist.append(render_cam(cam_wrist))

    fig, axes = plt.subplots(2, n_sample, figsize=(3 * n_sample, 6))
    for col in range(n_sample):
        axes[0, col].imshow(imgs_up[col])
        axes[0, col].set_title(f"f{sample_indices[col]} ({sample_labels[col]})", fontsize=7)
        axes[0, col].axis("off")
        axes[1, col].imshow(imgs_wrist[col])
        axes[1, col].axis("off")
    axes[0, 0].set_ylabel("up", fontsize=10)
    axes[1, 0].set_ylabel("wrist", fontsize=10)

    title = f"{candidate.name} | {candidate.mount_link} | pos={candidate.pos} fov={candidate.fov}"
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()

    out_path = out_dir / f"{candidate.name}_camera_views.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)

    try:
        cam_wrist.detach()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="Automated wrist camera parameter sweep")
    ap.add_argument("--out-dir", required=True, help="Output directory for results")
    ap.add_argument("--episodes-per-candidate", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=5, help="Generate visualization for top-K candidates")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-bbox-detection", action="store_true")
    args = ap.parse_args()

    ensure_display()
    import genesis as gs
    import torch

    gs.init(backend=(gs.cpu if args.cpu else gs.gpu), logging_level="warning")

    cube_z = CUBE_SIZE[2] / 2.0
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 30, substeps=4),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True, enable_joint_limit=True,
            box_box_detection=(not args.no_bbox_detection),
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    cube = scene.add_entity(
        morph=gs.morphs.Box(size=CUBE_SIZE, pos=(0.55, 0.0, cube_z)),
        material=gs.materials.Rigid(friction=1.5),
        surface=gs.surfaces.Default(color=(1.0, 0.3, 0.3, 1.0)),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    cam_up = scene.add_camera(
        res=(640, 480), pos=(0.55, 0.55, 0.55),
        lookat=(0.55, 0.0, 0.10), fov=45, GUI=False,
    )
    cam_wrist = scene.add_camera(
        res=(640, 480), pos=(0.1, 0.0, 0.1),
        lookat=(0.0, 0.0, 0.0), fov=65, GUI=False,
    )
    scene.build()

    motors_dof = [franka.get_joint(name).dofs_idx_local[0] for name in JOINT_NAMES]
    franka.set_dofs_kp(KP, motors_dof)
    franka.set_dofs_kv(KV, motors_dof)
    franka.set_dofs_force_range(FORCE_LOWER, FORCE_UPPER, motors_dof)

    rng = np.random.RandomState(args.seed)
    cube_positions = [
        (rng.uniform(0.4, 0.7), rng.uniform(-0.2, 0.2))
        for _ in range(args.episodes_per_candidate)
    ]

    candidates = build_candidate_grid()
    print(f"[sweep] {len(candidates)} candidates × {args.episodes_per_candidate} episodes")
    print(f"[sweep] cube positions: {cube_positions}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[SweepResult] = []
    t0 = time.time()

    for i, cand in enumerate(candidates):
        r = evaluate_candidate(
            scene, franka, cube, cam_wrist, cand,
            cube_positions, gs, torch,
        )
        results.append(r)
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(candidates) - i - 1)
        print(f"[sweep] {i+1}/{len(candidates)} {cand.name} "
              f"grasp_vis={r.grasp_vis_rate:.0%} grasp_area={r.grasp_avg_area:.0f} "
              f"score={r.score:.3f} (ETA {eta:.0f}s)")

    results.sort(key=lambda r: r.score, reverse=True)

    csv_path = out_dir / "sweep_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "name", "mount", "pos_x", "pos_y", "pos_z",
                         "lookat_x", "lookat_y", "lookat_z", "fov",
                         "grasp_vis", "grasp_area", "lift_vis",
                         "weighted_vis", "score"])
        for rank, r in enumerate(results):
            writer.writerow([
                rank + 1, r.name, r.mount_link,
                r.pos[0], r.pos[1], r.pos[2],
                r.lookat[0], r.lookat[1], r.lookat[2],
                r.fov, f"{r.grasp_vis_rate:.3f}",
                f"{r.grasp_avg_area:.0f}", f"{r.lift_vis_rate:.3f}",
                f"{r.overall_weighted_vis:.3f}", f"{r.score:.4f}",
            ])
    print(f"\n[sweep] results saved to {csv_path}")

    print(f"\n{'='*70}")
    print(f"TOP {args.top_k} CANDIDATES")
    print(f"{'='*70}")
    for rank, r in enumerate(results[:args.top_k]):
        print(f"  #{rank+1} {r.name} [{r.mount_link}] "
              f"pos={r.pos} lookat={r.lookat} fov={r.fov}")
        print(f"       grasp_vis={r.grasp_vis_rate:.0%} grasp_area={r.grasp_avg_area:.0f} "
              f"lift_vis={r.lift_vis_rate:.0%} score={r.score:.3f}")

    print(f"\n[sweep] generating top-{args.top_k} visualizations...")
    top_candidates = [c for c in candidates if c.name in {r.name for r in results[:args.top_k]}]
    for cand in top_candidates:
        save_topk_images(scene, franka, cube, cam_wrist, cam_up,
                         cand, cube_positions, out_dir / "top_views", gs, torch)
        print(f"[sweep] saved {cand.name} visualization")

    summary = {
        "total_candidates": len(candidates),
        "episodes_per_candidate": args.episodes_per_candidate,
        "cube_positions": [(float(x), float(y)) for x, y in cube_positions],
        "elapsed_seconds": time.time() - t0,
        "top_results": [
            {
                "rank": i + 1, "name": r.name, "mount_link": r.mount_link,
                "pos": list(r.pos), "lookat": list(r.lookat), "fov": r.fov,
                "grasp_vis_rate": r.grasp_vis_rate,
                "grasp_avg_area": r.grasp_avg_area,
                "lift_vis_rate": r.lift_vis_rate,
                "overall_weighted_vis": r.overall_weighted_vis,
                "score": r.score,
            }
            for i, r in enumerate(results[:args.top_k])
        ],
    }
    (out_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[sweep] done in {summary['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    main()
