"""Pre-compile Taichi kernels for the workshop kitchen scene (CPU backend).

Run once during Docker build to populate the Taichi offline cache.
Subsequent scene.build() calls reuse cached kernels (< 1 min vs 20-30 min).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPT_DIR))

from genesis_scene_utils import ensure_display
ensure_display()

import genesis as gs
gs.init(backend=gs.cpu, logging_level="warning")

from pick_common import add_pick_args, build_scene

ap = argparse.ArgumentParser()
add_pick_args(ap)
ap.set_defaults(
    scene="rustic_kitchen",
    anchor="floor_origin",
    camera_layout="up_wrist",
)
args = ap.parse_args([])

scene, *_ = build_scene(args, gs)
scene.build()

for _ in range(10):
    scene.step()

print("[warmup] Taichi kernel cache populated successfully.")
