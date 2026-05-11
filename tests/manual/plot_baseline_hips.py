"""Plot hip-COM trajectory from the git-committed baseline npz and the latest MPS run."""
import subprocess, tempfile, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
FIXTURE_REL = "tests/data/zelda_20260428_151934_fga_horizontal_head_turns"
FIXTURE_DIR = REPO / FIXTURE_REL
L_HIP, R_HIP = 11, 12

def load_from_git(name):
    tmp = Path(tempfile.gettempdir()) / f"_bl_{name}"
    subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{FIXTURE_REL}/{name}"],
                   stdout=open(tmp, "wb"), check=True)
    return np.load(str(tmp), allow_pickle=True)

def hip_com(kps_slot):
    """kps_slot: (frames, 52, 3) -> (frames, 3) hip COM."""
    l = kps_slot[:, L_HIP, :]
    r = kps_slot[:, R_HIP, :]
    l_ok = np.isfinite(l).all(axis=-1, keepdims=True)
    r_ok = np.isfinite(r).all(axis=-1, keepdims=True)
    both = l_ok & r_ok
    com = np.where(both, (l + r) / 2, np.where(l_ok, l, np.where(r_ok, r, np.nan)))
    return com

# Load baseline from git
bl = load_from_git("keypoints_3d.npz")
bl_kp = bl["keypoints_3d"]
bl_ts = bl["timestamps"]
bl_R = bl["view_transform_R"]
bl_t = bl["view_transform_t"]

print(f"=== Baseline (git HEAD) ===")
print(f"  shape: {bl_kp.shape}  frames={bl_kp.shape[0]}")
print(f"  view_transform_R:\n{bl_R}")
print(f"  view_transform_t: {bl_t}")

# Load fresh run output (prefer working-tree keypoints_3d.npz over stale .mps.npz)
mps_path = FIXTURE_DIR / "keypoints_3d.npz"
if not mps_path.exists():
    mps_path = FIXTURE_DIR / "keypoints_3d.mps.npz"
has_mps = mps_path.exists()
if has_mps:
    mps = np.load(str(mps_path), allow_pickle=True)
    mps_kp = mps["keypoints_3d"]
    mps_ts = mps["timestamps"]
    mps_R = mps["view_transform_R"]
    mps_t = mps["view_transform_t"]
    print(f"\n=== MPS run ===")
    print(f"  shape: {mps_kp.shape}  frames={mps_kp.shape[0]}")
    print(f"  view_transform_R:\n{mps_R}")
    print(f"  view_transform_t: {mps_t}")
    print(f"\n  View transforms match: R={np.allclose(bl_R, mps_R, atol=1e-6)}, "
          f"t={np.allclose(bl_t, mps_t, atol=1e-6)}")

# Plot
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not available, skipping plot")
    sys.exit(0)

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
labels = ["X (m)", "Y (m)", "Z (m)"]

# Find the active person slot in baseline (first with data)
for p in range(bl_kp.shape[1]):
    finite = np.isfinite(bl_kp[:, p, :, :]).all(axis=-1).any(axis=-1)
    if finite.any():
        bl_com = hip_com(bl_kp[:, p])
        for ax_i in range(3):
            axes[ax_i].plot(bl_ts, bl_com[:, ax_i], 'o-', markersize=2,
                           label=f"baseline slot{p}", alpha=0.7)
        break

if has_mps:
    for p in range(mps_kp.shape[1]):
        finite = np.isfinite(mps_kp[:, p, :, :]).all(axis=-1).any(axis=-1)
        if finite.any():
            mps_com = hip_com(mps_kp[:, p])
            for ax_i in range(3):
                axes[ax_i].plot(mps_ts, mps_com[:, ax_i], '.-', markersize=1,
                               label=f"mps slot{p}", alpha=0.7)
            break

for ax_i in range(3):
    axes[ax_i].set_ylabel(labels[ax_i])
    axes[ax_i].legend(loc="upper right")
    axes[ax_i].grid(True, alpha=0.3)

axes[0].set_title("Hip-COM trajectory: baseline (git) vs MPS run")
axes[2].set_xlabel("Time (s)")
plt.tight_layout()

out = FIXTURE_DIR / "hip_com_baseline_vs_mps.png"
plt.savefig(str(out), dpi=150)
print(f"\nPlot saved: {out}")
