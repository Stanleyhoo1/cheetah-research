"""
View an AcinoSet 3D reconstruction (fte.pickle, ekf.pickle, sba.pickle or fte.mat)
as an animated 3D cheetah skeleton.

Usage:
    python view_fte.py path/to/run/fte/fte.pickle            # interactive window
    python view_fte.py path/to/run/fte/fte.pickle --save run.mp4   # or run.gif
    python view_fte.py path/to/run                             # finds fte/fte.pickle
    python view_fte.py fte.pickle --static                     # whole run, fixed axes

In the window: drag to rotate, space to pause, left/right arrows to step frames.
Left limbs are blue and right limbs are orange, so a left/right swap shows up
as the colours jumping between legs.
"""
import argparse
import os
import pickle
import sys

import numpy as np
import matplotlib

if "--save" in sys.argv:
    matplotlib.use("Agg")  # no window needed when only saving a file
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Marker order used by AcinoSet's code when it saved positions (lib/misc.py, get_markers).
MARKERS = [
    "nose", "l_eye", "r_eye", "neck_base",
    "spine", "tail_base", "tail1", "tail2",
    "r_shoulder", "r_front_knee", "r_front_ankle",
    "l_shoulder", "l_front_knee", "l_front_ankle",
    "r_hip", "r_back_knee", "r_back_ankle",
    "l_hip", "l_back_knee", "l_back_ankle",
]

# Bones, from AcinoSet's get_skeleton().
LINKS = [
    ("nose", "l_eye"), ("nose", "r_eye"), ("nose", "neck_base"),
    ("l_eye", "neck_base"), ("r_eye", "neck_base"),
    ("neck_base", "spine"), ("spine", "tail_base"),
    ("tail_base", "tail1"), ("tail1", "tail2"),
    ("neck_base", "r_shoulder"), ("r_shoulder", "r_front_knee"), ("r_front_knee", "r_front_ankle"),
    ("neck_base", "l_shoulder"), ("l_shoulder", "l_front_knee"), ("l_front_knee", "l_front_ankle"),
    ("tail_base", "r_hip"), ("r_hip", "r_back_knee"), ("r_back_knee", "r_back_ankle"),
    ("tail_base", "l_hip"), ("l_hip", "l_back_knee"), ("l_back_knee", "l_back_ankle"),
]

LEFT, RIGHT, BODY = "#2f6fdb", "#e8771e", "#555555"


def link_colour(a, b):
    """Colour a bone by side: legs only (eyes and spine stay grey)."""
    leg = [m for m in (a, b) if not m.endswith("eye")]
    if any(m.startswith("l_") for m in leg):
        return LEFT
    if any(m.startswith("r_") for m in leg):
        return RIGHT
    return BODY


def find_file(path):
    """Accept a results file, an fte/ folder, or a run folder."""
    if os.path.isfile(path):
        return path
    for candidate in ("fte/fte.pickle", "fte.pickle", "fte/fte.mat", "fte.mat"):
        p = os.path.join(path, candidate)
        if os.path.isfile(p):
            return p
    sys.exit(f"No fte.pickle or fte.mat found in {path}")


def load_positions(fpath):
    """Return (positions [N, 20, 3] in metres, first frame number, label)."""
    label = os.path.splitext(os.path.basename(fpath))[0].upper()
    if fpath.endswith(".mat"):
        from scipy.io import loadmat
        data = loadmat(fpath)
        positions = np.asarray(data["positions"], dtype=float)
        start = int(np.ravel(data.get("start_frame", [0]))[0])
    else:
        with open(fpath, "rb") as f:
            data = pickle.load(f)
        # EKF files store a smoothed version as well; prefer it, as AcinoSet does.
        key = "smoothed_positions" if "smoothed_positions" in data else "positions"
        positions = np.asarray(data[key], dtype=float)
        start = int(data.get("start_frame", 0))
    if positions.ndim != 3 or positions.shape[1:] != (len(MARKERS), 3):
        sys.exit(f"Unexpected positions shape {positions.shape}; expected (frames, {len(MARKERS)}, 3)")
    # AcinoSet stores start_frame 0-based; show 1-based frame numbers to match the videos' frame count.
    return positions, start, label


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="fte.pickle / fte.mat, or the run folder that contains fte/")
    ap.add_argument("--save", help="write an .mp4 or .gif instead of opening a window")
    ap.add_argument("--static", action="store_true", help="fixed axes over the whole run instead of following the cheetah")
    ap.add_argument("--fps", type=float, default=30, help="playback speed in frames per second (default 30)")
    args = ap.parse_args()

    fpath = find_file(args.path)
    pos, start, label = load_positions(fpath)
    n = len(pos)
    idx = {m: i for i, m in enumerate(MARKERS)}
    print(f"Loaded {fpath}: {n} frames, {len(MARKERS)} markers, starting at video frame {start + 1}")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_box_aspect((1, 1, 1))

    finite = pos[np.isfinite(pos).all(axis=2)]
    lo, hi = finite.min(axis=0), finite.max(axis=0)
    half = 0.6 * max(hi - lo) if args.static else 1.0  # 1 m either side when following

    if args.static:
        # Faint path of the neck base over the whole run, for context.
        ax.plot(*pos[:, idx["neck_base"]].T, color="#bbbbbb", lw=0.8)

    lines = [ax.plot([], [], [], color=link_colour(a, b), lw=2.5)[0] for a, b in LINKS]
    dots = ax.plot([], [], [], "o", color="#222222", ms=3)[0]
    title = ax.set_title("")

    state = {"i": 0, "paused": False}

    def draw(i):
        p = pos[i]
        for line, (a, b) in zip(lines, LINKS):
            seg = p[[idx[a], idx[b]]]
            line.set_data(seg[:, 0], seg[:, 1])
            line.set_3d_properties(seg[:, 2])
        dots.set_data(p[:, 0], p[:, 1])
        dots.set_3d_properties(p[:, 2])
        centre = np.nanmean(p, axis=0) if not args.static else (lo + hi) / 2
        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[1] - half, centre[1] + half)
        ax.set_zlim(centre[2] - half, centre[2] + half)
        title.set_text(f"{label}  video frame {start + i + 1}  ({i + 1}/{n})")
        return lines + [dots, title]

    def step(_):
        if not state["paused"]:
            state["i"] = (state["i"] + 1) % n
        return draw(state["i"])

    def on_key(event):
        if event.key == " ":
            state["paused"] = not state["paused"]
        elif event.key == "right":
            state["i"] = (state["i"] + 1) % n
        elif event.key == "left":
            state["i"] = (state["i"] - 1) % n

    if args.save:
        anim = FuncAnimation(fig, draw, frames=n, interval=1000 / args.fps)
        writer = "pillow" if args.save.lower().endswith(".gif") else "ffmpeg"
        anim.save(args.save, writer=writer, fps=args.fps)
        print(f"Saved {args.save}")
    else:
        fig.canvas.mpl_connect("key_press_event", on_key)
        anim = FuncAnimation(fig, step, interval=1000 / args.fps, cache_frame_data=False)  # noqa: F841
        plt.show()


if __name__ == "__main__":
    main()