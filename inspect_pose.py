"""
Step through pose predictions frame by frame: keypoints, their confidence,
and (for SuperAnimal output) the detector's bounding boxes.

Usage (run inside an environment with opencv, pandas, tables and matplotlib):
    python inspect_pose.py VIDEO PREDICTIONS.h5
    python inspect_pose.py VIDEO PREDICTIONS.h5 --json PREDICTIONS.json   # adds boxes (SuperAnimal)
    python inspect_pose.py VIDEO PREDICTIONS.h5 --export annotated.mp4    # write a video instead

If --json is not given, a .json file with the same name as the .h5 is used when it exists.

Colours: left body parts are blues and purples, right body parts are reds and oranges,
and midline parts (nose, spine, tail) are greens and yellows, so a left/right swap shows
up as the colour families jumping between legs. Points below --pcutoff are hollow grey.

Keys in the viewer window:
    right / left        next / previous frame
    up / down           10 frames forward / back
    pageup / pagedown   100 frames forward / back
    home / end          first / last frame
    space               play / pause
    hover over a dot    shows its name and confidence
    l                   show / hide the legend panel
    n                   show / hide names next to the dots
    c                   show / hide confidence next to the dots
    b                   show / hide bounding boxes
    h                   show / hide points below --pcutoff (e.g. ones a tracker marks as hidden)
    p                   print every keypoint of this frame in the terminal
    s                   save this frame (with legend) as a PNG
    The toolbar's magnifier zooms into a region (dots stay the same size); the house resets.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd


# ---------- loading ----------

def load_predictions(h5_path):
    """Return (bodyparts, individuals, array [frames, individuals, bodyparts, 3] of x, y, likelihood, frame ids)."""
    df = pd.read_hdf(h5_path)
    cols = df.columns
    if "scorer" in cols.names:
        df = df.droplevel("scorer", axis=1)
        cols = df.columns
    if "individuals" not in cols.names:  # single-animal file: add a dummy individual level
        df.columns = pd.MultiIndex.from_tuples(
            [("animal", *c) for c in cols], names=["individuals", *cols.names]
        )
        cols = df.columns
    individuals = list(dict.fromkeys(cols.get_level_values("individuals")))
    bodyparts = list(dict.fromkeys(cols.get_level_values("bodyparts")))
    n = len(df)
    arr = np.full((n, len(individuals), len(bodyparts), 3), np.nan)
    for i, ind in enumerate(individuals):
        for j, bp in enumerate(bodyparts):
            for k, c in enumerate(("x", "y", "likelihood")):
                if (ind, bp, c) in df.columns:
                    arr[:, i, j, k] = df[(ind, bp, c)].to_numpy(dtype=float)
    # DeepLabCut's top-down models write -1 for x, y and likelihood when the detector found
    # no animal in a frame. Treat those as missing, not as a point at pixel (-1, -1).
    no_detection = (arr[..., 0] == -1) & (arr[..., 1] == -1)
    arr[no_detection] = np.nan
    # Files without a likelihood column (e.g. 3D reprojections) count as fully confident.
    missing = np.isnan(arr[..., 2]) & ~np.isnan(arr[..., 0])
    arr[..., 2][missing] = 1.0
    frame_ids = df.index.to_numpy()
    if not np.issubdtype(frame_ids.dtype, np.integer):
        frame_ids = np.arange(n)
    return bodyparts, individuals, arr, frame_ids


def load_boxes(json_path):
    """Return a list (one entry per frame) of (boxes [N, 4] as x, y, w, h; scores [N])."""
    with open(json_path) as f:
        preds = json.load(f)
    out = []
    for p in preds:
        boxes = np.asarray(p.get("bboxes", []), dtype=float).reshape(-1, 4)
        scores = np.asarray(p.get("bbox_scores", [np.nan] * len(boxes)), dtype=float).ravel()
        out.append((boxes, scores))
    return out


# ---------- colours ----------

LEFT = ["#4e79ff", "#00b4d8", "#9b5de5", "#3a86ff", "#48cae4", "#7b2cbf", "#5e60ce", "#90e0ef",
        "#6930c3", "#0096c7", "#b5179e", "#4361ee"]
RIGHT = ["#ff3b30", "#ff9500", "#ff4d9d", "#e85d04", "#ffb703", "#d00000", "#fb8500", "#ff70a6",
         "#dc2f02", "#f48c06", "#c9184a", "#ffd166"]
MIDDLE = ["#2dc653", "#ffe600", "#ffffff", "#8ac926", "#c7f9cc", "#06d6a0", "#e9ff70", "#b5e48c",
          "#fcf6bd", "#38b000"]


def side(name):
    n = name.lower()
    if n.startswith("l_") or "left" in n:
        return "L"
    if n.startswith("r_") or "right" in n:
        return "R"
    return "M"


def assign_colours(bodyparts):
    """Hex colour per body part: left = blues/purples, right = reds/oranges, midline = greens/yellows."""
    used = {"L": 0, "R": 0, "M": 0}
    pools = {"L": LEFT, "R": RIGHT, "M": MIDDLE}
    out = []
    for bp in bodyparts:
        s = side(bp)
        out.append(pools[s][used[s] % len(pools[s])])
        used[s] += 1
    return out


def hex_to_bgr(h):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


GREY = "#a0a0a0"


# ---------- drawing for exported videos and saved frames (OpenCV) ----------

def outlined_text(img, s, org, scale, colour, thick):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def draw_cv(frame, pose, bodyparts, colours, boxes, opts):
    """Draw one frame with OpenCV: dots, optional labels, boxes, and a legend panel."""
    img = frame.copy()
    h, w = img.shape[:2]
    s = max(w, h) / 1600  # scale marks and text with the video size
    r, fs, th = max(2, int(opts["dot"] * s)), 0.4 * s, max(1, int(round(s)))

    if opts["boxes"] and boxes is not None:
        for (x, y, bw, bh), sc in zip(*boxes):
            p1, p2 = (int(x), int(y)), (int(x + bw), int(y + bh))
            cv2.rectangle(img, p1, p2, (0, 255, 255), max(1, int(2 * s)))
            label = "box" if np.isnan(sc) else f"box {sc:.2f}"
            outlined_text(img, label, (p1[0], max(p1[1] - int(6 * s), int(14 * s))), fs, (0, 255, 255), th)

    conf_now = np.full(len(bodyparts), np.nan)
    for ind in pose:
        for j, (x, y, lik) in enumerate(ind):
            if np.isnan(x) or np.isnan(y):
                continue
            conf_now[j] = np.nanmax([conf_now[j], lik])
            centre = (int(round(x)), int(round(y)))
            confident = lik >= opts["pcutoff"]
            if opts["hide_low"] and not confident:
                continue
            colour = hex_to_bgr(colours[j]) if confident else hex_to_bgr(GREY)
            cv2.circle(img, centre, r, colour, -1 if confident else max(1, th), cv2.LINE_AA)
            parts = ([bodyparts[j]] if opts["names"] else []) + ([f"{lik:.2f}"] if opts["conf"] else [])
            if parts:
                outlined_text(img, " ".join(parts), (centre[0] + r + 2, centre[1] - r), fs * 0.8,
                              (255, 255, 255) if confident else hex_to_bgr(GREY), th)

    if opts["legend"]:
        row_h = int(18 * s)
        pad = int(8 * s)
        width = int(230 * s)
        height = pad * 2 + row_h * len(bodyparts)
        panel = img[pad:pad + height, pad:pad + width]
        img[pad:pad + height, pad:pad + width] = (panel * 0.35).astype(img.dtype)  # darken behind legend
        for j, bp in enumerate(bodyparts):
            cy = pad * 2 + row_h * j + row_h // 2
            c = conf_now[j]
            ok = not np.isnan(c) and c >= opts["pcutoff"]
            colour = hex_to_bgr(colours[j]) if ok else hex_to_bgr(GREY)
            cv2.circle(img, (pad * 2 + int(5 * s), cy), max(2, int(5 * s)), colour, -1 if ok else 1, cv2.LINE_AA)
            txt = f"{bp}  {'-' if np.isnan(c) else f'{c:.2f}'}"
            cv2.putText(img, txt, (pad * 2 + int(16 * s), cy + int(5 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                        fs, (255, 255, 255) if ok else hex_to_bgr(GREY), th, cv2.LINE_AA)
    return img


# ---------- video access ----------

class Video:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            sys.exit(f"Could not open video {path}")
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.next_index = 0

    def read(self, i):
        if i != self.next_index:  # seeking is slower, so only do it when we jump
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = self.cap.read()
        self.next_index = i + 1
        return frame if ok else None


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("h5", help="pose predictions in DeepLabCut's .h5 format")
    ap.add_argument("--json", help="SuperAnimal .json with bounding boxes (default: same name as the .h5)")
    ap.add_argument("--pcutoff", type=float, default=0.5, help="confidence below which points are hollow and grey (default 0.5)")
    ap.add_argument("--start", type=int, default=0, help="first frame to show")
    ap.add_argument("--end", type=int, help="last frame (export only; default: end of video)")
    ap.add_argument("--dot", type=float, default=4, help="dot size (default 4)")
    ap.add_argument("--labels", action="store_true", help="start with names shown next to the dots")
    ap.add_argument("--hide-low", action="store_true", help="hide points below --pcutoff instead of drawing them hollow")
    ap.add_argument("--export", help="write an annotated .mp4 instead of opening a window")
    args = ap.parse_args()

    bodyparts, individuals, poses, frame_ids = load_predictions(args.h5)
    row_of = {int(f): i for i, f in enumerate(frame_ids)}
    json_path = args.json or os.path.splitext(args.h5)[0] + ".json"
    boxes = load_boxes(json_path) if os.path.isfile(json_path) else None
    vid = Video(args.video)
    colours = assign_colours(bodyparts)
    opts = {"names": args.labels, "conf": False, "boxes": True, "legend": True,
            "pcutoff": args.pcutoff, "dot": args.dot, "hide_low": args.hide_low}
    print(f"{len(bodyparts)} keypoints, {len(individuals)} individual(s), {len(frame_ids)} predicted frames, "
          f"{vid.n} video frames" + (f", boxes from {os.path.basename(json_path)}" if boxes else ", no box file"))

    def pose_at(i):
        row = row_of.get(i)
        return poses[row] if row is not None else np.full((len(individuals), len(bodyparts), 3), np.nan)

    def boxes_at(i):
        return boxes[i] if boxes is not None and i < len(boxes) else None

    def render_cv(i):
        frame = vid.read(i)
        return None if frame is None else draw_cv(frame, pose_at(i), bodyparts, colours, boxes_at(i), opts)

    if args.export:
        end = min(args.end if args.end is not None else vid.n - 1, vid.n - 1)
        first = render_cv(args.start)
        h, w = first.shape[:2]
        out = cv2.VideoWriter(args.export, cv2.VideoWriter_fourcc(*"mp4v"), vid.fps, (w, h))
        out.write(first)
        for i in range(args.start + 1, end + 1):
            img = render_cv(i)
            if img is None:
                break
            out.write(img)
        out.release()
        print(f"Saved {args.export} (frames {args.start} to {end})")
        return

    # ---------- interactive viewer (matplotlib) ----------
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.widgets import Slider

    # Keys like s, p, c and l also trigger matplotlib's own shortcuts; switch those off.
    for k in ("keymap.save", "keymap.pan", "keymap.back", "keymap.forward", "keymap.home",
              "keymap.fullscreen", "keymap.yscale", "keymap.xscale", "keymap.grid", "keymap.grid_minor"):
        plt.rcParams[k] = []

    state = {"i": max(0, min(args.start, vid.n - 1)), "playing": False}
    fig = plt.figure(figsize=(14, 7.5))
    ax = fig.add_axes([0.01, 0.1, 0.78, 0.84])
    ax.axis("off")
    leg = fig.add_axes([0.80, 0.1, 0.19, 0.84])
    leg.axis("off")
    leg.set_xlim(0, 1)
    leg.set_ylim(len(bodyparts) + 0.5, -0.8)

    frame0 = vid.read(state["i"])
    im = ax.imshow(cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB))
    n_pts = len(individuals) * len(bodyparts)
    pt_colours = np.tile(np.array(colours, dtype=object), len(individuals))
    pt_names = np.tile(np.array(bodyparts, dtype=object), len(individuals))
    scat = ax.scatter(np.zeros(n_pts), np.zeros(n_pts), s=(args.dot * 1.8) ** 2, linewidths=1.2, zorder=3)
    labels = [ax.text(0, 0, "", fontsize=7, color="white", zorder=4,
                      bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5, lw=0)) for _ in range(n_pts)]
    hover = ax.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points", fontsize=9, zorder=5,
                        bbox=dict(boxstyle="round", fc="white", alpha=0.9), visible=False)
    box_patches = []

    leg.text(0, -0.5, "keypoint", fontsize=8, weight="bold")
    leg.text(0.98, -0.5, "conf", fontsize=8, weight="bold", ha="right")
    leg_dots = leg.scatter(np.full(len(bodyparts), 0.04), np.arange(len(bodyparts)), s=40,
                           c=colours, linewidths=1.2, clip_on=False)
    leg_names = [leg.text(0.1, j, bp, fontsize=8, va="center") for j, bp in enumerate(bodyparts)]
    leg_conf = [leg.text(0.98, j, "", fontsize=8, va="center", ha="right", family="monospace")
                for j in range(len(bodyparts))]

    slider = Slider(fig.add_axes([0.08, 0.03, 0.7, 0.03]), "frame", 0, vid.n - 1, valinit=state["i"], valstep=1)
    current = {"xy": np.full((n_pts, 2), np.nan), "lik": np.full(n_pts, np.nan)}

    def show(i):
        i = int(max(0, min(i, vid.n - 1)))
        frame = vid.read(i)
        if frame is None:
            return
        state["i"] = i
        im.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        pose = pose_at(i).reshape(-1, 3)
        xy, lik = pose[:, :2], pose[:, 2]
        current["xy"], current["lik"] = xy, lik
        valid = ~np.isnan(xy).any(axis=1)
        confident = valid & (lik >= opts["pcutoff"])
        face = np.where(confident, pt_colours, "none")
        edge = np.where(confident, "black", GREY)
        drawn = valid & (confident | ~opts["hide_low"])
        scat.set_offsets(np.where(drawn[:, None], xy, np.nan))
        scat.set_facecolors(list(face))
        scat.set_edgecolors(list(edge))

        for k, t in enumerate(labels):
            parts = ([pt_names[k]] if opts["names"] else []) + ([f"{lik[k]:.2f}"] if opts["conf"] else [])
            t.set_visible(bool(parts) and drawn[k])
            if parts and drawn[k]:
                t.set_position((xy[k, 0] + 8, xy[k, 1] - 8))
                t.set_text(" ".join(parts))
                t.set_color("white" if confident[k] else GREY)

        for p in box_patches:
            p.remove()
        box_patches.clear()
        bx = boxes_at(i)
        if opts["boxes"] and bx is not None:
            for (x, y, bw, bh), sc in zip(*bx):
                box_patches.append(ax.add_patch(Rectangle((x, y), bw, bh, fill=False, ec="yellow", lw=1.2)))
                box_patches.append(ax.text(x, y - 6, "box" if np.isnan(sc) else f"box {sc:.2f}",
                                           color="yellow", fontsize=8, va="bottom"))

        # Legend: best confidence per body part across individuals
        per_bp = pose_at(i)[..., 2]
        best = np.full(len(bodyparts), np.nan)
        if not np.all(np.isnan(per_bp)):
            best = np.nanmax(np.where(np.isnan(per_bp), -np.inf, per_bp), axis=0)
            best[np.isinf(best)] = np.nan
        ok = ~np.isnan(best) & (best >= opts["pcutoff"])
        leg_dots.set_facecolors([c if o else "none" for c, o in zip(colours, ok)])
        leg_dots.set_edgecolors(["black" if o else GREY for o in ok])
        for j in range(len(bodyparts)):
            leg_conf[j].set_text("-" if np.isnan(best[j]) else f"{best[j]:.2f}")
            leg_names[j].set_color("black" if ok[j] else GREY)
            leg_conf[j].set_color("black" if ok[j] else GREY)
        leg.set_visible(opts["legend"])

        n_found = int(valid.sum())
        mean_conf = np.nanmean(lik) if n_found else float("nan")
        ax.set_title(f"frame {i} / {vid.n - 1}    {n_found} points    mean confidence {mean_conf:.2f}    "
                     f"(arrows step, space plays, hover for names, l/n/c/b/h toggle, p prints)", fontsize=9)
        if int(slider.val) != i:
            slider.eventson = False
            slider.set_val(i)
            slider.eventson = True
        hover.set_visible(False)
        fig.canvas.draw_idle()

    def on_move(e):
        if e.inaxes is not ax or e.x is None:
            if hover.get_visible():
                hover.set_visible(False)
                fig.canvas.draw_idle()
            return
        xy = current["xy"]
        valid = ~np.isnan(xy).any(axis=1)
        if not valid.any():
            return
        disp = ax.transData.transform(np.where(valid[:, None], xy, 0))
        d = np.hypot(disp[:, 0] - e.x, disp[:, 1] - e.y)
        d[~valid] = np.inf
        k = int(np.argmin(d))
        if d[k] < 12:
            ind = individuals[k // len(bodyparts)]
            who = f"{ind}: " if len(individuals) > 1 else ""
            hover.xy = tuple(xy[k])
            hover.set_text(f"{who}{pt_names[k]}  {current['lik'][k]:.2f}")
            hover.set_visible(True)
        else:
            hover.set_visible(False)
        fig.canvas.draw_idle()

    def print_frame():
        i = state["i"]
        print(f"\nframe {i}")
        for ind, pose in zip(individuals, pose_at(i)):
            if np.all(np.isnan(pose[:, 0])):
                continue
            print(f"  {ind}")
            for bp, (x, y, lik) in sorted(zip(bodyparts, pose), key=lambda t: -np.nan_to_num(t[1][2], nan=-1)):
                flag = "" if lik >= opts["pcutoff"] else "   <- below pcutoff"
                print(f"    {bp:<20} x={x:8.1f}  y={y:8.1f}  conf={lik:.3f}{flag}")
        bx = boxes_at(i)
        if bx is not None:
            for (x, y, bw, bh), sc in zip(*bx):
                print(f"  box x={x:.0f} y={y:.0f} w={bw:.0f} h={bh:.0f} score={sc:.3f}")

    steps = {"right": 1, "left": -1, "up": 10, "down": -10, "pageup": 100, "pagedown": -100}

    def on_key(e):
        if e.key in steps:
            show(state["i"] + steps[e.key])
        elif e.key == "home":
            show(0)
        elif e.key == "end":
            show(vid.n - 1)
        elif e.key == " ":
            state["playing"] = not state["playing"]
        elif e.key in ("n", "c", "b", "l", "h"):
            opts[{"n": "names", "c": "conf", "b": "boxes", "l": "legend", "h": "hide_low"}[e.key]] ^= True
            show(state["i"])
        elif e.key == "p":
            print_frame()
        elif e.key == "s":
            name = f"frame_{state['i']:05d}.png"
            cv2.imwrite(name, render_cv(state["i"]))
            vid.next_index = -1  # force a seek next time, since render_cv advanced the reader
            print(f"Saved {name}")

    def tick():
        if state["playing"]:
            if state["i"] >= vid.n - 1:
                state["playing"] = False
            else:
                show(state["i"] + 1)

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    slider.on_changed(lambda v: show(v))
    timer = fig.canvas.new_timer(interval=int(1000 / min(vid.fps, 30)))
    timer.add_callback(tick)
    timer.start()
    show(state["i"])
    plt.show()


if __name__ == "__main__":
    main()