"""
Run your trained DeepLabCut model on videos, and use Track-On wherever DeepLabCut loses a point.

DeepLabCut is trusted whenever it is confident. When a keypoint's confidence drops below the
cutoff (it lost the point, or is only guessing), Track-On fills that stretch:
  - it starts from DeepLabCut's last confident position before the stretch and tracks forwards,
  - and from the first confident position after it and tracks backwards,
  - then the two are blended (more weight on whichever end is closer).
Track-On only ever tracks across the short stretch it fills, so it has little time to drift.

Run from the repository root.

Usage, if one conda env has both DeepLabCut and Track-On's packages:
    python scripts/analysis/dlc_trackon.py cheetah-dlc/config.yaml data/zorroTopFlick1_2017_12_09/cam*.mp4 \\
        --shuffle 10 --trackon ~/track_on --ckpt path/to/track_on_r.pt

If they are in separate envs (the usual "dlc" and "trackon" setup), do it in two steps:
    1. (dlc env)     python -c "import deeplabcut; deeplabcut.analyze_videos('cheetah-dlc/config.yaml', ['VIDEO'], shuffle=10, destfolder='results/RUN_FOLDER_NAME')"
    2. (trackon env) python scripts/analysis/dlc_trackon.py cheetah-dlc/config.yaml VIDEO --dlc-h5 results/RUN_FOLDER_NAME/camN...h5 --trackon ~/track_on --ckpt path/to/track_on_r.pt

Options:
    --cutoff 0.6     below this DeepLabCut confidence a point counts as lost (default: pcutoff in config.yaml)
    --max-gap 60     don't fill stretches longer than this many frames (the point is probably really gone)
    --min-good 3     a confident stretch shorter than this is not trusted as a starting point
    --margin 150     pixels kept around the cheetah when cropping for Track-On
    --no-crop        give Track-On the whole frame instead of a box around the cheetah
    --out-dir results
    --no-video       skip the video with the points drawn on

Outputs go in --out-dir/RUN_FOLDER_NAME/, per video:
    NAME_dlc_trackon.h5   DeepLabCut format (x, y, likelihood). Points filled by Track-On get
                          likelihood = the cutoff, so they count as "just good enough".
    NAME_source.csv       for every frame and keypoint: dlc, trackon, or low (DeepLabCut was unsure
                          and Track-On could not fill it, so DeepLabCut's guess is kept)
    NAME_dlc_trackon.mp4  the video with points drawn: solid dot = DeepLabCut, ring = Track-On.
                          Unsure points that weren't filled are not drawn.

Works without a screen, so it can run over SSH.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

MODEL_W, MODEL_H = 512, 384  # Track-On's internal input size (width, height)


# ---------------------------------------------------------------- DeepLabCut part

def run_deeplabcut(config, video, shuffle, out_dir):
    """Run the trained model on one video and return the path of its .h5 result."""
    try:
        import deeplabcut
    except ImportError:
        sys.exit("DeepLabCut isn't installed in this env. Run step 1 from the usage notes in the dlc env, "
                 "then pass its result here with --dlc-h5.")
    scorer = deeplabcut.analyze_videos(str(config), [str(video)], videotype=video.suffix, shuffle=shuffle,
                                       destfolder=str(out_dir), save_as_csv=False)
    if isinstance(scorer, str) and (out_dir / f"{video.stem}{scorer}.h5").exists():
        return out_dir / f"{video.stem}{scorer}.h5"
    found = sorted(out_dir.glob(f"{video.stem}DLC*.h5"), key=os.path.getmtime)
    if not found:
        sys.exit(f"DeepLabCut finished but no result file for {video.name} was found in {out_dir}")
    return found[-1]


def load_dlc(path):
    df = pd.read_hdf(path)
    if "scorer" in df.columns.names:
        df = df.droplevel("scorer", axis=1)
    if "individuals" in df.columns.names:  # multi-animal file: use the first individual
        first = df.columns.get_level_values("individuals")[0]
        df = df.xs(first, level="individuals", axis=1)
    bodyparts = list(dict.fromkeys(df.columns.get_level_values("bodyparts")))
    get = lambda c: df.xs(c, level="coords", axis=1)[bodyparts].to_numpy(float)
    return bodyparts, get("x"), get("y"), get("likelihood")


# ---------------------------------------------------------------- finding what to fill

def runs(mask):
    """Start and end (inclusive) of each stretch of True values."""
    m = np.r_[False, mask, False].astype(int)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def find_gaps(lik, start, end, cutoff, min_good, max_gap):
    """For each keypoint, the stretches inside [start, end] where DeepLabCut lost it.

    Returns a list of (keypoint index, first frame, last frame, frame to track forwards from or None,
    frame to track backwards from or None). Frames are video frame numbers.
    """
    gaps = []
    for k in range(lik.shape[1]):
        good = lik[start:end + 1, k] >= cutoff
        for a, b in runs(good):  # don't trust very short confident stretches as starting points
            if b - a + 1 < min_good:
                good[a:b + 1] = False
        for a, b in runs(~good):
            if b - a + 1 > max_gap:
                continue
            before = start + a - 1 if a > 0 else None
            after = start + b + 1 if b < len(good) - 1 else None
            if before is None and after is None:
                continue  # never confident in this range: nothing to start from
            gaps.append((k, start + a, start + b, before, after))
    return gaps


# ---------------------------------------------------------------- Track-On part

def track(model, frames, queries, device):
    """Track points through frames, starting each one at its own frame.

    Same as Track-On's Predictor.forward, except frames go to the GPU one at a time,
    so long clips fit in memory. frames: (T, H, W, 3) uint8 RGB. queries: (N, 3) of (t, x, y).
    Returns positions (T, N, 2) and Track-On's visible flags (T, N).
    """
    import torch
    from utils.coord_utils import get_points_on_a_grid

    T, H, W, _ = frames.shape
    q = torch.as_tensor(np.asarray(queries, dtype=np.float32), device=device)
    n = q.shape[0]
    if model.support_grid_size > 0:  # extra background points Track-On uses to help, as in Predictor.forward
        grid = get_points_on_a_grid(model.support_grid_size, (H, W), device).squeeze(0)
        q = torch.cat([q, torch.cat([torch.zeros(len(grid), 1, device=device), grid], dim=1)])
    times, coords = q[:, 0].long(), q[:, 1:]

    model.reset()
    model.initial_capacity = q.shape[0]
    xy = np.full((T, n, 2), np.nan, dtype=np.float32)
    vis = np.zeros((T, n), dtype=bool)
    order = []  # which query each of Track-On's internal slots belongs to
    with torch.no_grad():
        for t in range(T):
            new = times == t
            new_q = None
            if new.any():
                new_q = coords[new]
                order.extend(new.nonzero(as_tuple=True)[0].tolist())
            if model.N == 0 and new_q is None:
                continue
            frame = torch.from_numpy(frames[t]).to(device).permute(2, 0, 1).float()[None]
            p, v = model.forward_frame(frame, new_queries=new_q)
            idx = np.array(order)
            mine = idx < n
            xy[t, idx[mine]] = p.cpu().numpy()[mine]
            vis[t, idx[mine]] = v.cpu().numpy()[mine]
    return xy, vis


def read_frames(video, start, end, box):
    """Frames start..end, cropped to box and resized to Track-On's size. Reads in order from the
    first frame, like DeepLabCut does, so frame numbers always match its results."""
    x0, y0, x1, y1 = box
    cap = cv2.VideoCapture(str(video))
    frames, i = [], 0
    while i <= end:
        ok, f = cap.read()
        if not ok:
            break
        if i >= start:
            f = cv2.resize(f[y0:y1, x0:x1], (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return np.stack(frames) if frames else None


def fill_gaps(gaps, xs, ys, start, frames, box, model, device):
    """Run Track-On for every gap and return {(keypoint, frame): (x, y)} in full-frame pixels."""
    x0, y0, x1, y1 = box
    sx, sy = MODEL_W / (x1 - x0), MODEL_H / (y1 - y0)
    T = len(frames)
    to_model = lambda f, k: ((xs[f, k] - x0) * sx, (ys[f, k] - y0) * sy)

    fwd = [(g, (g[3] - start, *to_model(g[3], g[0]))) for g in gaps if g[3] is not None]
    bwd = [(g, ((T - 1) - (g[4] - start), *to_model(g[4], g[0]))) for g in gaps if g[4] is not None]

    tracks = {}  # (gap, "f" or "b") -> positions over the clip, in normal time order
    if fwd:
        xy, _ = track(model, frames, [q for _, q in fwd], device)
        for j, (g, _) in enumerate(fwd):
            tracks[(g, "f")] = xy[:, j]
    if bwd:
        xy, _ = track(model, frames[::-1].copy(), [q for _, q in bwd], device)
        xy = xy[::-1]
        for j, (g, _) in enumerate(bwd):
            tracks[(g, "b")] = xy[:, j]

    filled = {}
    for g in gaps:
        k, a, b = g[0], g[1], g[2]
        f, bk = tracks.get((g, "f")), tracks.get((g, "b"))
        for frame in range(a, b + 1):
            t = frame - start
            if f is not None and bk is not None:
                w = (frame - a + 1) / (b - a + 2)  # 0 near the start of the gap, 1 near the end
                p = (1 - w) * f[t] + w * bk[t]
            else:
                p = f[t] if f is not None else bk[t]
            if np.all(np.isfinite(p)):
                filled[(k, frame)] = (p[0] / sx + x0, p[1] / sy + y0)
    return filled


# ---------------------------------------------------------------- saving

def draw_video(video, out_path, xs, ys, source, bodyparts):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    colours = [tuple(int(c) for c in cv2.cvtColor(np.uint8([[[int(180 * i / len(bodyparts)), 255, 255]]]),
                                                  cv2.COLOR_HSV2BGR)[0, 0]) for i in range(len(bodyparts))]
    r = max(3, H // 150)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or i >= len(source):
            break
        for k in range(len(bodyparts)):
            if source[i, k] == "low" or not np.isfinite(xs[i, k]):
                continue
            c = (int(xs[i, k]), int(ys[i, k]))
            if source[i, k] == "dlc":
                cv2.circle(frame, c, r, colours[k], -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, c, r + 2, colours[k], max(2, r // 2), cv2.LINE_AA)
        writer.write(frame)
        i += 1
    cap.release()
    writer.release()


def read_pcutoff(config):
    try:
        from deeplabcut.utils.auxiliaryfunctions import read_config
        return float(read_config(str(config)).get("pcutoff", 0.6))
    except Exception:
        import re
        m = re.search(r"^pcutoff:\s*([0-9.]+)", open(config).read(), re.M)
        return float(m.group(1)) if m else 0.6


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="cheetah-dlc/config.yaml")
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--shuffle", type=int, default=1, help="which trained model (the --shuffle you trained with)")
    ap.add_argument("--dlc-h5", help="use this DeepLabCut result instead of running the model (one video only)")
    ap.add_argument("--trackon", required=True, help="folder where you cloned Track-On")
    ap.add_argument("--ckpt", required=True, help="Track-On checkpoint file")
    ap.add_argument("--cutoff", type=float, help="default: pcutoff from config.yaml")
    ap.add_argument("--max-gap", type=int, default=60)
    ap.add_argument("--min-good", type=int, default=3)
    ap.add_argument("--margin", type=int, default=150)
    ap.add_argument("--no-crop", action="store_true")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    config = Path(args.config).expanduser().absolute()
    out_dir = Path(args.out_dir).expanduser().absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = [Path(v).expanduser().absolute() for v in args.videos]
    if args.dlc_h5 and len(videos) > 1:
        sys.exit("--dlc-h5 works with one video at a time.")
    cutoff = args.cutoff if args.cutoff is not None else read_pcutoff(config)
    print(f"A point counts as lost below confidence {cutoff}")

    model = device = None
    for video in videos:
        print(f"\n=== {video.parent.name}/{video.name} ===")
        run_dir = out_dir / video.parent.name  # one folder per run: every run has a cam1.mp4
        run_dir.mkdir(parents=True, exist_ok=True)
        h5 = Path(args.dlc_h5).expanduser() if args.dlc_h5 else run_deeplabcut(config, video, args.shuffle, run_dir)
        bodyparts, xs, ys, lik = load_dlc(h5)
        n_frames = len(lik)
        source = np.where(lik >= cutoff, "dlc", "low").astype(object)

        # Frames where the cheetah is in view: DeepLabCut confidently sees at least 3 keypoints.
        seen = np.flatnonzero((lik >= cutoff).sum(axis=1) >= 3)
        gaps = []
        if len(seen):
            start, end = int(seen[0]), int(seen[-1])
            gaps = find_gaps(lik, start, end, cutoff, args.min_good, args.max_gap)
        if not gaps:
            print("Nothing to fill" + ("" if len(seen) else " (DeepLabCut never confidently sees the cheetah)"))
        else:
            print(f"Cheetah in view in frames {start} to {end}; {len(gaps)} stretches to fill")
            cap = cv2.VideoCapture(str(video))
            W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if args.no_crop:
                box = (0, 0, W, H)
            else:
                good = lik[start:end + 1] >= cutoff
                gx, gy = xs[start:end + 1][good], ys[start:end + 1][good]
                box = (max(int(gx.min()) - args.margin, 0), max(int(gy.min()) - args.margin, 0),
                       min(int(gx.max()) + args.margin, W), min(int(gy.max()) + args.margin, H))

            if model is None:
                import torch
                sys.path.insert(0, str(Path(args.trackon).expanduser()))
                from model.trackon_predictor import Predictor
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if device == "cpu":
                    print("Warning: no GPU found; Track-On will be very slow.")
                model = Predictor(checkpoint_path=str(Path(args.ckpt).expanduser())).to(device).eval()

            frames = read_frames(video, start, end, box)
            if frames is None:
                sys.exit(f"Could not read frames {start} to {end} from {video}")
            filled = fill_gaps(gaps, xs, ys, start, frames, box, model, device)
            for (k, f), (x, y) in filled.items():
                xs[f, k], ys[f, k] = x, y
                lik[f, k] = cutoff
                source[f, k] = "trackon"

        # Save in DeepLabCut's layout.
        cols = pd.MultiIndex.from_product([["DLC+TrackOn"], bodyparts, ["x", "y", "likelihood"]],
                                          names=["scorer", "bodyparts", "coords"])
        data = np.stack([xs, ys, lik], axis=2).reshape(n_frames, -1)
        out_h5 = run_dir / f"{video.stem}_dlc_trackon.h5"
        pd.DataFrame(data, columns=cols).to_hdf(out_h5, key="df_with_missing", mode="w")
        pd.DataFrame(source, columns=bodyparts).to_csv(run_dir / f"{video.stem}_source.csv", index_label="frame")

        counts = {s: int((source == s).sum()) for s in ("dlc", "trackon", "low")}
        print(f"points from DeepLabCut: {counts['dlc']}, filled by Track-On: {counts['trackon']}, "
              f"still unsure: {counts['low']}")
        per_kp = (source == "trackon").sum(axis=0)
        if per_kp.any():
            print("filled per keypoint: " + ", ".join(f"{bp} {c}" for bp, c in zip(bodyparts, per_kp) if c))
        if not args.no_video:
            draw_video(video, run_dir / f"{video.stem}_dlc_trackon.mp4", xs, ys, source, bodyparts)
        print(f"saved in {run_dir}: {out_h5.name}, {video.stem}_source.csv"
              + ("" if args.no_video else f", {video.stem}_dlc_trackon.mp4"))


if __name__ == "__main__":
    main()