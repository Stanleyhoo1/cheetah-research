"""
Track DeepLabCut keypoints through a video with Track-On.

For each keypoint, the script finds the frame where DeepLabCut was most confident about it
and starts tracking it there, both forwards and backwards in time. The result is saved in
DeepLabCut's .h5 format, so you can open it with inspect_pose.py.

Usage (in the trackon environment):
    python scripts/analysis/trackon_from_dlc.py VIDEO DLC.h5 --trackon ~/track_on --ckpt ~/track_on/track_on_r.pt

Useful options:
    --start / --end    frame range (default: every frame where DeepLabCut confidently sees the cheetah)
    --pcutoff 0.8      how confident DeepLabCut must be for a frame to be used as a starting point
    --no-crop          use the whole frame instead of a box around the cheetah
    --margin 150       pixels added around the cheetah when cropping
    --exclude lure     keypoints to skip (default: lure)
    --forward-only     skip the backward pass (faster, but points are only tracked after their start frame)

Why frames are resized: Track-On shrinks every frame to 512 x 384 inside the model anyway, so the
script does that resizing itself first (with a cleaner method). That keeps GPU memory small enough
for a whole run, and loses no detail compared with sending full-size frames.

Output columns: x, y in full-frame pixels, and "likelihood" = Track-On's visible (1) / hidden (0) flag.
Frames the tracker never reached are left empty.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

MODEL_W, MODEL_H = 512, 384  # Track-On's internal input size (width, height)


def load_dlc(path, exclude):
    df = pd.read_hdf(path)
    if "scorer" in df.columns.names:
        df = df.droplevel("scorer", axis=1)
    if "individuals" in df.columns.names:  # multi-animal file: use the first individual
        first = df.columns.get_level_values("individuals")[0]
        df = df.xs(first, level="individuals", axis=1)
    keep = [bp for bp in dict.fromkeys(df.columns.get_level_values("bodyparts")) if bp not in exclude]
    return df[keep], keep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("dlc_h5", help="DeepLabCut predictions for this video (e.g. a file from dlc_pw)")
    ap.add_argument("--trackon", required=True, help="folder where you cloned Track-On")
    ap.add_argument("--ckpt", required=True, help="Track-On checkpoint, e.g. track_on_r.pt")
    ap.add_argument("--start", type=int)
    ap.add_argument("--end", type=int, help="last frame, inclusive")
    ap.add_argument("--pcutoff", type=float, default=0.8)
    ap.add_argument("--no-crop", action="store_true")
    ap.add_argument("--margin", type=int, default=150)
    ap.add_argument("--exclude", nargs="*", default=["lure"])
    ap.add_argument("--forward-only", action="store_true")
    ap.add_argument("--out", help="output .h5 (default: <video name>_trackon.h5 next to this script's working folder)")
    args = ap.parse_args()

    dlc, bodyparts = load_dlc(args.dlc_h5, set(args.exclude))
    lik = dlc.xs("likelihood", level="coords", axis=1)
    xs_all = dlc.xs("x", level="coords", axis=1)
    ys_all = dlc.xs("y", level="coords", axis=1)

    # ---- 1. Frame range: where DeepLabCut confidently sees at least 3 keypoints ----
    seen = (lik >= args.pcutoff).sum(axis=1) >= 3
    if args.start is None or args.end is None:
        if not seen.any():
            sys.exit("DeepLabCut never confidently sees the cheetah in this file; try a lower --pcutoff.")
        frames_seen = seen[seen].index
    start = int(args.start if args.start is not None else frames_seen.min())
    end = int(args.end if args.end is not None else frames_seen.max())
    clip_lik = lik.loc[start:end]
    print(f"frames {start} to {end} ({end - start + 1} frames)")

    # ---- 2. One starting frame per keypoint: where DeepLabCut is most confident ----
    seeds = {}
    for bp in bodyparts:
        best_frame = clip_lik[bp].idxmax()
        if pd.notna(best_frame) and clip_lik[bp][best_frame] >= args.pcutoff:
            seeds[bp] = (int(best_frame), float(xs_all[bp][best_frame]), float(ys_all[bp][best_frame]))
    skipped = [bp for bp in bodyparts if bp not in seeds]
    if not seeds:
        sys.exit("No keypoint reaches the confidence cutoff in this range.")
    print("starting frames:", ", ".join(f"{bp} @ {s[0]}" for bp, s in seeds.items()))
    if skipped:
        print("not tracked (never confident enough):", ", ".join(skipped))

    # ---- 3. Crop box around everywhere the cheetah confidently appears ----
    cap = cv2.VideoCapture(args.video)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.no_crop:
        x0, y0, x1, y1 = 0, 0, W, H
    else:
        good = clip_lik >= args.pcutoff
        gx = xs_all.loc[start:end][good].to_numpy(float)
        gy = ys_all.loc[start:end][good].to_numpy(float)
        x0, x1 = max(int(np.nanmin(gx)) - args.margin, 0), min(int(np.nanmax(gx)) + args.margin, W)
        y0, y1 = max(int(np.nanmin(gy)) - args.margin, 0), min(int(np.nanmax(gy)) + args.margin, H)
    cw, ch = x1 - x0, y1 - y0
    sx, sy = MODEL_W / cw, MODEL_H / ch
    print(f"crop x {x0}-{x1}, y {y0}-{y1} ({cw} x {ch} px), resized to {MODEL_W} x {MODEL_H}")

    # ---- 4. Load the clip: crop, then resize to Track-On's size ----
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(end - start + 1):
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f[y0:y1, x0:x1], (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    T = len(frames)
    if T == 0:
        sys.exit("Could not read any frames in that range.")
    clip = np.stack(frames)  # (T, H, W, 3)

    import torch
    sys.path.insert(0, os.path.expanduser(args.trackon))
    from model.trackon_predictor import Predictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: no GPU found; this will be very slow.")
    model = Predictor(checkpoint_path=os.path.expanduser(args.ckpt)).to(device).eval()
    video = torch.from_numpy(clip).permute(0, 3, 1, 2).float()[None].to(device)  # (1, T, 3, H, W)

    names = list(seeds)
    t_seed = np.array([seeds[bp][0] - start for bp in names])
    q_xy = np.array([[(seeds[bp][1] - x0) * sx, (seeds[bp][2] - y0) * sy] for bp in names])

    def run(vid, times):
        q = torch.tensor(np.c_[times, q_xy], dtype=torch.float32)[None].to(device)
        with torch.no_grad():
            tr, vis = model(vid, q)
        return tr[0].cpu().numpy(), vis[0].cpu().numpy()  # (T, N, 2), (T, N)

    # ---- 5. Track forwards from each start frame, and backwards on the reversed clip ----
    xy_f, vis_f = run(video, t_seed)
    xy = np.full((T, len(names), 2), np.nan)
    vis = np.full((T, len(names)), np.nan)
    after = np.arange(T)[:, None] >= t_seed[None, :]
    xy[after] = xy_f[after]
    vis[after] = vis_f[after]
    if not args.forward_only:
        xy_b, vis_b = run(torch.flip(video, dims=[1]), (T - 1) - t_seed)
        xy_b, vis_b = xy_b[::-1], vis_b[::-1]  # back into normal time order
        before = ~after
        xy[before] = xy_b[before]
        vis[before] = vis_b[before]

    # ---- 6. Back to full-frame pixels and save in DeepLabCut's format ----
    xy[..., 0] = xy[..., 0] / sx + x0
    xy[..., 1] = xy[..., 1] / sy + y0
    cols = pd.MultiIndex.from_product([["TrackOn"], names, ["x", "y", "likelihood"]],
                                      names=["scorer", "bodyparts", "coords"])
    data = np.concatenate([xy, vis[..., None]], axis=2).reshape(T, -1)
    out = args.out or os.path.splitext(os.path.basename(args.video))[0] + "_trackon.h5"
    pd.DataFrame(data, columns=cols, index=range(start, start + T)).to_hdf(out, key="df", mode="w")
    print(f"Saved {out}: {len(names)} keypoints over frames {start} to {start + T - 1}")


if __name__ == "__main__":
    main()