"""
Take evenly spaced pictures from one short moment of a run, from every camera,
and add them to your DeepLabCut project ready for labelling.

All cameras get the SAME frame numbers, so each moment is seen from every angle.
That lets you check labels across cameras later.

Usage (dlc environment):
    python scripts/labeling/extract_sequence.py RUN_FOLDER NAME --start SECONDS            # 25 frames in a row
    python scripts/labeling/extract_sequence.py RUN_FOLDER NAME --start SECONDS --end SECONDS   # 25 frames spread out

Examples:
    25 frames in a row starting at 2.0 s:
        python scripts/labeling/extract_sequence.py data/julesBottomFlick21_20171221 jules_flick21_turn --start 2.0
    25 frames spread evenly from 2.0 s to 3.5 s:
        python scripts/labeling/extract_sequence.py data/julesBottomFlick21_20171221 jules_flick21_turn --start 2.0 --end 3.5

Options:
    --frames 25           pictures per camera (Shengyang suggests 20 to 30)
    --start-frame/--end-frame   use frame numbers instead of seconds
    --cams 1 2 3          only these cameras (default: every camN.mp4 in the folder)
    --config PATH         project settings file (default: cheetah-dlc/config.yaml in this repo)

Each camera's pictures go into cheetah-dlc/labeled-data/NAME_camN/, named by
frame number (img00240.png is frame 240), so the same name means the same moment.
"""
import argparse
import math
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_folder", help="folder with cam1.mp4 ... cam6.mp4")
    ap.add_argument("name", help="short unique name for this moment, e.g. jules_20171221_flick21_turn")
    ap.add_argument("--start", type=float, help="start time in seconds")
    ap.add_argument("--end", type=float, help="end time in seconds")
    ap.add_argument("--start-frame", type=int)
    ap.add_argument("--end-frame", type=int)
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--cams", type=int, nargs="*")
    ap.add_argument("--config", default=str(REPO / "cheetah-dlc" / "config.yaml"))
    args = ap.parse_args()

    from deeplabcut.utils.auxiliaryfunctions import read_config, write_config

    config = Path(args.config).expanduser().absolute()
    project = config.parent
    run = Path(args.run_folder).expanduser().absolute()
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", args.name):
        sys.exit("Use only letters, numbers, _ and - in NAME.")

    videos = {}
    for v in sorted(run.glob("cam*.mp4")):
        m = re.fullmatch(r"cam(\d+)\.mp4", v.name)
        if m and (not args.cams or int(m.group(1)) in args.cams):
            videos[int(m.group(1))] = v
    if not videos:
        sys.exit(f"No camN.mp4 files found in {run}")

    # Work out the frame numbers once, from the first camera, and use them for every camera.
    first = cv2.VideoCapture(str(next(iter(videos.values()))))
    fps = first.get(cv2.CAP_PROP_FPS)
    total = int(first.get(cv2.CAP_PROP_FRAME_COUNT))
    first.release()
    if args.start_frame is not None:
        f0 = args.start_frame
    elif args.start is not None:
        f0 = round(args.start * fps)
    else:
        sys.exit("Give --start (seconds) or --start-frame.")
    if args.end_frame is not None:
        f1 = args.end_frame
    elif args.end is not None:
        f1 = round(args.end * fps)
    else:
        f1 = f0 + args.frames - 1  # no end given: that many frames in a row
    f1 = min(f1, total - 1)
    if not 0 <= f0 < f1:
        sys.exit(f"Bad range: frames {f0} to {f1} (the video has {total} frames at {fps:.0f} fps).")
    wanted = sorted(set(np.linspace(f0, f1, args.frames).round().astype(int).tolist()))
    digits = max(1, math.ceil(math.log10(total + 1)))
    print(f"{fps:.0f} fps, frames {f0} to {f1} ({(f1 - f0) / fps:.2f} s), {len(wanted)} pictures per camera")

    cfg = read_config(str(config))
    video_sets = dict(cfg.get("video_sets") or {})
    (project / "videos").mkdir(exist_ok=True)

    for cam, video in videos.items():
        folder_name = f"{args.name}_cam{cam}"
        out_dir = project / "labeled-data" / folder_name
        if out_dir.exists() and any(out_dir.glob("*.png")):
            print(f"{folder_name}: already exists, skipped (delete the folder to redo it)")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        # A link to the video under the folder's name, so DeepLabCut can match them later.
        link = project / "videos" / f"{folder_name}.mp4"
        if not link.exists():
            link.symlink_to(video)

        cap = cv2.VideoCapture(str(video))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n != total:
            print(f"  note: cam{cam} has {n} frames, cam{next(iter(videos))} has {total}; check they are in sync")
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        targets, saved, i = set(wanted), 0, 0
        while i <= wanted[-1]:  # read in order from the start: slower than jumping, but always the exact frame
            ok, frame = cap.read()
            if not ok:
                break
            if i in targets:
                cv2.imwrite(str(out_dir / f"img{str(i).zfill(digits)}.png"), frame)
                saved += 1
            i += 1
        cap.release()

        video_sets[str(link)] = {"crop": f"0, {w}, 0, {h}"}
        print(f"{folder_name}: {saved} pictures saved")

    cfg["video_sets"] = video_sets
    write_config(str(config), cfg)
    print("\nLabel each folder with:")
    for cam in videos:
        print(f"  python scripts/labeling/label_frames.py {config} {args.name}_cam{cam}")
    print("\nThen record the moment in sequences.csv (so its pictures can be rebuilt elsewhere):")
    print("  python scripts/labeling/update_sequences.py")


if __name__ == "__main__":
    main()