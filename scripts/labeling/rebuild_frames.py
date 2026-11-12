"""
Rebuild the labelled pictures (imgNNN.png) from the videos, using sequences.csv.

The pictures are not in git: they are large, and each one can be re-made because its file
name is its frame number. Use this on a new machine after cloning the repo and copying the
videos into data/ (see README: Lab machine).

For every moment in sequences.csv and every camera it lists, this reads data/RUN/camN.mp4 and
writes frames start_frame..end_frame to cheetah-dlc/labeled-data/NAME_camN/imgNNN.png, the
same way extract_sequence.py made them. Pictures that already exist are left alone, so it is
safe to run again. Afterwards it checks that every picture named in each folder's labels
(CollectedData_*.h5) exists.

Usage (dlc environment):
    python scripts/labeling/rebuild_frames.py
    python scripts/labeling/rebuild_frames.py --only phantom_turn1 lily_brake1
    python scripts/labeling/rebuild_frames.py --data /path/to/videos
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[2]


def save_frames(video, wanted, out_dir, digits):
    """Write the wanted frame numbers of video as imgNNN.png. Same method as extract_sequence.py."""
    cap = cv2.VideoCapture(str(video))
    targets, saved, i = set(wanted), 0, 0
    while i <= max(wanted):  # read in order from the start: slower than jumping, but always the exact frame
        ok, frame = cap.read()
        if not ok:
            break
        if i in targets:
            cv2.imwrite(str(out_dir / f"img{str(i).zfill(digits)}.png"), frame)
            saved += 1
        i += 1
    cap.release()
    return saved


def labelled_images(folder):
    """Picture names listed in the folder's CollectedData_*.h5 files."""
    import pandas as pd

    names = set()
    for h5 in folder.glob("CollectedData_*.h5"):
        for idx in pd.read_hdf(h5).index:
            names.add(idx[-1] if isinstance(idx, tuple) else str(idx).replace("\\", "/").split("/")[-1])
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequences", default=str(REPO / "sequences.csv"), help="default: sequences.csv in this repo")
    ap.add_argument("--data", default=str(REPO / "data"), help="folder holding the run folders (default: data/ in this repo)")
    ap.add_argument("--config", default=str(REPO / "cheetah-dlc" / "config.yaml"),
                    help="project settings file; only its folder is used (default: this repo's)")
    ap.add_argument("--only", nargs="+", metavar="NAME", help="rebuild only these moments")
    args = ap.parse_args()

    data = Path(args.data).expanduser().absolute()
    labeled = Path(args.config).expanduser().absolute().parent / "labeled-data"
    with open(Path(args.sequences).expanduser(), newline="") as f:
        rows = list(csv.DictReader(f))
    if args.only:
        unknown = set(args.only) - {r["name"] for r in rows}
        if unknown:
            sys.exit(f"Not in {args.sequences}: {sorted(unknown)}")
        rows = [r for r in rows if r["name"] in args.only]

    problems = 0
    for row in rows:
        name, run = row["name"], row["run"]
        wanted = list(range(int(row["start_frame"]), int(row["end_frame"]) + 1))
        cams = [int(c) for c in row["cameras"].split()]
        videos = {cam: data / run / f"cam{cam}.mp4" for cam in cams}
        missing = [str(v) for v in videos.values() if not v.is_file()]
        if missing:
            print(f"{name}: video(s) not found, skipped: {', '.join(missing)}")
            problems += 1
            continue

        # File names are padded to the digit count of the first camera's frame count, as in extract_sequence.py.
        cap = cv2.VideoCapture(str(videos[cams[0]]))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        digits = max(1, math.ceil(math.log10(total + 1)))

        for cam, video in videos.items():
            out_dir = labeled / f"{name}_cam{cam}"
            out_dir.mkdir(parents=True, exist_ok=True)
            todo = [i for i in wanted if not (out_dir / f"img{str(i).zfill(digits)}.png").exists()]
            saved = save_frames(video, todo, out_dir, digits) if todo else 0
            note = f" (only {saved} of {len(todo)} could be read)" if saved < len(todo) else ""
            print(f"{out_dir.name}: {saved} written, {len(wanted) - len(todo)} already there{note}")
            if note:
                problems += 1

            absent = sorted(n for n in labelled_images(out_dir) if not (out_dir / n).exists())
            if absent:
                print(f"  {len(absent)} labelled picture(s) still missing, e.g. {absent[:3]}")
                problems += 1

    print("\nDone." if not problems else f"\nFinished with {problems} problem(s); see above.")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
