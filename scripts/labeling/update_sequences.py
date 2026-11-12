"""
Write sequences.csv: one row per labelled moment, worked out from the folders in
cheetah-dlc/labeled-data and the video links in cheetah-dlc/videos.

extract_sequence.py names each folder NAME_camN and each picture imgNNN.png by its frame
number, and links cheetah-dlc/videos/NAME_camN.mp4 to the run's camN.mp4. That is enough to
rebuild every picture from the videos later (rebuild_frames.py), so the pictures themselves
do not need to be in git.

Columns:
    name          the NAME given to extract_sequence.py
    run           folder in data/ holding cam1.mp4 ... cam6.mp4
    start_frame   first frame (inclusive)
    end_frame     last frame (inclusive)
    cameras       camera numbers, space separated

A moment is listed once at least one of its cameras has saved labels (CollectedData_*.h5).
Rows already in sequences.csv that can't be worked out from this machine's folders (for
example on a machine without the video links) are kept as they are.

Run it on the laptop after extracting a new moment, then commit sequences.csv.

Usage (any Python 3):
    python scripts/labeling/update_sequences.py
    python scripts/labeling/update_sequences.py --config cheetah-dlc/config.yaml --out sequences.csv
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COLUMNS = ["name", "run", "start_frame", "end_frame", "cameras"]


def read_sequences(path):
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {row["name"]: row for row in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "cheetah-dlc" / "config.yaml"),
                    help="project settings file; only its folder is used (default: this repo's)")
    ap.add_argument("--out", default=str(REPO / "sequences.csv"), help="file to write (default: sequences.csv in this repo)")
    args = ap.parse_args()

    project = Path(args.config).expanduser().absolute().parent
    labeled = project / "labeled-data"
    out = Path(args.out).expanduser().absolute()
    if not labeled.is_dir():
        sys.exit(f"No labeled-data folder in {project}")

    # Collect each camera folder of each moment.
    moments = defaultdict(dict)  # name -> {cam: {"frames": [...], "run": str|None, "labelled": bool}}
    for folder in sorted(p for p in labeled.iterdir() if p.is_dir()):
        if folder.name.endswith("_labeled"):  # check_labels output
            continue
        m = re.fullmatch(r"(.+)_cam(\d+)", folder.name)
        if not m:
            print(f"{folder.name}: not named NAME_camN, skipped")
            continue
        frames = sorted(int(p.stem[3:]) for p in folder.glob("img*.png") if re.fullmatch(r"img\d+", p.stem))
        if not frames:
            print(f"{folder.name}: no imgNNN.png pictures, skipped")
            continue
        run = None
        link = project / "videos" / f"{folder.name}.mp4"
        if link.is_symlink():
            target = Path(os.readlink(link))
            run = target.parent.name
            if target.name != f"cam{m.group(2)}.mp4":
                print(f"  note: {link.name} points to {target.name}, not cam{m.group(2)}.mp4")
        moments[m.group(1)][int(m.group(2))] = {
            "frames": frames, "run": run,
            "labelled": any(folder.glob("CollectedData_*.h5")),
        }

    old_rows = read_sequences(out)
    rows = {}
    for name, cams in sorted(moments.items()):
        if not any(c["labelled"] for c in cams.values()):
            print(f"{name}: no camera has saved labels yet, not listed")
            continue
        runs = {c["run"] for c in cams.values()}
        if None in runs or len(runs) != 1:
            if name in old_rows:
                print(f"{name}: video links missing or disagree ({runs}), kept the existing row")
                rows[name] = old_rows[name]
            else:
                print(f"{name}: video links missing or disagree ({runs}), not listed")
            continue
        first = min(c["frames"][0] for c in cams.values())
        last = max(c["frames"][-1] for c in cams.values())
        for cam, c in sorted(cams.items()):
            if (c["frames"][0], c["frames"][-1]) != (first, last) or len(c["frames"]) != last - first + 1:
                print(f"  note: {name}_cam{cam} has frames {c['frames'][0]}-{c['frames'][-1]} "
                      f"({len(c['frames'])} pictures); the row covers {first}-{last}")
        rows[name] = {"name": name, "run": runs.pop(), "start_frame": first, "end_frame": last,
                      "cameras": " ".join(str(cam) for cam in sorted(cams))}

    for name, row in old_rows.items():
        if name not in rows and name not in moments:
            print(f"{name}: in {out.name} but no folders here, kept the existing row")
            rows[name] = row

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for name in sorted(rows):
            w.writerow({k: rows[name][k] for k in COLUMNS})
    print(f"\nWrote {len(rows)} moment(s) to {out}")


if __name__ == "__main__":
    main()
