"""
Turn downloaded AcinoSet labelled folders into a DeepLabCut project.

Each AcinoSet folder holds pictures (img*.png) plus CollectedData_*.h5 / .csv,
the clicks for those pictures. AcinoSet gives no config.yaml, so this script
makes a new project, copies the folders in, and keeps only Shengyang's 15 points.
Your downloaded folders are left untouched.

Usage (in the dlc environment):
    python scripts/labeling/make_project.py cheetah_labels cheetah-dlc

Run it again after downloading more folders: it adds the new ones and skips
folders already in the project (so fixes you made there are kept).
"""
import argparse
import os
import shutil
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from PIL import Image

KEEP = ["neck_base", "spine", "l_shoulder", "r_shoulder",
        "l_hip", "r_hip", "tail_base", "tail_mid", "tail_tip",
        "l_back_paw", "r_back_paw",
        "l_back_knee", "r_back_knee", "l_back_ankle", "r_back_ankle"]

# AcinoSet's files call the tail points tail1 and tail2; we use Shengyang's names.
FROM_ACINOSET = {"tail1": "tail_mid", "tail2": "tail_tip"}

SKELETON = [["neck_base", "spine"], ["spine", "tail_base"],
            ["tail_base", "tail_mid"], ["tail_mid", "tail_tip"],
            ["neck_base", "l_shoulder"], ["neck_base", "r_shoulder"],
            ["tail_base", "l_hip"], ["tail_base", "r_hip"],
            ["l_hip", "l_back_knee"], ["l_back_knee", "l_back_ankle"], ["l_back_ankle", "l_back_paw"],
            ["r_hip", "r_back_knee"], ["r_back_knee", "r_back_ankle"], ["r_back_ankle", "r_back_paw"]]


def load_labels(folder):
    """Read a folder's clicks (prefer the .h5, fall back to the .csv)."""
    h5s = sorted(folder.glob("CollectedData_*.h5"))
    if h5s:
        return pd.read_hdf(h5s[0])
    csvs = sorted(folder.glob("CollectedData_*.csv"))
    if csvs:
        return pd.read_csv(csvs[0], header=[0, 1, 2], index_col=0)
    return None


def tidy(df, folder_name, scorer):
    """Same layout DeepLabCut writes: rows (labeled-data, folder, image), only our 15 points."""
    images = [str(i).replace("\\", "/").split("/")[-1] if not isinstance(i, tuple) else i[-1] for i in df.index]
    df = df.copy()
    df.index = pd.MultiIndex.from_tuples([("labeled-data", folder_name, im) for im in images])
    df = df.droplevel("scorer", axis=1) if "scorer" in df.columns.names else df
    if "individuals" in df.columns.names:
        df = df.droplevel("individuals", axis=1)
    df = df.rename(columns=FROM_ACINOSET, level="bodyparts")
    found = set(df.columns.get_level_values("bodyparts"))
    missing = [bp for bp in KEEP if bp not in found]
    cols = pd.MultiIndex.from_product([[scorer], KEEP, ["x", "y"]], names=["scorer", "bodyparts", "coords"])
    out = pd.DataFrame(index=df.index, columns=cols, dtype=float)
    for bp in KEEP:
        if bp in found:
            out[(scorer, bp, "x")] = pd.to_numeric(df[(bp, "x")], errors="coerce").values
            out[(scorer, bp, "y")] = pd.to_numeric(df[(bp, "y")], errors="coerce").values
    return out, missing


def link_or_copy(src, dst):
    """Hard link when possible (no extra disk space), otherwise copy."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", help="folder containing the downloaded AcinoSet folders")
    ap.add_argument("project", help="folder to create the DeepLabCut project in")
    ap.add_argument("--scorer", default="UCT", help="labeller name used in the files (default UCT)")
    args = ap.parse_args()

    from deeplabcut.utils.auxiliaryfunctions import create_config_template, read_config, write_config

    src_root = Path(args.labels).expanduser().absolute()
    project = Path(args.project).expanduser().absolute()
    config = project / "config.yaml"
    for sub in ("labeled-data", "videos", "training-datasets"):
        (project / sub).mkdir(parents=True, exist_ok=True)

    if config.exists():
        cfg = read_config(str(config))
    else:
        cfg, _ = create_config_template()
        cfg.update({
            "Task": "cheetah", "scorer": args.scorer, "date": date.today().strftime("%b%d"),
            "project_path": str(project), "multianimalproject": False,
            "bodyparts": KEEP, "skeleton": SKELETON, "video_sets": {},
            "default_augmenter": "default", "default_net_type": "hrnet_w32", "engine": "pytorch",
            "cropping": False, "start": 0, "stop": 1, "numframes2pick": 20,
            "TrainingFraction": [0.95], "iteration": 0, "snapshotindex": -1, "detector_snapshotindex": -1,
            "x1": 0, "x2": 640, "y1": 277, "y2": 624, "batch_size": 8, "detector_batch_size": 1,
            "corner2move2": (50, 50), "move2corner": True, "skeleton_color": "black",
            "pcutoff": 0.6, "dotsize": 8, "alphavalue": 0.7, "colormap": "rainbow",
        })
    scorer = cfg["scorer"]
    video_sets = dict(cfg.get("video_sets") or {})

    folders = sorted(p for p in src_root.iterdir() if p.is_dir()) if src_root.is_dir() else []
    if not folders:
        print(f"No AcinoSet folders found in {src_root}: making the project without them.")

    added = 0
    for folder in folders:
        dest = project / "labeled-data" / folder.name
        if (dest / f"CollectedData_{scorer}.h5").exists():
            print(f"{folder.name}: already in the project, skipped")
            continue
        df = load_labels(folder)
        images = sorted(folder.glob("*.png"))
        if df is None or not images:
            print(f"{folder.name}: no CollectedData file or no pictures, skipped")
            continue

        df, missing = tidy(df, folder.name, scorer)
        dest.mkdir(parents=True, exist_ok=True)
        for im in images:
            if not (dest / im.name).exists():
                link_or_copy(im, dest / im.name)
        df.to_hdf(dest / f"CollectedData_{scorer}.h5", key="df_with_missing", mode="w")
        df.to_csv(dest / f"CollectedData_{scorer}.csv")

        # DeepLabCut finds each folder through a video entry with the same name.
        # There is no real video here, so the path is a placeholder; only its name matters.
        w, h = Image.open(images[0]).size
        video_sets[str(project / "videos" / f"{folder.name}.mp4")] = {"crop": f"0, {w}, 0, {h}"}

        clicked = int(df.xs("x", level="coords", axis=1).notna().sum().sum())
        note = f", no clicks at all for {missing}" if missing else ""
        print(f"{folder.name}: {len(df)} pictures, {clicked} points clicked{note}")
        added += 1

    cfg["video_sets"] = video_sets
    write_config(str(config), cfg)
    print(f"\nAdded {added} folder(s). Project settings: {config}")


if __name__ == "__main__":
    main()