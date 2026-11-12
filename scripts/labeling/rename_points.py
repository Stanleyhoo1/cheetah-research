"""
Rename keypoints everywhere in a DeepLabCut project: the settings file
(point list and skeleton) and every saved label file.

Close the labelling window first, so it cannot save the old names back.

Usage (dlc environment):
    python scripts/labeling/rename_points.py cheetah-dlc/config.yaml tail1=tail_mid tail2=tail_tip
"""
import sys
from pathlib import Path

import pandas as pd


def main():
    if len(sys.argv) < 3 or not all("=" in a for a in sys.argv[2:]):
        sys.exit(__doc__)
    from deeplabcut.utils.auxiliaryfunctions import read_config, write_config

    config = Path(sys.argv[1]).expanduser().absolute()
    renames = dict(a.split("=", 1) for a in sys.argv[2:])

    cfg = read_config(str(config))
    missing = [old for old in renames if old not in cfg["bodyparts"]]
    if missing:
        sys.exit(f"Not in the project's point list: {missing}. Current points: {cfg['bodyparts']}")
    cfg["bodyparts"] = [renames.get(bp, bp) for bp in cfg["bodyparts"]]
    cfg["skeleton"] = [[renames.get(a, a), renames.get(b, b)] for a, b in (cfg.get("skeleton") or [])]
    write_config(str(config), cfg)
    print(f"settings: {', '.join(f'{o} -> {n}' for o, n in renames.items())}")

    files = sorted((config.parent / "labeled-data").glob("*/CollectedData_*.h5"))
    files += sorted((config.parent / "labeled-data").glob("*/machinelabels*.h5"))
    for f in files:
        df = pd.read_hdf(f)
        df = df.rename(columns=renames, level="bodyparts")
        df.to_hdf(f, key="df_with_missing", mode="w")
        if f.name.startswith("CollectedData"):
            df.to_csv(f.with_suffix(".csv"))
        print(f"labels: {f.parent.name}/{f.name}")
    if not files:
        print("labels: none saved yet, nothing else to change")


if __name__ == "__main__":
    main()