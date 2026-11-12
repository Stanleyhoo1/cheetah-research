"""
Swap two keypoints in your saved labels, for when you clicked one side as the other
(for example every right shoulder saved as l_shoulder).

It only changes the folders you name, so your other folders stay as they are.
A backup of each label file is saved first (CollectedData_*.h5.bak).

Close the labelling window first, so it cannot save the old labels back.

Usage (dlc environment):
    python scripts/labeling/swap_points.py cheetah-dlc/config.yaml l_shoulder r_shoulder zorro_20171209_straight_cam3
    python scripts/labeling/swap_points.py cheetah-dlc/config.yaml l_shoulder r_shoulder "zorro_20171209_straight_cam*"

Put a name with * in quotes. It matches folder names inside cheetah-dlc/labeled-data.
"""
import fnmatch
import shutil
import sys
from pathlib import Path

import pandas as pd


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    config = Path(sys.argv[1]).expanduser().absolute()
    a, b = sys.argv[2], sys.argv[3]
    patterns = sys.argv[4:]

    labeled = config.parent / "labeled-data"
    folders = sorted(p for p in labeled.iterdir() if p.is_dir()
                     and any(fnmatch.fnmatch(p.name, pat) for pat in patterns))
    if not folders:
        sys.exit(f"No folders in {labeled} match {patterns}")

    for folder in folders:
        files = sorted(folder.glob("CollectedData_*.h5"))
        if not files:
            print(f"{folder.name}: no saved labels, skipped")
            continue
        for f in files:
            df = pd.read_hdf(f)
            level = df.columns.names.index("bodyparts")
            parts = set(df.columns.get_level_values("bodyparts"))
            missing = [p for p in (a, b) if p not in parts]
            if missing:
                sys.exit(f"{f}: no point called {missing}. Points in this file: {sorted(parts)}")

            cols_a = [c for c in df.columns if c[level] == a]
            cols_b = [tuple(b if i == level else v for i, v in enumerate(c)) for c in cols_a]
            n_a = int(df[cols_a].notna().any(axis=1).sum())
            n_b = int(df[cols_b].notna().any(axis=1).sum())

            shutil.copy2(f, f.with_name(f.name + ".bak"))
            values_a = df[cols_a].to_numpy().copy()
            df[cols_a] = df[cols_b].to_numpy()
            df[cols_b] = values_a
            df.to_hdf(f, key="df_with_missing", mode="w")
            df.to_csv(f.with_suffix(".csv"))
            print(f"{folder.name}: swapped {a} ({n_a} pictures) and {b} ({n_b} pictures)")

    print("\nDone. Open a folder in the labelling window to check it looks right.")
    print("To undo, run the same command again (swapping twice puts them back).")


if __name__ == "__main__":
    main()