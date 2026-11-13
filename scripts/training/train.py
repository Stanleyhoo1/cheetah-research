"""
Train a DeepLabCut model on the labelled pictures in the project, then test it.

Steps:
  1. drops pictures you left blank (a cheetah with no points would teach "no cheetah here")
  2. splits pictures into training and test sets, and picks the starting weights
  3. trains
  4. tests it and draws its guesses next to your labels

Fair comparisons between experiments:
  By default DeepLabCut picks test pictures at random. Neighbouring frames of the same moment
  then end up in both sets, so the test score looks better than it really is, and every
  experiment gets different test pictures. With --test, whole folders are kept out of training
  and used as the test set, so experiments can be compared on exactly the same pictures.

Usage (dlc environment, from the repository root):
    python scripts/training/train.py cheetah-dlc/config.yaml --epochs 200 --shuffle 3

    # keep two whole moments out of training and test on them
    python scripts/training/train.py cheetah-dlc/config.yaml --shuffle 10 \\
        --test "zorro_20171209_flick_cam*" "jules_flick_2017_09_02_cam*"

    # same test set, but start from ImageNet weights instead of SuperAnimal
    python scripts/training/train.py cheetah-dlc/config.yaml --shuffle 11 --init imagenet \\
        --test "zorro_20171209_flick_cam*" "jules_flick_2017_09_02_cam*"

    # same test set, but train only on some folders (e.g. only your own, not AcinoSet's)
    python scripts/training/train.py cheetah-dlc/config.yaml --shuffle 12 --train "*_cam*" \\
        --test "zorro_20171209_flick_cam*" "jules_flick_2017_09_02_cam*"

Names with * must be in quotes. They match folder names in cheetah-dlc/labeled-data
(upper and lower case count as different).
"""
import argparse
from fnmatch import fnmatchcase
from pathlib import Path, PureWindowsPath


def fixed_split(config, test_patterns, train_patterns):
    """Row numbers for training and testing, in the same order DeepLabCut merges the labels
    (it reads every folder listed in config.yaml, joins them and sorts by picture name)."""
    import pandas as pd
    from deeplabcut.utils.auxiliaryfunctions import read_config

    cfg = read_config(config)
    labeled = Path(cfg["project_path"]) / "labeled-data"
    frames = []
    for video in cfg["video_sets"]:
        folder = PureWindowsPath(video).stem  # works for Linux and Windows paths
        f = labeled / folder / f"CollectedData_{cfg['scorer']}.h5"
        if not f.exists():
            continue
        df = pd.read_hdf(f)
        if df.columns.levels[0][0] != cfg["scorer"]:
            continue  # DeepLabCut skips files labelled under another scorer name too
        if not isinstance(df.index, pd.MultiIndex):
            df.index = pd.MultiIndex.from_tuples([tuple(str(i).replace("\\", "/").split("/")) for i in df.index])
        frames.append(df)
    rows = pd.concat(frames).sort_index().index
    folders = [r[1] for r in rows]

    test, train = [], []
    for i, folder in enumerate(folders):
        if any(fnmatchcase(folder, p) for p in test_patterns):
            test.append(i)
        elif not train_patterns or any(fnmatchcase(folder, p) for p in train_patterns):
            train.append(i)

    count = lambda idx: len({folders[i] for i in idx})
    print(f"   training: {len(train)} pictures from {count(train)} folders")
    print(f"   testing:  {len(test)} pictures from {count(test)} folders")
    unused = len(folders) - len(train) - len(test)
    if unused:
        print(f"   not used: {unused} pictures (not matched by --train)")
    if not test:
        raise SystemExit(f"No folders match --test {test_patterns}")
    if not train:
        raise SystemExit("No pictures left for training.")
    return train, test


def add_training_fraction(config, fraction):
    """Make sure config.yaml lists this training fraction, and return its position in the list.

    DeepLabCut files each dataset under its training fraction (for example "trainset76") and,
    when training or testing, only looks under the fractions listed in config.yaml. A --test
    split usually gives a fraction that isn't listed yet. Only one line is added; the rest of
    config.yaml stays byte-for-byte the same.
    """
    import re

    text = Path(config).read_text()
    found = re.search(r"^TrainingFraction:\n((?:- [0-9.]+\n)+)", text, re.M)
    if not found:
        raise SystemExit("Couldn't find the TrainingFraction list in config.yaml")
    fractions = [float(line[2:]) for line in found.group(1).splitlines()]
    if fraction not in fractions:
        fractions.append(fraction)
        Path(config).write_text(text[:found.end()] + f"- {fraction}\n" + text[found.end():])
        print(f"   added training fraction {fraction} to config.yaml")
    return fractions.index(fraction)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--epochs", type=int, default=100, help="passes through all the training pictures (default 100)")
    ap.add_argument("--shuffle", type=int, default=1, help="experiment number (default 1); use a new one per experiment")
    ap.add_argument("--batch-size", type=int, default=8, help="lower to 4 if you get 'CUDA out of memory'")
    ap.add_argument("--workers", type=int, default=4,
                    help="CPU helpers that load pictures in parallel while the GPU trains (default 4)")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="test during training every N epochs (default 0: only once, at the end)")
    ap.add_argument("--init", choices=["superanimal", "imagenet"], default="superanimal",
                    help="starting weights: SuperAnimal-Quadruped (already knows four-legged animals, default) "
                         "or ImageNet (general pictures only)")
    ap.add_argument("--test", nargs="+", metavar="FOLDER", help="folders to keep out of training and test on")
    ap.add_argument("--train", nargs="+", metavar="FOLDER",
                    help="only train on these folders (default: every folder not in --test). Needs --test.")
    args = ap.parse_args()
    if args.train and not args.test:
        ap.error("--train needs --test, so the test pictures are chosen too")

    import deeplabcut
    from deeplabcut.modelzoo import build_weight_init

    config = str(Path(args.config).expanduser().absolute())

    print("\n1. Removing blank pictures from the label files")
    deeplabcut.dropunlabeledframes(config)

    print("\n2. Setting up training")
    split = {}
    trainingsetindex = 0  # which TrainingFraction in config.yaml this experiment is filed under
    if args.test:
        train_rows, test_rows = fixed_split(config, args.test, args.train)
        split = {"trainIndices": [train_rows], "testIndices": [test_rows]}
        # same rounding DeepLabCut uses when it names the dataset folder
        fraction = round(len(train_rows) / (len(train_rows) + len(test_rows)), 2)
        trainingsetindex = add_training_fraction(config, fraction)
    else:
        print("   test pictures picked at random (use --test to choose them)")
    weight_init = None
    if args.init == "superanimal":
        print("   starting from SuperAnimal-Quadruped (downloads the weights the first time)")
        weight_init = build_weight_init(
            cfg=config, super_animal="superanimal_quadruped",
            model_name="hrnet_w32", detector_name=None, with_decoder=False,
        )
    else:
        print("   starting from ImageNet weights")
    deeplabcut.create_training_dataset(
        config, Shuffles=[args.shuffle], net_type="hrnet_w32", weight_init=weight_init,
        userfeedback=False, **split,
    )

    print(f"\n3. Training for {args.epochs} epochs")
    updates = {
        # Load pictures in parallel: decoding big PNGs on one CPU thread was the bottleneck.
        "train_settings.dataloader_workers": args.workers,
        "train_settings.dataloader_pin_memory": True,
        # Testing full-size pictures mid-training can overflow the GPU's memory, and Windows
        # then quietly borrows normal RAM, which makes everything crawl. Test at the end instead.
        "runner.eval_interval": args.eval_every if args.eval_every > 0 else args.epochs + 1,
        # Slow the learning down near the end (at 80% and 95%) whatever the number of epochs;
        # this is the step that makes the model settle and become confident.
        "runner.scheduler.params.milestones": [int(args.epochs * 0.8), int(args.epochs * 0.95)],
    }
    deeplabcut.train_network(
        config, shuffle=args.shuffle, trainingsetindex=trainingsetindex, device="cuda:0", epochs=args.epochs,
        save_epochs=max(1, args.epochs // 4), batch_size=args.batch_size,
        pytorch_cfg_updates=updates,
    )

    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()  # hand the training memory back before testing

    print("\n4. Testing")
    deeplabcut.evaluate_network(config, shuffles=[args.shuffle], trainingsetindex=trainingsetindex,
                                plotting=True, per_keypoint_evaluation=True)
    print("\nDone. Pictures with your labels (+) and the model's guesses (dots) are in "
          f"{Path(config).parent / 'evaluation-results-pytorch'}")


if __name__ == "__main__":
    main()