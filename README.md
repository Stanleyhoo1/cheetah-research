# Cheetah pose estimation

Tracking 15 body keypoints on running cheetahs in synchronised multi-camera video from
[AcinoSet](https://github.com/African-Robotics-Unit/AcinoSet), so that cheetah motion can later
be retargeted onto a quadruped robot.

- **Pose model.** A [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) 3 model (HRNet-W32,
  initialised from the SuperAnimal-Quadruped weights) trained on hand labels made in this repo.
- **Gap filling.** [Track-On](https://github.com/gorkaydemir/track_on), an online point tracker,
  is seeded from DeepLabCut's most confident detections and fills in points where DeepLabCut
  loses track or has very low confidence.

## Keypoints

| Midline     | Left           | Right          |
|-------------|----------------|----------------|
| `neck_base` | `l_shoulder`   | `r_shoulder`   |
| `spine`     | `l_hip`        | `r_hip`        |
| `tail_base` | `l_back_knee`  | `r_back_knee`  |
| `tail_mid`  | `l_back_ankle` | `r_back_ankle` |
| `tail_tip`  | `l_back_paw`   | `r_back_paw`   |

The point order and skeleton are defined in [`cheetah-dlc/config.yaml`](cheetah-dlc/config.yaml).

## Repository layout

```
cheetah-research/
├── README.md
├── LICENSE
├── environment.yml              conda env "dlc" for the laptop (with labelling GUI)
├── environment-lab.yml          conda env "dlc" for a headless machine
├── sequences.csv                every labelled moment: run, frame range, cameras
├── scripts/
│   ├── setup_project.py         point config.yaml at this checkout
│   ├── labeling/
│   │   ├── make_project.py      create the DLC project / import AcinoSet's labelled folders
│   │   ├── extract_sequence.py  pull the same frames from every camera, ready to label
│   │   ├── label_frames.py      open the napari labelling window on one folder
│   │   ├── swap_points.py       swap two keypoints in saved labels (left/right mix-ups)
│   │   ├── rename_points.py     rename keypoints in config.yaml and every label file
│   │   ├── update_sequences.py  regenerate sequences.csv from labeled-data
│   │   └── rebuild_frames.py    re-extract the labelled pictures from the videos
│   ├── training/
│   │   └── train.py             drop blank frames, build dataset, train, evaluate
│   └── analysis/
│       ├── inspect_pose.py      step through 2D predictions on a video
│       ├── trackon_from_dlc.py  Track-On seeded from DeepLabCut predictions
│       └── view_fte.py          animate an AcinoSet 3D reconstruction
├── notebooks/
│   └── superanimal_inference.ipynb   zero-shot SuperAnimal inference, model-zoo listing
└── cheetah-dlc/                 the DeepLabCut project
    ├── config.yaml
    └── labeled-data/<NAME>_cam<N>/
        ├── CollectedData_UCT.h5 / .csv   hand labels (tracked)
        └── img<frame>.png                pictures (not tracked, rebuilt from videos)
```

Run every command from the repository root. Each script explains its options with `--help`.

### What is in git

Only small files that cannot be regenerated: code, docs, `cheetah-dlc/config.yaml`, the hand
labels (`CollectedData_*.h5` and `.csv`) and `sequences.csv`. Everything else is ignored (see
[`.gitignore`](.gitignore)) and can be downloaded or rebuilt:

| Not tracked | How to get it back |
|---|---|
| `data/` (AcinoSet videos) | download from AcinoSet, or `rsync` from another machine |
| `labeled-data/*/img*.png` | `scripts/labeling/rebuild_frames.py` (file names are frame numbers) |
| `cheetah-dlc/videos/` | symlinks made by `extract_sequence.py` |
| `training-datasets/`, `dlc-models*/`, `evaluation-results*/` | `scripts/training/train.py` |
| `labeled-data/*_labeled/` | `deeplabcut.check_labels` |
| `results/` | DeepLabCut / Track-On analysis output |

## Setup (laptop, WSL)

```bash
conda env create -f environment.yml      # creates the "dlc" env
conda activate dlc
python scripts/setup_project.py          # set project_path in config.yaml for this machine
```

`environment.yml` also sets the variables the napari labelling window needs under WSL
(`QT_QPA_PLATFORM=xcb QT_XCB_GL_INTEGRATION=xcb_glx PYOPENGL_PLATFORM=glx`).

Put the AcinoSet videos in `data/<run>/cam1.mp4 … cam6.mp4`; the runs used so far are the `run`
column of [`sequences.csv`](sequences.csv). Then rebuild the labelled pictures:

```bash
python scripts/labeling/rebuild_frames.py
```

Track-On is a separate repository with its own environment. Clone it outside this repo (for
example to `~/track_on`) and follow its README to create a `trackon` env and get the checkpoint.

## Workflow

**extract → label → check labels → train → analyse**

### 1. Extract a moment

Pick a short moment of a run and pull the same frame numbers from every camera:

```bash
python scripts/labeling/extract_sequence.py data/romeoFlick_2019_02_27 romeo_flick_2019_02_27 \
    --start-frame 120 --cams 2 3                   # 25 consecutive frames by default
python scripts/labeling/update_sequences.py        # record it in sequences.csv
```

This creates `cheetah-dlc/labeled-data/<NAME>_cam<N>/img<frame>.png`, links the video into
`cheetah-dlc/videos/`, and registers it in `config.yaml`. Use `--start`/`--end` for seconds
instead of frame numbers, and `--frames` to change the count.

### 2. Label

```bash
python scripts/labeling/label_frames.py cheetah-dlc/config.yaml romeo_flick_2019_02_27_cam2
python scripts/labeling/label_frames.py cheetah-dlc/config.yaml romeo_flick_2019_02_27_cam2 1   # on screen 1
```

Save with Ctrl+S and close with File > Exit (Ctrl+Q). Labels are written to
`CollectedData_UCT.h5` / `.csv` in the folder. Leave frames blank where the cheetah is not
visible; `train.py` drops them.

### 3. Check labels

```bash
python -c "import deeplabcut; deeplabcut.check_labels('cheetah-dlc/config.yaml')"
```

This draws the labels onto each picture in `labeled-data/<NAME>_cam<N>_labeled/`. To fix
mistakes (close the labelling window first):

```bash
# one side labelled as the other, in some folders (keeps a .bak of each label file)
python scripts/labeling/swap_points.py cheetah-dlc/config.yaml l_shoulder r_shoulder "zorro_20171209_flick_cam*"
# rename keypoints in config.yaml and every label file
python scripts/labeling/rename_points.py cheetah-dlc/config.yaml tail1=tail_mid tail2=tail_tip
```

### 4. Train

```bash
python scripts/training/train.py cheetah-dlc/config.yaml --epochs 200 --shuffle 3
```

This drops blank frames, builds a training set initialised from SuperAnimal-Quadruped, trains
HRNet-W32, and evaluates with per-keypoint errors. Use a new `--shuffle` number for each
experiment. Models go to `cheetah-dlc/dlc-models-pytorch/`, and evaluation tables and plots to
`cheetah-dlc/evaluation-results-pytorch/`. If the GPU runs out of memory, lower `--batch-size`
(default 8) to 4.

### 5. Analyse

Run a trained model on videos:

```python
import deeplabcut

config = "cheetah-dlc/config.yaml"
videos = ["data/julesRun_2019_03_09/cam1.mp4"]
deeplabcut.analyze_videos(config, videos, shuffle=3, destfolder="results/shuffle3")
deeplabcut.create_labeled_video(config, videos, shuffle=3, destfolder="results/shuffle3", draw_skeleton=True)
```

Then inspect, gap-fill and visualise:

```bash
# step through predictions frame by frame (left = blues, right = reds, so swaps stand out)
python scripts/analysis/inspect_pose.py data/julesRun_2019_03_09/cam1.mp4 results/shuffle3/<prediction>.h5

# fill gaps with Track-On (in the trackon env); output opens in inspect_pose.py
python scripts/analysis/trackon_from_dlc.py data/julesRun_2019_03_09/cam1.mp4 results/shuffle3/<prediction>.h5 \
    --trackon ~/track_on --ckpt path/to/track_on_r.pt --out results/shuffle3/cam1_trackon.h5

# animate an AcinoSet 3D reconstruction (fte.pickle, ekf.pickle, sba.pickle or fte.mat)
python scripts/analysis/view_fte.py path/to/run/fte/fte.pickle
```

[`notebooks/superanimal_inference.ipynb`](notebooks/superanimal_inference.ipynb) runs the
pretrained SuperAnimal models zero-shot for comparison.

### Starting a project from scratch

`config.yaml` and the labels are already in git, so this is only needed for a fresh project.
`make_project.py` creates a DeepLabCut project and imports AcinoSet's own hand-labelled folders,
keeping the 15 keypoints above (AcinoSet's `tail1`/`tail2` become `tail_mid`/`tail_tip`):

```bash
python scripts/labeling/make_project.py cheetah_labels cheetah-dlc
```

## Lab machine (SSH, no display)

Labelling happens on the laptop; training and video analysis run on the lab computer. In the
commands below, `lab` is its SSH host (an alias in `~/.ssh/config`, or `user@hostname`), and
the repo is assumed to be at `~/cheetah-research` there.

**1. Clone and set up** (on the lab machine):

```bash
git clone git@github.com:Stanleyhoo1/cheetah-research.git ~/cheetah-research
cd ~/cheetah-research
conda env create -f environment-lab.yml  # no labelling GUI; sets MPLBACKEND=Agg
conda activate dlc
python scripts/setup_project.py
```

**2. Get the labelled pictures.** Training needs the PNGs, which are not in git. Either copy the
videos and rebuild the pictures (about 0.8 GB to transfer, and the videos are needed for
analysis anyway):

```bash
# on the laptop
rsync -avP --exclude='*:Zone.Identifier' --exclude='*:mshield' data/ lab:cheetah-research/data/
# on the lab machine
python scripts/labeling/rebuild_frames.py
```

or copy the pictures themselves (about 17 GB; labels come from git, so only PNGs are sent):

```bash
# on the laptop
rsync -avPm --exclude='*_labeled/' --include='*/' --include='img*.png' --exclude='*' \
    cheetah-dlc/labeled-data/ lab:cheetah-research/cheetah-dlc/labeled-data/
```

**3. Train inside tmux**, so training keeps running if the SSH connection drops:

```bash
tmux new -s train                # later: tmux attach -t train
conda activate dlc
cd ~/cheetah-research
export MPLBACKEND=Agg            # no display; environment-lab.yml already sets this
python scripts/training/train.py cheetah-dlc/config.yaml --epochs 200 --shuffle 3 2>&1 | tee train-shuffle3.log
# detach with Ctrl-b then d; the run continues in the background
```

**4. Copy results back** (on the laptop):

```bash
rsync -avP lab:cheetah-research/cheetah-dlc/evaluation-results-pytorch/ cheetah-dlc/evaluation-results-pytorch/
rsync -avP lab:cheetah-research/results/ results/
rsync -avP lab:cheetah-research/cheetah-dlc/dlc-models-pytorch/ cheetah-dlc/dlc-models-pytorch/   # to use a model on the laptop
```

**Keeping config.yaml in sync.** On the lab machine, `config.yaml` shows as modified because its
`project_path` differs. Don't commit it from there. To pull new labels or settings:

```bash
git checkout cheetah-dlc/config.yaml     # drop the local project_path change
git pull
python scripts/setup_project.py
```

## Data

### Videos

`data/<run>/cam<N>.mp4`: synchronised AcinoSet camera videos, one folder per run (90 fps at
1920×1080 for the 2017 runs, 120 fps at 2704×1520 for the 2019 runs). They are not
redistributed here; see [AcinoSet](https://github.com/African-Robotics-Unit/AcinoSet) for
access and terms.

### `sequences.csv`

One row per labelled moment, enough to rebuild every picture from the videos:

| Column | Meaning |
|---|---|
| `name` | moment name; its folders are `labeled-data/<name>_cam<N>` |
| `run` | run folder in `data/` |
| `start_frame`, `end_frame` | frame range, inclusive; pictures are `img<frame>.png` |
| `cameras` | camera numbers, space separated |

Frame numbers in file names are zero-padded to the digit count of the video's length (three
digits for every run so far). `update_sequences.py` writes this file from the folder and image
names in `labeled-data` and the video links in `cheetah-dlc/videos`.

### Label files

`CollectedData_UCT.h5` is a pandas DataFrame (`pd.read_hdf`) with one row per picture, indexed by
`(labeled-data, <folder>, img<frame>.png)`, and a column MultiIndex
`scorer / bodyparts / coords` with `coords` = `x`, `y` in pixels (NaN where not labelled). The
`.csv` next to it holds the same table.

### Prediction files

DeepLabCut and `trackon_from_dlc.py` write the same layout with one row per video frame and
`coords` = `x`, `y`, `likelihood`. For Track-On output, `likelihood` is its visible (1) / hidden
(0) flag.

## References

- D. Joska et al., "AcinoSet: A 3D Pose Estimation Dataset and Baseline Models for Cheetahs in
  the Wild," *ICRA*, 2021.
- A. Mathis et al., "DeepLabCut: markerless pose estimation of user-defined body parts with
  deep learning," *Nature Neuroscience*, 21, 2018.
- S. Ye et al., "SuperAnimal pretrained pose estimation models for behavioral analysis,"
  *Nature Communications*, 15, 2024.
- G. Aydemir, X. Cai, W. Xie, F. Güney, "Track-On: Transformer-based Online Point Tracking with
  Memory," *ICLR*, 2025.

## License

The code is released under the [MIT License](LICENSE). The AcinoSet videos are not part of this
repository and remain under AcinoSet's terms.
