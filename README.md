# cheetah-research

3D point reconstruction of a running cheetah from six synchronised camera views
(the [AcinoSet](https://github.com/African-Robotics-Unit/AcinoSet) setup).
The repo holds small helper scripts for running 2D pose estimation on the camera
videos, tracking keypoints through time, and inspecting the 2D and 3D results.

## Layout

```
data/
  cam1.mp4 ... cam6.mp4   raw camera videos
  dlc/                    raw DeepLabCut predictions per camera (.h5 + metadata .pickle)
  dlc_pw/                 DeepLabCut predictions used for the pairwise 3D step
  fte_pw/                 AcinoSet full-trajectory-estimation output (3D reconstruction)
results/                  SuperAnimal inference output, one folder per video
superanimal_test/         early SuperAnimal test run on cam1
code.ipynb                scratch notebook: loading .h5 files, running SuperAnimal
```

The `data/`, `results/`, `superanimal_test/` folders, videos, `.h5` predictions
and model weights are git-ignored because of their size. Copy them in by hand.

## Scripts

### `inspect_pose.py` – step through 2D predictions on a video

```bash
python inspect_pose.py data/cam3.mp4 data/dlc/cam3DLC_resnet152_CheetahOct14shuffle1_500000.h5
python inspect_pose.py VIDEO PREDICTIONS.h5 --json PREDICTIONS.json      # adds SuperAnimal boxes
python inspect_pose.py VIDEO PREDICTIONS.h5 --export annotated.mp4       # write a video instead
```

Left body parts are drawn in blues and purples, right parts in reds and oranges,
midline parts in greens and yellows, so a left/right swap is easy to spot.
Arrow keys step frames, space plays, `p` prints every keypoint, `s` saves a PNG.
Run the script with `--help` for the full key list.

### `trackon_test.py` – track DeepLabCut keypoints with Track-On

For each keypoint, finds the frame where DeepLabCut was most confident, then
tracks that point forwards and backwards through the video with
[Track-On](https://github.com/gorkaydemir/track_on). Output is written in
DeepLabCut's `.h5` format so it opens in `inspect_pose.py`.

```bash
python trackon_test.py data/cam3.mp4 data/dlc_pw/cam3DLC_resnet152_CheetahOct14shuffle3_650000.h5 \
    --trackon ~/track_on --ckpt track_on_r.pt
```

Needs a clone of Track-On and its checkpoint (`track_on_r.pt`, not committed).
Useful flags: `--start/--end`, `--pcutoff`, `--no-crop`, `--forward-only`.

### `test.py` – view an AcinoSet 3D reconstruction

Animates `fte.pickle` (or `ekf.pickle`, `sba.pickle`, `fte.mat`) as a 3D skeleton.

```bash
python test.py path/to/run/fte/fte.pickle
python test.py path/to/run/fte/fte.pickle --save run.mp4
```

Drag to rotate, space to pause, arrow keys to step. Left limbs are blue, right
limbs orange.

## Running SuperAnimal pose estimation

DeepLabCut's pretrained `superanimal_quadruped` model needs no training. It
first finds the cheetah with a detector, then predicts keypoints inside the box.

```python
import deeplabcut

deeplabcut.video_inference_superanimal(
    ["data/cam3.mp4"],
    superanimal_name="superanimal_quadruped",
    model_name="hrnet_w32",                      # top-down pose model
    detector_name="fasterrcnn_resnet50_fpn_v2",  # finds the cheetah first
    max_individuals=1,
    device="cuda",
    dest_folder="results/cam3.mp4",
)
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install deeplabcut torch opencv-python-headless pandas tables matplotlib
```

Python 3.12 with DeepLabCut 3.0 is what the scripts were written against.

## File formats

Prediction `.h5` files are pandas DataFrames with a column MultiIndex of
`scorer / bodyparts / coords`, where `coords` is `x`, `y`, `likelihood`, one row
per frame. Multi-animal files add an `individuals` level. Load with
`pd.read_hdf(path)`.
