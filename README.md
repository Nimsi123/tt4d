# TT4D: A Pipeline and Dataset for Table Tennis 4D Reconstruction From Monocular Videos

[Project page](https://nimsi123.github.io/tt4d/) · [arXiv](https://arxiv.org/abs/2605.01234) · ACM Multimedia 2026

This repository has the pipeline that turns broadcast table tennis videos into per-rally reconstructions (3D ball
trajectory and spin, SMPL meshes of the players, camera calibration) and a viewer for the resulting TT4D clips.

## Setup

Linux with an NVIDIA GPU. One conda env runs every stage and the viewer:

```bash
git clone https://github.com/Nimsi123/tt4d.git && cd tt4d
conda env create -f tt4d.yml
conda activate tt4d
bash setup_tt4d.sh
```

`setup_tt4d.sh` installs this repo (`pip install -e .`), patches chumpy, and installs detectron2, the 4DHumans
neural renderer, PHALP and [SAM3](https://github.com/facebookresearch/sam3) (commit `99d02f2`) from GitHub.

### Models

Not included in the repository:

| Put it at | Used by | Source |
|---|---|---|
| `src/assets/smpl/SMPL_MALE.pkl` | stages 10–11, viewer | [SMPL](https://smpl.is.tue.mpg.de): `basicmodel_m_lbs_10_207_0_v1.*.pkl`, renamed |
| `src/human_pose_tracker/4DHumans/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` | stage 8 | [SMPLify](https://smplify.is.tue.mpg.de) |
| `src/ball_tracker/finetune_10/TrackNet_best.pt` | stage 5 | [download](https://people.eecs.berkeley.edu/~nimsi/tt4d/TrackNet_best.pt) |
| `src/ball_uplifting/model/model_empiricalAugmentation_spincorrection.pt` | stage 11 | [download](https://people.eecs.berkeley.edu/~nimsi/tt4d/model_empiricalAugmentation_spincorrection.pt) |
| Hugging Face `facebook/sam3` | stage 12 | request access on Hugging Face, then `hf auth login` |

```bash
mkdir -p src/assets/smpl src/human_pose_tracker/4DHumans/data src/ball_tracker/finetune_10 src/ball_uplifting/model
wget -P src/ball_tracker/finetune_10 https://people.eecs.berkeley.edu/~nimsi/tt4d/TrackNet_best.pt
wget -P src/ball_uplifting/model https://people.eecs.berkeley.edu/~nimsi/tt4d/model_empiricalAugmentation_spincorrection.pt
```

The SMPL models are licensed by MPI, so you have to register and download them yourself. The scoreboard detector and
table segmentation weights are in the repository. The rest download on first run: YOLOv5 and PaddleOCR (stage 1, to
`~/.cache/torch/hub` and `~/.paddleocr`), 4DHumans and PHALP (stage 8, about 5 GB in `~/.cache/4DHumans` and
`~/.cache/phalp`) and SAM3 (stage 12, about 3 GB in `~/.cache/huggingface`).

To only view the dataset you need the env and `SMPL_MALE.pkl`.

## Dataset

From the repo root:

```bash
wget https://people.eecs.berkeley.edu/~nimsi/tt4d/tt4d_pickles.zip   # 57 GB
unzip tt4d_pickles.zip -d pipeline_outputs && rm tt4d_pickles.zip
```

To try things first, `tt4d_pickles_small.zip` (5 MB, 10 clips: 7 singles, 3 doubles) has the same layout:

```bash
wget https://people.eecs.berkeley.edu/~nimsi/tt4d/tt4d_pickles_small.zip
unzip tt4d_pickles_small.zip -d pipeline_outputs
```

`pipeline_outputs/tt4d_pickles/` has one `<clip>_tt4d_processed.pkl` per rally, where `<clip>` is
`<video_id>_<start frame>_<end frame>_<score>` and `video_id` is the YouTube ID.

## Visualizing a clip

```bash
python src/visualization/viser_visualization.py --tt4d-file pipeline_outputs/tt4d_pickles/<clip>_tt4d_processed.pkl
# e.g. --tt4d-file pipeline_outputs/tt4d_pickles/ji-QkjJACZI_308547_309232_2_5_2_5_tt4d_processed.pkl (doubles, in the small zip)
```

Open http://localhost:8080 (`--port` to change it; on a remote machine, first run
`ssh -N -L 8080:localhost:8080 <user>@<server>`). The
viewer loops the rally: table, ball with its trail, and each player as an SMPL mesh and skeleton. The broadcast
video and camera are added when the clip folder recorded in the pickle exists on your machine, i.e. for clips you
ran through the pipeline yourself. `--gameplay-dir pipeline_outputs/shared/batch_clipped_long_<i>/<clip>` loads a
clip folder directly; `--lite` skips the video. The `overflow encountered in cast` and "SMPL model, with only 10 shape
coefficients" warnings at startup are harmless.

Each clip is a pickled `TT4D` object (`src/ttg/types/tt4d.py`); loading it needs this repo installed:

```python
import pickle

tt4d = pickle.load(open("<clip>_tt4d_processed.pkl", "rb"))
```

- `ball_2d` (T, 4): frame, visibility, x, y from the 2D tracker. `ball_3d`, `spin_3d` (T, 3): lifted ball and spin.
- `humans`: per player, a list of T dicts with `global_orient` (1, 3, 3), `body_pose` (23, 3, 3), `betas` (10,),
  `transl` (3,) (SMPL, rotation matrices, world frame). `skeletons`: per player (T, 21, 3) joints.
- `camera_matrix`, `rvecs`, `tvecs`: calibration. World frame: origin at the table centre, z up.
- `segments`, `hit_indices`: rally structure. `fps`, `dropped_frames`: frame timing after removing duplicated frames.
- `label`: singles or doubles. Doubles clips have four entries in `humans` but two in `skeletons`.

## Running the pipeline

### Small example

```bash
wget https://people.eecs.berkeley.edu/~nimsi/tt4d/pipeline_outputs_test.zip && unzip pipeline_outputs_test.zip   # at the repo root
```

This gives two match videos in `pipeline_outputs_test/batch_unclipped_long_1000/` with stage 1 already done. Run
stages 2–12 from the table below with `--test` added to every command (which makes each runner use
`pipeline_outputs_test/` instead of `pipeline_outputs/shared/`) and `--resource-ids` set to your GPUs. With 4 GPUs this
takes about 2 hours. The videos yield 41 clips (28 doubles, 13 singles); 34 reach stage 11 and about 22 of those come
out valid. View a valid one with

```bash
python src/visualization/viser_visualization.py --gameplay-dir pipeline_outputs_test/batch_clipped_long_1000/BqMxIHMOvRQ_12790_13210_1_3_0_4
```

The viewer only works on clips whose `nima_recons/<clip>_tt4d_processed.pkl` is valid.

### Input videos

`video_links.txt` lists the 3,581 YouTube videos the dataset was built from. The pipeline reads
`pipeline_outputs/shared/batch_unclipped_long_<i>/<video_id>/video/<video_id>.mp4`. To download them into batch 1
(needs `pip install yt-dlp`; YouTube may ask for `--cookies`):

```bash
mkdir -p pipeline_outputs/shared/batch_unclipped_long_1 && cd pipeline_outputs/shared/batch_unclipped_long_1
yt-dlp -f "bv*[height>=480][vcodec^=avc1]/bv*[height>=480]" -o "%(id)s.%(ext)s" -a ../../../video_links.txt
for f in *.webm; do ffmpeg -i "$f" -c:v libx264 -pix_fmt yuv420p "${f%.webm}.mp4" && rm "$f"; done
for f in *.mp4; do mkdir -p "${f%.mp4}/video" && mv "$f" "${f%.mp4}/video/"; done
```

Only the video stream is needed. We used batches of about 70 videos, but any split works.

### Stages

Run each command from the directory shown (several scripts use relative paths), with the env activated. Every
runner takes `--resource-ids 0 1 2` (GPUs) or `--num-resources N` (CPU processes), plus `--workers-per-resource`,
and `--batch-start`/`--batch-end` (inclusive, default 1 to 52) to choose which `batch_*_long_<i>` folders to process.
Batches without a folder are skipped.

Times are for one doubles match video (28 clips) on a shared machine with 4 GPUs.

| # | Stage | Directory (`src/...`) | Command | Output (per clip) | Time |
|---|---|---|---|---|---|
| 1 | Scoreboard OCR | `rally_clipper/scoreboard_ocr` | `python main.py --resource-ids 0` | `sb/sb.pkl` (per video) | 17 min |
| 2 | Clip into rallies | `rally_clipper` | `python first_clipper.py --num-resources 8` | `batch_clipped_long_<i>/<clip>/video/` | 10 s |
| 3 | Singles/doubles (scoreboard) | `rally_clipper` | `python game_type_distinguish.py --num-resources 8` | `metadata/{singles,doubles}.txt` | 1 s |
| 4 | Camera calibration | `calibration` | `python main.py --resource-ids 0 1 2 3 --workers-per-resource 2` | `thomas_calibration/calib.yaml` | 7 min |
| 5 | 2D ball tracking | `ball_tracker` | `python only_tracknet_main.py --resource-ids 0 1 2 3` | `ball_tracker_only_tracknet/<clip>_ball.csv` | 32 min |
| 6 | Trim clips to ball play | `rally_clipper` | `python second_clipper_2.py --num-resources 8` | `video_second_clipped_only_tracknet/` | 9 s |
| 7 | Duplicated frames | `video_thinning` | `python main.py --num-resources 8` | `video_thinning/frames_to_drop.json` | 8 min |
| 8 | Human pose (4DHumans) | `human_pose_tracker/4DHumans` | `python main.py --resource-ids 0 1 2 3 --workers-per-resource 2` | `human_pose_tracker_only_tracknet/4DHumans/` | 21 min |
| 9 | Singles/doubles (humans) | `rally_clipper` | `python game_type_distinguish_humans.py --num-resources 8` | `metadata/{singles,doubles}_humans.txt` | 2 s |
| 10 | Partial reconstruction | `mono_view_recons/runners` | `python partial_recons.py --resource-ids 0 --workers-per-resource 2` | `nima_recons/<clip>_tt4d.pkl` | 15 s |
| 11 | Ball lifting, final TT4D | `ball_uplifting` | `python -m uplifting_multisequence.main --resource-ids 0 --workers-per-resource 2` | `nima_recons/<clip>_tt4d_processed.pkl` | 6 min |
| 12 | Racket segmentation (SAM3) | `racket_tracker` | `python main_sam3.py --resource-ids 0 1` | `racket_tracker_sam3/` | 1 min/clip |

Each stage skips clips it has already finished, so a stage can be re-run after a crash. Clips dropped at stage 4
(moving camera) or stage 6 (no ball play) get no pickle; clips that reach stage 10 are always saved, and the ones that
fail have `invalid` and `failure_reason` set.

Dependencies to be aware of:
- Stage 5 only runs on clips whose calibration succeeded.
- Stage 10 only runs on clips where the scoreboard label (stage 3) and the humans label (stage 9) agree, and it uses
  stage 7's output.
- Stage 12 runs on singles clips only and is optional: the 4DHumans reconstructions do not use it. Its racket masks
  decide each player's handedness when loading GENMO poses (`TT4D.from_gameplay_dir(..., pose_source="genmo")`);
  GENMO itself is not part of this repository.

To collect every valid clip under `pipeline_outputs/shared/` into `pipeline_outputs/tt4d_pickles/` (this replaces
that folder after asking, so move a downloaded dataset elsewhere first):

```bash
cd src && python -m ttg.infra.create_tt4d_dataset
```

To render the 2D ball track onto the video after stage 5: `cd src/ball_tracker && python overlay_ball_on_videos.py --num-resources 8`.

## Third-party code

Vendored with their licenses: [TT3D](https://github.com/cogsys-tuebingen/tt3d) calibration (`src/calibration/tt3d`),
[TrackNetV3](https://github.com/qaz812345/TrackNetV3) (`src/ball_tracker`),
[4DHumans](https://github.com/shubham-goel/4D-Humans) (`src/human_pose_tracker/4DHumans`), and the ball lifting
model of Kienzle et al. (`src/ball_uplifting`, GPLv3).

## Citation

```bibtex
@inproceedings{rahmanian2026tt4d,
  author    = {Rahmanian, Nima and Kienzle, Daniel and Gossard, Thomas and Kalaria, Dvij and Lienhart, Rainer and Sastry, S. Shankar},
  title     = {TT4D: A Pipeline and Dataset for Table Tennis 4D Reconstruction From Monocular Videos},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia (MM '26)},
  year      = {2026},
  doi       = {10.1145/3767308.3835475},
}
```

The project website is in `docs/` and is served by GitHub Pages.
