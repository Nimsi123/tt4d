# Ball lifting (stage 11)

Inference-only copy of the 3D ball trajectory and spin model from
"Uplifting Table Tennis: A Robust, Real-World Application for 3D Trajectory and Spin Estimation"
(Kienzle et al., WACV 2026; [paper](https://arxiv.org/abs/2511.20250), [code](https://kiedani.github.io/WACV2026/index.html)),
licensed as in `LICENSE.txt`. Training, data generation and detection code were removed.

- `uplifting_multisequence/model.py`: the multi-stage transformer (`get_model`).
- `uplifting_multisequence/helper.py`: constants, table keypoints, camera projection, spin-axis transform.
- `uplifting_multisequence/transformations.py`: the inference input normalization.
- `uplifting_multisequence/main.py`: the stage 11 runner. From this directory:
  `python -m uplifting_multisequence.main --resource-ids 0 --workers-per-resource 2`.
- `model/model_empiricalAugmentation_spincorrection.pt` (not in git): the checkpoint the runner loads.
