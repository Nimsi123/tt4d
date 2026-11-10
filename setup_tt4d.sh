#!/bin/bash
# Finishes the tt4d env after `conda env create -f tt4d.yml`. Run from the repo root with the env activated.
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# 1. This repo, importable from anywhere
pip install -e .

# 2. chumpy patches for modern NumPy / Python
bash fix_chumpy.sh

# 3. detectron2, built without its own CUDA ops (the 4DHumans detectors do not use them, and compiling
#    them would need the full CUDA dev headers). Hiding the GPUs is what selects the CPU-only build.
CUDA_VISIBLE_DEVICES="" pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/detectron2@1e3e13bbf607b54f62205c4c33922521822fb298"

# 4. Neural renderer used by 4DHumans (track.py), compiled with the nvcc that tt4d.yml installs
CUDA_HOME="$CONDA_PREFIX" CPATH="$CONDA_PREFIX/targets/x86_64-linux/include" \
    pip install --no-deps --no-build-isolation \
    "git+https://github.com/shubham-goel/NMR.git@e990b3c70f48d39231f607c79d76ce3db4bf7483"

# 5. PHALP tracker (--no-deps: its own requirements would replace pinned packages), with the two files this repo patches
pip install --no-deps "git+https://github.com/brjathu/PHALP.git@677074a9bd7acac58c0b98a31b04ae54b93dcd2f"
PHALP_DIR=$(python -c "import importlib.util, os; print(os.path.dirname(importlib.util.find_spec('phalp').origin))")
cp src/human_pose_tracker/4DHumans/PHALP.py "$PHALP_DIR/trackers/PHALP.py"
cp src/human_pose_tracker/4DHumans/visualizer.py "$PHALP_DIR/visualize/visualizer.py"

# 6. SAM3 (racket tracker, stage 12), pinned to the commit the dataset was made with; its dependencies are in tt4d.yml
pip install --no-deps "git+https://github.com/facebookresearch/sam3.git@99d02f28c8d34610ab35559b72642b01f10fdb1e"

echo "------------------------------------------------"
python -c "import neural_renderer, phalp, detectron2, paddleocr, segmentation_models_pytorch, sam3; print('tt4d env setup complete.')"
