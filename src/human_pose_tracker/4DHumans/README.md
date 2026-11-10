Repository that handles the tracking of the players in a video of two people playing ping pong. Based on Humans in 4D.

# Conda environment setup:

```bash
conda env create --file env.yml
conda activate pose
conda remove cuda-version
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install -n pose gxx_linux-64
```

# PHALP setup:

1. `pip install git+https://github.com/brjathu/PHALP.git`
2. Replace `phalp/trackers/PHALP.py` with the local version in this repo. To help find the file, run `find ~ -name "PHALP.py"`.
3. Replace `phalp/visualize/visualizer.py` with the local version in this repo.
4. To complete the two steps above, consider running something along the lines of `find ~ -name "PHALP.py"`, `rm <path>`, `cp PHALP.py <path>`.

## On Slurm machine, 
if `Failed to build neural-renderer-pytorch`,
1. (in the pose env AND not in an srun env AND in `~/miniconda3/envs/pose/lib/python3.10/site-packages`) `git clone https://github.com/shubham-goel/NMR.git`
2. activate the srun env `srun --gres=gpu:10 --qos=low --nodelist=em2 --pty $SHELL`
3. cd into `NMR`
4. `python setup.py build`
5. `python setup.py install`
6. Check `import neural_renderer`
7. exit out of srun
8. retry `pip install git+https://github.com/brjathu/PHALP.git`
9. In 4DHumans folder, run `download_phalp_requirements.txt`

# SMPL Mesh

Make sure to find you have the `phalp/3D/models/smpl/SMPL_NEUTRAL.pkl` files

# DEMO

```
python demo.py \
    --img_folder example_data/images \
    --out_folder demo_out \
    --batch_size=48 --side_view --save_mesh --full_frame
```

# Archive (probably no longer needed)
The following are steps that I used to have to take for the pose environment, but not needed anymore. 

3. Hopefully we don't have to do anything below now that I've updated the env.yml
4. conda remove cuda-version
5. conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
6. conda install gxx_linux-64
