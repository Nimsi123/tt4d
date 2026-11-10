import os

from phalp.utils.utils_download import cache_url

if __name__ == "__main__":
    from phalp.configs.base import CACHE_DIR

    os.makedirs(os.path.join(CACHE_DIR, "phalp"), exist_ok=True)
    os.makedirs(os.path.join(CACHE_DIR, "phalp/3D"), exist_ok=True)
    os.makedirs(os.path.join(CACHE_DIR, "phalp/weights"), exist_ok=True)
    os.makedirs(os.path.join(CACHE_DIR, "phalp/ava"), exist_ok=True)

    smpl_path = os.path.join(CACHE_DIR, "phalp/3D/models/smpl/SMPL_NEUTRAL.pkl")

    if not os.path.exists(smpl_path):
        # We are downloading the SMPL model here for convenience. Please accept the license
        # agreement on the SMPL website: https://smpl.is.tue.mpg.

        os.makedirs(os.path.join(CACHE_DIR, "phalp/3D/models/"), exist_ok=True)
        os.makedirs(os.path.join(CACHE_DIR, "phalp/3D/models/smpl/"), exist_ok=True)
        # os.makedirs(os.path.join(CACHE_DIR, "phalp/3D/models/smpl"), exist_ok=True)
        # os.system('wget https://github.com/classner/up/raw/master/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
        # convert_pkl('basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
        # os.system('rm basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')

        from rtt_utils.paths import get_home_path

        src_pkl = os.path.join(
            get_home_path(), 'src/human_pose_tracker/4DHumans/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
        )
        if not os.path.exists(src_pkl):
            raise FileNotFoundError(
                f'SMPL model not found at {src_pkl}. Download basicModel_neutral_lbs_10_207_0_v1.0.0.pkl from '
                'https://smplify.is.tue.mpg.de/ and place it there (see the README).'
            )
        os.system(f'cp {src_pkl} {smpl_path}')

    download_files = {
        "head_faces.npy": [
            "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/head_faces.npy",
            os.path.join(CACHE_DIR, "phalp/3D"),
        ],
        "mean_std.npy": [
            "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/mean_std.npy",
            os.path.join(CACHE_DIR, "phalp/3D"),
        ],
        "smpl_mean_params.npz": [
            "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/smpl_mean_params.npz",
            os.path.join(CACHE_DIR, "phalp/3D"),
        ],
        "SMPL_to_J19.pkl": [
            "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/SMPL_to_J19.pkl",
            os.path.join(CACHE_DIR, "phalp/3D"),
        ],
        "texture.npz": [
            "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/texture.npz",
            os.path.join(CACHE_DIR, "phalp/3D"),
        ],
        "bmap_256.npy": [
            "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/bmap_256.npy",
            os.path.join(CACHE_DIR, "phalp/3D"),
        ],
        "fmap_256.npy": [
            "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/fmap_256.npy",
            os.path.join(CACHE_DIR, "phalp/3D"),
        ],
        "hmar_v2_weights.pth": [
            "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/weights/hmar_v2_weights.pth",
            os.path.join(CACHE_DIR, "phalp/weights"),
        ],
        "pose_predictor.pth": [
            "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/weights/pose_predictor_40006.ckpt",
            os.path.join(CACHE_DIR, "phalp/weights"),
        ],
        "pose_predictor.yaml": [
            "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/weights/config_40006.yaml",
            os.path.join(CACHE_DIR, "phalp/weights"),
        ],
        # data for ava dataset
        "ava_labels.pkl": [
            "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/ava/ava_labels.pkl",
            os.path.join(CACHE_DIR, "phalp/ava"),
        ],
        "ava_class_mapping.pkl": [
            "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/ava/ava_class_mappping.pkl",
            os.path.join(CACHE_DIR, "phalp/ava"),
        ],
        'resnet50-19c8e357.pth': [
            'https://download.pytorch.org/models/resnet50-19c8e357.pth',
            os.path.expanduser('~/.cache/torch/hub/checkpoints'),
        ],
        "hmr2_data.tar.gz": [
            "https://people.eecs.berkeley.edu/~jathushan/projects/4dhumans/hmr2_data.tar.gz",
            os.path.expanduser('~/.cache/4DHumans'),
        ],
    }

    for file_name, url in download_files.items():
        output_path = os.path.join(url[1], file_name)
        if not os.path.exists(output_path):
            print("Downloading file: " + file_name)
            # output = gdown.cached_download(url[0], output_path, fuzzy=True)
            output = cache_url(url[0], output_path)
            assert os.path.exists(output_path), f"{output} does not exist"

            # if ends with tar.gz, tar -xzf
            if file_name.endswith(".tar.gz"):
                print("Extracting file: " + file_name)
                os.system("tar -xvf " + output_path + " -C " + url[1])
