import os
from pathlib import Path

from rtt_utils.paths import get_home_path
from rtt_utils.smpl import load_smpl_model
from ttg.types.tt4d import TT4D

SRC_DIR = os.path.join(get_home_path(), "pipeline_outputs", "shared")
DST_DIR = os.path.join(get_home_path(), "pipeline_outputs", "nima_recons_tt4d_pickles_genmo")


def main():
    smpl_model = load_smpl_model(device="cuda:0")
    # if os.path.exists(DST_DIR):
    #     answer = input(
    #         f"{DST_DIR} already exists. Really delete it so a new one can be created? [y/N] "
    #     ).strip().lower()
    #     if answer != "y":
    #         print("Aborted.")
    #         sys.exit(0)
    #     shutil.rmtree(DST_DIR)
    # os.makedirs(DST_DIR)

    for path in Path(SRC_DIR).rglob("*_tt4d_processed.pkl"):
        dst = os.path.join(DST_DIR, path.name)
        if os.path.exists(dst):
            continue

        gameplay_dir = path.parent.parent
        tt4d = TT4D.from_gameplay_dir(str(gameplay_dir), smpl_model, pose_source="genmo")
        if tt4d.invalid:
            # print(f"Skipping (invalid): {path}")
            continue

        # if not (hasattr(tt4d, "handedness") and tt4d.handedness is not None):
        #     continue
        # tt4d.translation_fix_wrist_ball() do this on the downstream machine

        ...
        # need something other than shutil copy


if __name__ == "__main__":
    main()
