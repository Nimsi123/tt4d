import os
import shutil
import sys
from pathlib import Path

from rtt_utils.paths import get_home_path
from ttg.types.tt4d import TT4D

SRC_DIR = os.path.join(get_home_path(), "pipeline_outputs", "shared")
DST_DIR = os.path.join(get_home_path(), "pipeline_outputs", "tt4d_pickles")


def main():
    if os.path.exists(DST_DIR):
        answer = (
            input(f"{DST_DIR} already exists. Really delete it so a new one can be created? [y/N] ").strip().lower()
        )
        if answer != "y":
            print("Aborted.")
            sys.exit(0)
        shutil.rmtree(DST_DIR)
    os.makedirs(DST_DIR)

    for path in Path(SRC_DIR).rglob("*_tt4d_processed.pkl"):
        tt4d = TT4D.load(str(path))
        if tt4d.invalid:
            # print(f"Skipping (invalid): {path}")
            continue
        dst = os.path.join(DST_DIR, path.name)
        shutil.copy(path, dst)
        # print(f"Copied {path} -> {dst}")


if __name__ == "__main__":
    main()
