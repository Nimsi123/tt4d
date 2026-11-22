from setuptools import find_packages, setup

setup(
    name='robot_table_tennis',
    version='0.1',
    packages=find_packages(where="src"),  # Look inside `src/`
    package_dir={"": "src"},  # Treat `src/` as the root
    extras_require={
        "viewer": [
            "numpy==1.26.4", "scipy==1.16.1", "opencv-python==4.8.1.78",
            "pillow==11.3.0", "pyyaml==6.0.2", "tyro==0.9.28", "viser==1.0.10",
        ],
    },
)
