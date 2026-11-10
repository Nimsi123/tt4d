from setuptools import find_packages, setup

setup(
    name='robot_table_tennis',
    version='0.1',
    packages=find_packages(where="src"),  # Look inside `src/`
    package_dir={"": "src"},  # Treat `src/` as the root
)
