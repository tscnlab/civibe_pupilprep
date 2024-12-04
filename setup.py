from setuptools import setup, find_packages


with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="civibe-pupilprep-utils",
    version="0.1.3",
    description="Package for preprocessing pupillometry data recorded with RetinaWISE software",
    author="Diana Glebowicz, Hannah Sophie Heinrichs",
    author_email="diana.glebowicz@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    license_files=("LICENSE.txt"),
    install_requires=requirements,
)
