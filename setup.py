from setuptools import find_packages, setup

setup(
    name="erdos-pylot",
    version="0.3.3",
    author="Pylot Team",
    description=("A platform for developing autonomous vehicles."),
    long_description=open("README.md").read(),
    url="https://github.com/erdos-project/pylot",
    keywords=("autonomous vehicles driving python CARLA simulation"),
    packages=find_packages(),
    license="Apache 2.0",
    install_requires=[
        "absl-py",
        "cvxpy",
        "erdos>=0.3.1",
        "lapsolver",
        "motmetrics",
        "numpy<1.20",  # Update once TF supports a newer version.
        "open3d==0.13.0",
        "opencv-python>=4.1.0.25",
        "opencv-contrib-python>=4.1.0.25",
        "pillow>=6.2.2",
        "pycocotools",
        "pygame==1.9.6",
        "pytest",
        "scikit-image==0.18.3",
        "scipy==1.7.1",
        "shapely==1.6.4",
        "tensorflow-gpu==2.9.3",
        "torch==1.4.0",
        "torchvision==0.5.0",
        ##### Tracking dependencies #####
        "Cython",
        "filterpy==1.4.1",
        "imgaug==0.2.8",
        "matplotlib==3.4.3",
        "nonechucks==0.3.1",
        "nuscenes-devkit",
        "progress",
        "pyquaternion",
        "scikit-learn==1.0",
        ##### CARLA dependencies #####
        "networkx==2.2",
    ],
)
