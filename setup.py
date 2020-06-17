from setuptools import find_packages, setup

setup(
    name="erdos-pylot",
    version="0.2",
    author="ERDOS Team",
    description=("A platform for developing autonomous vehicles."),
    long_description=open("README.md").read(),
    url="https://github.com/erdos-project/pylot",
    keywords=("autonomous vehicles driving python carla simulation"),
    packages=find_packages(),
    license="Apache 2.0",
    install_requires=[
        "absl-py",
        "cvxpy",
        "erdos>=0.2.0",
        "gdown",
        "lapsolver",
        "matplotlib==2.2.4",
        "networkx==2.2",
        "numpy<1.17",  # Update to newer numpy version once we switch to tf2
        "open3d-python==0.5.0.0",
        "opencv-python>=4.1.0.25",
        "opencv-contrib-python>=4.1.0.25",
        "pillow>=6.2.2",
        "pytest",
        "scikit-image<0.15",
        "scipy==1.2.2",
        "shapely==1.6.4",
        "tensorflow-gpu==1.15.2",
        "torch==1.3.1",
        "torchvision==0.2.1",
        ##### Sort tracking dependencies ####
        "filterpy==1.4.1",
        "numba==0.50.0",
        "scikit-learn==0.20.0",
    ],
)
