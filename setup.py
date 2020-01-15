from setuptools import find_packages, setup

setup(
    name="erdos-pylot",
    version="0.1",
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
        "matplotlib==2.2.4",
        "networkx==2.2",
        "numpy<1.17",  # Update to newer numpy version once we switch to tf2
        "open3d-python==0.4.0.0",
        "opencv-python>=4.1.0.25",
        "opencv-contrib-python>=4.1.0.25",
        "pid_controller",
        "pillow==6.2.1",
        "pptk",  # TODO(ionel): Fix pptk install (https://github.com/heremaps/pptk/issues/3)
        "pytest"
        "scikit-image<0.15",
        "scipy==1.2.2",
        "tensorflow-gpu>=1.12",
        "torch==0.4.1",
        "torchvision==0.2.1",
        ##### Sort tracking dependencies ####
        "filterpy==1.4.1",
        "numba==0.38.1",
        "scikit-learn==0.19.1",
    ],
)
