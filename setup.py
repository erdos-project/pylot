from setuptools import find_packages, setup

setup(
    name="erdos-pylot",
    version="0.1",
    author="ERDOS Team",
    description=("A platform for developing autonomous vehicles."),
    long_description=open("README.md").read(),
    url="https://github.com/erdos-project/pylot",
    keywords=("autonomous vehicles driving python"),
    packages=find_packages(),
    license="Apache 2.0",

    install_requires=[
        "absl-py",
        "cffi",
        "gdown",
        "erdos>=0.1.1",
        "matplotlib==2.2.4",
        "networkx>=2.2",
        "open3d-original",
        "opencv-python>=4.1.0.25",
        "opencv-contrib-python>=4.1.0.25",
        "pid_controller",
        "pillow",
        "pptk", # TODO(ionel): Fix pptk install (https://github.com/heremaps/pptk/issues/3)
        "tensorflow-gpu>=1.12",
        "protobuf",
        "ray",
        "rospkg",
        "scikit-image",
        "scipy",
        "setproctitle",
        "torch==0.4.1",
        "torchvision==0.2.1"
    ],
)
