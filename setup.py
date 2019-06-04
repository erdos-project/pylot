from setuptools import find_packages, setup

setup(
    name = "pylot",
    version = "0.1",
    packages = find_packages(),
    license = "Apache 2.0",
    url = "https://github.com/erdos-project/pylot",
    long_description = open("README.md").read(),
    install_requires = [
        "absl-py",
        "cffi",
        "gdown",
        "erdos",
        "matplotlib==2.2.4",
        "open3d-original",
        "opencv-python",
        "opencv-contrib-python",
        "pid_controller",
        "pillow",
#        "pptk",
        "protobuf",
        "ray",
        "rospkg",
        "scipy",
        "setproctitle",
        "torch==0.4.1",
        "torchvision==0.2.1"
    ],
)
