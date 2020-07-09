Quick start
===========

The easiest way to run Pylot is to use our Docker images. First, please ensure
you have `nvidia-docker` on your machine before you start installing Pylot.
In case you do not have `nvidia-docker` please
run ``./scripts/install-nvidia-docker.sh``.

We provide a Docker image to run the CARLA simulator in, and a Docker image with
Pylot and ERDOS already setup.

.. code-block:: bash

    docker pull erdosproject/pylot
    docker pull erdosproject/carla

Next, create a Docker network, a CARLA container, and a Pylot container:

.. code-block:: bash

    docker network create carla-net
    nvidia-docker run -itd --name carla --net carla-net erdosproject/carla /bin/bash
    nvidia-docker run -itd --name pylot -p 20022:22 --net carla-net erdosproject/pylot /bin/bash


Following, start the simulator in the CARLA container:

.. code-block:: bash

    nvidia-docker exec -i -t carla /bin/bash
    SDL_VIDEODRIVER=offscreen /home/erdos/workspace/CARLA_0.9.6/CarlaUE4.sh -opengl -windowed -carla-server -benchmark -fps=20 -quality-level=Epic

Finally, start Pylot in the container:

.. code-block:: bash

    nvidia-docker exec -i -t pylot /bin/bash
    cd workspace/pylot/
    python3 pylot.py --flagfile=configs/detection.conf --carla_host=carla

If everything worked ok, you should be able to see a window that visualizes
detected obstacles like the one below:

.. image:: images/pylot-obstacle-detection.png
     :align: center
