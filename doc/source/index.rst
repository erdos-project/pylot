Pylot
=====

Pylot is an autonomous vehicle platform for developing and testing autonomous
vehicle components on top of the CARLA simulator.

.. _`codebase on GitHub`: https://github.com/erdos-project/pylot

Quick start
-----------
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
    nvidia-docker run -itd --name carla_v1 --net carla-net erdosproject/carla /bin/bash
    nvidia-docker run -itd --name pylot_v1 -p 20022:22 --net carla-net erdosproject/pylot /bin/bash


Following, start the simulator in the CARLA container:

.. code-block:: bash

    nvidia-docker exec -i -t carla_v1 /bin/bash
    SDL_VIDEODRIVER=offscreen /home/erdos/workspace/CARLA_0.9.6/CarlaUE4.sh -opengl -windowed -carla-server -benchmark -fps=10 -quality-level=Epic

Finally, start Pylot in the container:

.. code-block:: bash

    nvidia-docker exec -i -t pylot_v1 /bin/bash
    cd workspace/pylot/
    python3 pylot.py --flagfile=configs/detection.conf --carla_host=carla_v1


More Information
----------------

Slides
~~~~~~
- `Talk given at RISE Retreat 2020 <https://drive.google.com/open?id=1i_Wl27ovVRzdz5UgozjoHgYgHL3HOmTd>`_

Getting Involved
----------------
- `GitHub Issues`_: For reporting bugs and feature requests.
- `Pull Requests`_: For submitting code contributions.

.. _`GitHub Issues`: https://github.com/erdos-project/pylot/issues
.. _`Pull Requests`: https://github.com/erdos-project/pylot/pulls


.. toctree::
   :maxdepth: -1
   :caption: Components

   pylot.control.rst
   pylot.debug.rst
   pylot.loggers.rst
   pylot.map.rst
   pylot.perception.rst
   pylot.planning.rst
   pylot.prediction.rst
   pylot.simulation.rst

Submodules
----------

pylot.config module
-------------------

pylot.component\_creator module
-------------------------------

.. automodule:: pylot.component_creator
    :members:
    :undoc-members:
    :show-inheritance:

pylot.utils module
------------------

.. automodule:: pylot.utils
    :members:
    :undoc-members:
    :show-inheritance:


Module contents
---------------

.. automodule:: pylot
    :members:
    :undoc-members:
    :show-inheritance:
