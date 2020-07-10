Quick start
===========

The easiest way to run Pylot is to use our Docker image. First, please ensure
you have `nvidia-docker` on your machine before you start installing Pylot.
In case you do not have `nvidia-docker` please
run ``./scripts/install-nvidia-docker.sh``.

We provide a Docker image with both Pylot and CARLA already setup.

.. code-block:: bash

    docker pull erdosproject/pylot
    nvidia-docker run -itd --name pylot -p 20022:22 erdosproject/pylot /bin/bash

Following, start the simulator in the container:    

.. code-block:: bash

    nvidia-docker exec -i -t pylot /home/erdos/workspace/pylot/scripts/run_simulator.sh

Finally, start Pylot in the container:

.. code-block:: bash

    nvidia-docker exec -i -t pylot /bin/bash
    cd workspace/pylot/
    python3 pylot.py --flagfile=configs/detection.conf

In case you desire to visualize outputs of different components (e.g., bounding boxes),
you have to forward X from the container. First, add your public ssh key to the
``~/.ssh/authorized_keys`` in the container:

.. code-block:: bash

    nvidia-docker cp ~/.ssh/id_rsa.pub pylot_new:/home/erdos/.ssh/authorized_keys
    nvidia-docker exec -i -t pylot_new sudo chown erdos /home/erdos/.ssh/authorized_keys
    nvidia-docker exec -i -t pylot /bin/bash
    sudo service ssh start
    exit

Finally, ssh into the container with X forwarding:

.. code-block:: bash

    ssh -p 20022 -X erdos@localhost /bin/bash
    cd /home/erdos/workspace/pylot/
    python3 pylot.py --flagfile=configs/detection.conf --visualize_detected_obstacles

If everything worked ok, you should be able to see a window that visualizes
detected obstacles like the one below:

.. image:: images/pylot-obstacle-detection.png
     :align: center
