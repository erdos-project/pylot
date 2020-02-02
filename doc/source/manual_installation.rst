Installation guide
==================

You can install Pylot on your base system by executing the following commands:

.. code-block:: bash

    ./install.sh
    pip install -e ./

Next, start the simulator:

.. code-block:: bash
                
    export CARLA_HOME=$PYLOT_HOME/dependencies/CARLA_0.9.6/
    ./scripts/run_simulator.sh

In a different terminal, setup the paths:

.. code-block:: bash

    export CARLA_HOME=$PYLOT_HOME/dependencies/CARLA_0.9.6/
    cd $PYLOT_HOME/scripts/
    source ./set_pythonpath.sh

Finally, run Pylot:

.. code-block:: bash

    cd  $PYLOT_HOME/
    python3 pylot.py --flagfile=configs/detection.conf
