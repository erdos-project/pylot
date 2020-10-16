Control
=======

The package provides operators and classes useful for controlling the ego
vehicle. These operators ensure that the vehicle closely follows a sequence
of waypoints sent by the `planning <planning.html>`_ component.

Execute the following command to run a demo of the MPC controller:

.. code-block:: bash

    python3 pylot.py --flagfile=configs/mpc_agent.conf


Execute the following command to run a demo using solely the PID controller:

.. code-block:: bash

    python3 pylot.py --flagfile=configs/e2e.conf

Important flags
---------------

- ``--control``: Sets which control algorithm to use: Pylot currently
  offers three alternatives:

  1. `mpc <pylot.control.mpc.html#module-pylot.control.mpc.mpc\_agent\_operator>`__:
     An operator that implements a model predictive controller.
  2. `pid <pylot.control.html#module-pylot.control.pid\_agent\_operator>`__:
     An operator that uses a PID controller to follow the waypoints.
  3. ``simulator_auto_pilot``: The simulator controls the ego-vehicle, and
     drives it on a predefined path. The path differs depending on the spawning
     position.

- ``--pid_p``: Sets the p parameter of the PID controller.
- ``--pid_i``: Sets the i parameter of the PID controller.
- ``--pid_d``: Sets the d parameter of the PID controller.


More information
----------------
See the `reference <pylot.control.html>`_ for more information.
