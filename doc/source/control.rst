Control
=======

The package provides operators and classes useful for controlling the ego
vehicle. These operators ensure that the vehicle closely follows a sequence
of waypoints sent by the `planning <planning.html>`_ component.

Execute the following command to run a demo of the MPC controller:

.. code-block:: bash

    python3 pylot.py --flagfile=configs/mpc_agent.conf


Execute the following command to run a demo of the Pylot agent, which uses a
PID controller:

.. code-block:: bash

    python3 pylot.py --flagfile=configs/pylot_agent_e2e.conf

Important flags:

- ``--control_agent``: Sets which control algorithm to use: Pylot currently
  offers two alternatives:

  1. `mpc <pylot.control.mpc.html#module-pylot.control.mpc.mpc\_agent\_operator>`__:
     An operator that implements a model predictive controller.
  2. `pylot <pylot.control.html#module-pylot.control.pylot\_agent\_operator>`__:
     An operator that uses a PID controller to follow the waypoints.

- ``--pid_p``: Sets the p parameter of the PID controller.
- ``--pid_i``: Sets the i parameter of the PID controller.
- ``--pid_d``: Sets the d parameter of the PID controller.
