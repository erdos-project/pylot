Planning
========

The package provides operators and classes useful for planning the trajectory
the ego vehicle must follow.

Execute the following command to run a planning demo:

.. code-block:: bash

    python3 pylot.py --flagfile=configs/rrt_star_planner.conf

Important flags:

- ``--planning_type``: Sets which planner to use. Pylot currentlys offers two
  alternatives: `waypoint planning <pylot.planning.html#module-pylot.planning.waypoint\_planning\_operator>`_
  and `rrt_star planning <pylot.planning.rrt_star.html#module-pylot.planning.rrt\_star.rrt\_star\_planning\_operator>`_.
- ``--visualize_waypoints``: Enables visualization of the waypoints computed
  by the planning operators.
- ``--draw_waypoints_on_world``: Enables drawing of waypoints directly in the
  CARLA simulator.
- ``--draw_waypoints_on_camera_frames``: Enables drawing of waypoints on camera
  frames.
