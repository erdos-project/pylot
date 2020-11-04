Obstacle tracking
=================

The package provides operators and classes useful for tracking obstacles
across frames:

- `ObjectTrackerOperator <pylot.perception.tracking.html#module-pylot.perception.tracking.object\_tracker\_operator>`__
  is the operator that provides four options for tracking obstacles:

   1. ``da_siam_rpn``: a high-quality DaSiamRPN network single obstacle tracker,
      which Pylot repurposed to track serially track multiple obstacles.
   2. ``sort``: uses a simple combination of Kalman Filter and Hungarian
      algorithm for tracking and matching (see `SORT <https://github.com/ICGog/sort>`_).
   3. ``deep_sort``: An extended version of SORT that integrates detection and
      appearance features (see `Deep SORT <https://github.com/ICGog/nanonets_object_tracking>`_).

- `ObstacleTrajectory <pylot.perception.tracking.html#module-pylot.perception.tracking.obstacle\_trajectory>`__
  is used to store the trajectories of the tracked obstacles.
- `MultiObjectTracker <pylot.perception.tracking.html#module-pylot.perception.tracking.multi\_object\_tracker>`__
  is an interfaces which must be implemented by multiple obstacles trackers.
- `MultiObjectDaSiamRPNTracker <pylot.perception.tracking.html#module-pylot.perception.tracking.da\_siam\_rpn\_tracker>`__
  is a class that implements a multiple obstacle tracker using the DASiamRPN
  neural network for single obstacle tracking. The class executes model
  inference for every single obstacle, and matches obstacles across frames using
  the Hungarian algorithm for bipartite graph matching.
- `MultiObjectDeepSORTTracker <pylot.perception.tracking.html#module-pylot.perception.tracking.deep\_sort\_tracker>`__
  is a wrapper class around the DeepSORT multi obstacle tracker. It executes
  the DeepSORT neural network on every frame.
- `MultiObjectSORTTracker <pylot.perception.tracking.html#module-pylot.perception.tracking.sort\_tracker>`__
  is wrapper class around the SORT tracker.

Execute the following command to run an obstacle tracking demo:

.. code-block:: bash

    python3 pylot.py --flagfile=configs/tracking.conf

.. image:: images/pylot-obstacle-detection.png
     :align: center
    
Important flags
---------------

- ``--obstacle_tracking``: Enables the obstacle tracking component of the stack.
- ``--tracker_type``: Sets which obstacle tracker the component use.
- ``--perfect_obstacle_tracking``: Enables the component to perfectly track
  obstacles using information it receives from the simulator (only works in
  simulation).
- ``--visualize_tracked_obstacles``: Enables visualization of tracked obstacles.

- ``--tracking_num_steps``: Limit on the number of past bounding boxes to track.
- ``--min_matching_iou``: Sets the minimum intersetion over union (IoU) two
  bounding boxes must have for the tracker matching state to consider them.

More information
----------------
See the `reference <pylot.perception.tracking.html>`_ for more information.
