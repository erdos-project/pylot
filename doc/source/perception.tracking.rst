Obstacle tracking
=================

The package provides operators and classes useful for tracking obstacles
across frames:

- `ObjectTrackerOperator <pylot.perception.tracking.html#module-pylot.perception.tracking.object\_tracker\_operator>`__
  is the operator that provides four options for tracking obstacles:

   1. ``da_siam_rpn``: a high-quality DaSiamRPN network single obstacle tracker,
      which Pylot repurposed to track serially track multiple obstacles.
   2. ``cv2``: a tracker that uses Kalman filters.
   3. ``sort``: uses a simple combination of Kalman Filter and Hungarian
      algorithm for tracking and matching (see `SORT <https://github.com/ICGog/sort>`_).
   4. ``deep_sort``: An extended version of SORT that integrates detection and
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
- `MultiObjectCV2Tracker <pylot.perception.tracking.html#module-pylot.perception.tracking.cv2\_tracker>`__
  implements a Kalman-Filter tracker using the CV2 Kalman-Filter implemenation.

Execute the following command to run an obstacle tracking demo:

.. code-block:: bash

    python3 pylot.py --flagfile=configs/tracking.conf

Important flags:

- ``--obstacle_tracking``: Enables the obstacle tracking component of the stack.
- ``--tracker_type``: Sets which obstacle tracker the component use.
- ``--perfect_obstacle_tracking``: Enables the component to perfectly track
  obstacles using information it receives from CARLA.
- ``--visualize_tracker_output``: Enables visualization of the tracked
  obstacles.
