Depth estimation
================

The package provides operators for estimating depth using cameras. It
currently offers a DepthEstimationOperator, which implements stereo
depth estimation using the `AnyNet <https://github.com/mileyan/AnyNet>`_ neural
network.

Execute the following command to run a depth estimation demo:

.. code-block:: bash

    python3 pylot.py --flagfile=configs/depth_estimation.conf

Important flags
---------------

- ``--depth_estimation``: Enables stereo depth estimation.
- ``--depth_estimation_model_path``: File path to a trained Anytime network
  model.
- ``--perfect_depth_estimation``: The component outputs frames with perfect
  depth values. This frames are obtained from the simulator.
- ``--offset_left_right_cameras``: Offset distance (in meteres) between the left
  and right cameras used for depth estimation.
- ``--visualize_depth_camera``: Enables visualization of the sensor depth
  camera.

More information
----------------
See the `reference <pylot.perception.depth_estimation.html>`_ for more information.
