How to collect data
===================

Pylot also provides a script for collecting CARLA data such as: RGB images,
segmented images, obstacle 2D bounding boxes, depth frames, point clouds,
traffic lights, obstacle trajectories, and data in Chauffeur format.

Run the following command to see what data you can collect, and which flags you
must set:

.. code-block:: bash

    python3 data_gatherer.py --help

Alternatively, you can inspect
`data_gatherer.conf <https://github.com/erdos-project/pylot/blob/master/configs/data_gatherer.conf>`_
for an example of a data collection setup.
