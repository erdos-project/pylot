Detection
=========

The package provides operators and classes useful for detecting obstacles,
traffic lights and lanes. It provides operators that use trained models and
algorithms, as well as operators that use data from the simulator to perfectly
detect obstacles, traffic lights, and lanes.

Obstacle detection
------------------

Pylot provides two options for obstacle detection:

  1. An obstacle detection operator that can use any model that adheres to the
     Tensorflow `object detection model zoo <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md>`__.
     By default, we provide three models that were trained on 1080p CARLA
     images (``faster-rcnn``, ``ssd-mobilenet-fpn-640``, and
     ``ssdlit-mobilenet-v2``), but models that have been trained on other data
     sets can be easily plugged in by changing the
     ``--obstacle_detection_model_paths`` flag.
  2. An operator that can infer any of the
     `EfficientDet <https://github.com/google/automl>`__ models. The
     EfficientDet models we use are not trained on CARLA data, but on the COCO
     data set.

To see a demo of obstacle detection run:

.. code-block:: bash

    python3 pylot.py --flagfile=configs/detection.conf

.. image:: images/pylot-obstacle-detection.png
     :align: center

Traffic light detection
-----------------------

The traffic light detection component uses Faster RCNN weight, which have been
trained on 1080p CARLA images.

To see a demo of traffic light detection run:

.. code-block:: bash

    python3 pylot.py --flagfile=configs/traffic_light.conf

.. image:: images/pylot-traffic-light-detection.png
     :align: center

Lane detection
--------------

To see a demo of lane detection run:

.. code-block:: bash

    python3 pylot.py --flagfile=configs/lane_detection.conf

.. image:: images/pylot-lane-detection.png
     :align: center

**Warning**: Our lane detection component works optimally with frames have a
resolution of 1280x720.

Important flags
---------------

- ``--obstacle_detection``: Enables the obstacle detection component of the
  stack. Depending on accuracy and runtime requirements, the component can use
  different obstacle detection models. Pylot currently offers three trained
  models: ``faster-rcnn``, ``ssd-mobilenet-fpn-640``, and
  ``ssdlit-mobilenet-v2``.
- ``--perfect_obstacle_detection``: Enables the component to use an obstacle
  detector which perfectly detects obstacles using ground information from
  the simulator.
- ``--simulator_obstacle_detection``: The component outputs obstacle info that
  is obtained directly from the simulator.
- ``--evaluate_obstacle_detection``: Compute and log accuracy metrics of the
  obstacle detection component.
- ``--visualize_detected_obstacles``: Enables visualization of detected
  obstacles.
- ``--traffic_light_detection``: Enables the traffic light detection component
  of the stack. This component attaches to the ego vehicle a forward facing
  camera with a narrow field of view, which the component uses to detect traffic
  lights.
- ``--perfect_traffic_ligth_detection``: Enables the component to use a traffic
  light detector which perfectly detects traffic lights using ground information
  from the simulator.
- ``--simulator_traffic_light_detection``: The component outputs traffic light
  ifo that is obtained directly from the simulator.
- ``--visualize_detected_traffic_lights``: Enables visualization of detected
  traffic lights.
- ``--lane_detection``: Enables the lane detection component, which currently
  implements a simple Canny edge detector.
- ``--lane_detection_type``: Specifies which lane detection solution to use.
  Pylot current supports a standard vision implementation that uses *Canny edge*,
  and a neural network-based implementation that uses *Lanenet*.
- ``--perfect_lane_detection``: Enables the component to perfectly detect lanes
  using information from the simulator.
- ``--visualize_lane_detection``: Enables visualization of detected lanes.

More information
----------------
See the `reference <pylot.perception.detection.html>`_ for more information.  
