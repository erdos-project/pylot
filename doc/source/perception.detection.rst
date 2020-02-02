Detection
=========

The package provides operators and classes useful for detecting obstacles,
traffic lights and lanes. It provides operators that use trained models and
algorithms, as well as operators that use data from CARLA to perfectly
detect obstacles, traffic lights, and lanes.

To see a demove of obstacle detection run:

.. code-block:: bash

    python3 pylot.py --flagfile=configs/detection.conf

To see a demove of traffic light detection run:

.. code-block:: bash

    python3 pylot.py --flagfile=configs/traffic_light.conf

To see a demove of lane detection run:

.. code-block:: bash

    python3 pylot.py --flagfile=configs/lane_detection.conf


Important flags:

- ``--obstacle_detection``: Enables the obstacle detection component of the
  stack. Depending on accuracy and runtime requirements, the component can use
  different obstacle detection models. Pylot currently offers three trained
  models: ``faster-rcnn``, ``ssd-mobilenet-fpn-640``, and
  ``ssdlit-mobilenet-v2``.
- ``--perfect_obstacle_detection``: Enables the component to use an obstacle
  detector which perfectly detects obstacles using ground information from
  CARLA.
- ``--carla_obstacle_detection``: The component outputs obstacle info that is
  obtained directly from CARLA.
- ``--evaluate_obstacle_detection``: Compute and log accuracy metrics of the
  obstacle detection component.
- ``--detection_metric``: Sets the accuracy metric to compute and log.
- ``--visualize_detected_obstacles``: Enables visualization of detected
  obstacles.
- ``--traffic_light_detection``: Enables the traffic light detection component
  of the stack. This component attaches to the ego vehicle a forward facing
  camera with a narrow field of view, which the component uses to detect traffic
  lights.
- ``--perfect_traffic_ligth_detection``: Enables the component to use a traffic
  light detector which perfectly detects traffic lights using ground information
  from CARLA.
- ``--carla_traffic_light_detection``: The component outputs traffic light info
  that is obtained directly from CARLA.
- ``--visualize_detected_traffic_lights``: Enables visualization of detected
  traffic lights.
- ``--lane_detection``: Enables the lane detection component, which currently
  implements a simple Canny edge detector.
- ``--perfect_lane_detection``: Enables the component to perfectly detect lanes
  using information from CARLA.
- ``--visualize_lane_detection``: Enables visualization of detected lanes.
