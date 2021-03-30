## Latest

## Pylot 0.3.2
  * Fixed bug in parsing CARLA version strings.
  * Fixed EfficientDet detection operator so that it correctly extracts bounding boxes.
  * Added indicator stream to logging operators so that they can be used for synchronizing.
  * Added CenterTrack obstacle tracker.
  * Fixed LaneNet imports.
  * Lock pygame to a working version.
  * Added code to handle top watermarks in watermark callbacks.
  * Added new tracker evaluation operator, and base eval class.
  * Added support for CARLA 0.9.11.
  * Added option to enable evaluation operators to the challenge agent.

## Pylot 0.3.1
  * Improved perfect lane detector to collect all lanes.
  * Added support for latest CARLA lidar type (CARLA >= 0.9.9.4).
  * Updated CARLA challenge agent to support perfect perception.
  * Added logic to close pygame window, and to shutdown on sigint.
  * Updated traffic light data collection script to work with latest CARLA versions.
  * Reduced dependency on CARLA throughout the code base.

## Pylot 0.3

  * Refactored planners to work with a common world representation.
  * Added EKF localization.
  * Added R2P2 prediction.
  * Added Lanenet lane detection.
  * Added EfficientDet obstacle detection.
  * Added driver script for running on a Lincoln MKZ vehicle.
  * Added drivers for Grasshopper cameras and Velodyne LiDAR.
  * Added pseudo-asynchronous execution mode for accuratly measuring the impact of runtime on driving experience.
  * Moved from cv2 visualization to a pygame-based interface for visualizing all the components.
  * Added an agent for the CARLA challenge.

## Pylot 0.2

  * Added support for CARLA 0.9.7
  * Refactored utils modules into classes.
  * Ensured all transforms from CARLA objects to Pylot objects are consistent.
