## Latest

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
