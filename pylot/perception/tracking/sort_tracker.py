import numpy as np

from sort.sort import *

from pylot.perception.detection.utils import BoundingBox2D, DetectedObstacle
from pylot.perception.tracking.multi_object_tracker import MultiObjectTracker


class MultiObjectSORTTracker(MultiObjectTracker):
    def __init__(self, flags):
        self.tracker = Sort()

    def reinitialize(self, frame, obstacles):
        """ Reinitializes a multiple obstacle tracker.

        Args:
            frame (:py:class:`~pylot.perception.camera_frame.CameraFrame`):
                Frame to reinitialize with.
            obstacles : List of perception.detection.utils.DetectedObstacle.
        """
        detections, labels, ids = self.convert_detections_for_sort_alg(obstacles)
        self.tracker.update(detections, labels, ids)

    def track(self, frame):
        """ Tracks obstacles in a frame.

        Args:
            frame (:py:class:`~pylot.perception.camera_frame.CameraFrame`):
                Frame to track in.
        """
        # each track in tracks has format ([xmin, ymin, xmax, ymax], id)
        obstacles = []
        for track in self.tracker.trackers:
            coords = track.predict()[0].tolist()
            # changing to xmin, xmax, ymin, ymax format
            bbox = BoundingBox2D(int(coords[0]), int(coords[2]),
                                 int(coords[1]), int(coords[3]))
            obstacles.append(DetectedObstacle(bbox, 0, track.label, track.id))
        return True, obstacles

    def convert_detections_for_sort_alg(self, obstacles):
        converted_detections = []
        labels = []
        ids = []
        for obstacle in obstacles:
            bbox = [
                obstacle.bounding_box.x_min, obstacle.bounding_box.y_min,
                obstacle.bounding_box.x_max, obstacle.bounding_box.y_max,
                obstacle.confidence
            ]
            converted_detections.append(bbox)
            labels.append(obstacle.label)
            ids.append(obstacle.id)
        return (np.array(converted_detections), labels, ids)
