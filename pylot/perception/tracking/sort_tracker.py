import numpy as np

from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.detection.utils import BoundingBox2D
from pylot.perception.tracking.multi_object_tracker import MultiObjectTracker

from sort.sort import Sort


class MultiObjectSORTTracker(MultiObjectTracker):
    def __init__(self, flags, logger):
        self._logger = logger
        self.tracker = Sort(max_age=flags.obstacle_track_max_age,
                            min_hits=1,
                            min_iou=flags.min_matching_iou)

    def reinitialize(self, frame, obstacles):
        """ Reinitializes a multiple obstacle tracker.

        Args:
            frame (:py:class:`~pylot.perception.camera_frame.CameraFrame`):
                Frame to reinitialize with.
            obstacles : List of perception.detection.obstacle.Obstacle.
        """
        detections, labels, ids = self.convert_detections_for_sort_alg(
            obstacles)
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
            xmin = int(coords[0])
            xmax = int(coords[2])
            ymin = int(coords[1])
            ymax = int(coords[3])
            if xmin < xmax and ymin < ymax:
                bbox = BoundingBox2D(xmin, xmax, ymin, ymax)
                obstacles.append(Obstacle(bbox, 0, track.label, track.id))
            else:
                self._logger.error(
                    "Tracker found invalid bounding box {} {} {} {}".format(
                        xmin, xmax, ymin, ymax))
        return True, obstacles

    def convert_detections_for_sort_alg(self, obstacles):
        converted_detections = []
        labels = []
        ids = []
        for obstacle in obstacles:
            bbox = [
                obstacle.bounding_box_2D.x_min, obstacle.bounding_box_2D.y_min,
                obstacle.bounding_box_2D.x_max, obstacle.bounding_box_2D.y_max,
                obstacle.confidence
            ]
            converted_detections.append(bbox)
            labels.append(obstacle.label)
            ids.append(obstacle.id)
        return (np.array(converted_detections), labels, ids)
