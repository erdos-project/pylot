from nanonets_object_tracking.deepsort import deepsort_rbc

from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.detection.utils import BoundingBox2D
from pylot.perception.tracking.multi_object_tracker import MultiObjectTracker


class MultiObjectDeepSORTTracker(MultiObjectTracker):
    def __init__(self, flags, logger):
        self._logger = logger
        # Initialize the deepsort object, which has a tracker object within it
        self._deepsort = deepsort_rbc(
            wt_path=flags.deep_sort_tracker_weights_path,
            max_age=flags.obstacle_track_max_age,
            min_iou=flags.min_matching_iou)

    def reinitialize(self, frame, obstacles):
        """ Reinitializes a multiple obstacle tracker.

        Args:
            frame (:py:class:`~pylot.perception.camera_frame.CameraFrame`):
                Frame to reinitialize with.
            obstacles: List of perception.detection.obstacle.Obstacle.
        """
        # update tracker with new detections
        self.track(frame, obstacles)

    def track(self, frame, obstacles=[]):
        """ Tracks obstacles in a frame.

        Args:
            frame (:py:class:`~pylot.perception.camera_frame.CameraFrame`):
                Frame to track in.
        """
        # If obstacles, run deep sort to update tracker with detections.
        # Otherwise, step each confirmed track one step forward.
        if obstacles:
            bboxes, labels, confidence_scores, ids = [], [], [], []
            for obstacle in obstacles:
                bboxes.append(obstacle.bounding_box_2D.as_width_height_bbox())
                labels.append(obstacle.label)
                confidence_scores.append(obstacle.confidence)
                ids.append(obstacle.id)
            self._deepsort.run_deep_sort(frame.frame, confidence_scores,
                                         bboxes, labels, ids)
        else:
            for track in self._deepsort.tracker.tracks:
                if track.is_confirmed():
                    track.predict(self._deepsort.tracker.kf)
        tracked_obstacles = []
        for track in self._deepsort.tracker.tracks:
            if track.is_confirmed():
                # Converts x, y, w, h bbox to tlbr bbox (top left and bottom
                # right coords).
                bbox = track.to_tlbr()
                # Converts to xmin, xmax, ymin, ymax format.
                xmin = int(bbox[0])
                xmax = int(bbox[2])
                ymin = int(bbox[1])
                ymax = int(bbox[3])
                if xmin < xmax and ymin < ymax:
                    bbox = BoundingBox2D(xmin, xmax, ymin, ymax)
                    tracked_obstacles.append(
                        Obstacle(bbox, 0, track.label, track.track_id))
                else:
                    self._logger.error(
                        "Tracker found invalid bounding box {} {} {} {}".
                        format(xmin, xmax, ymin, ymax))
        return True, tracked_obstacles
