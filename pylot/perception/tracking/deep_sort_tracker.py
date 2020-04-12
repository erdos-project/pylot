from nanonets_object_tracking.deepsort import deepsort_rbc

from pylot.perception.detection.utils import BoundingBox2D, DetectedObstacle
from pylot.perception.tracking.multi_object_tracker import MultiObjectTracker


class MultiObjectDeepSORTTracker(MultiObjectTracker):
    def __init__(self, flags, logger):
        # Initialize the deepsort object, which has a tracker object within it
        self._deepsort = deepsort_rbc(
            wt_path=flags.deep_sort_tracker_weights_path,
            min_iou=flags.min_matching_iou)

    def reinitialize(self, frame, obstacles):
        """ Reinitializes a multiple obstacle tracker.

        Args:
            frame (:py:class:`~pylot.perception.camera_frame.CameraFrame`):
                Frame to reinitialize with.
            obstacles: List of perception.detection.utils.DetectedObstacle.
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
                bboxes.append(obstacle.bounding_box.as_width_height_bbox())
                labels.append(obstacle.label)
                confidence_scores.append(obstacle.confidence)
                ids.append(obstacle.id)
                self._deepsort.run_deep_sort(
                    frame.frame, confidence_scores, bboxes, labels, ids)
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
                bbox_2d = BoundingBox2D(int(bbox[0]), int(bbox[2]), int(bbox[1]),
                                        int(bbox[3]))
                tracked_obstacles.append(
                    DetectedObstacle(bbox_2d, 0, track.label, track.track_id))
        return True, tracked_obstacles
