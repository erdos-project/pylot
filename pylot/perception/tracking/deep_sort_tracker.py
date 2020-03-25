from absl import flags

from nanonets_object_tracking.deepsort import deepsort_rbc

from pylot.perception.detection.utils import BoundingBox2D, DetectedObstacle
from pylot.perception.tracking.multi_object_tracker import MultiObjectTracker

flags.DEFINE_string(
    'deep_sort_tracker_person_weights_path',
    'dependencies/models/tracking/deep-sort-carla/ped_feature_extractor',
    'Path to weights for person feature extractor model')


class MultiObjectDeepSORTTracker(MultiObjectTracker):
    def __init__(self, flags, logger):
        # Initialize the deepsort object, which has a tracker object within it
        self._deepsort = deepsort_rbc(
            wt_path=flags.deep_sort_tracker_person_weights_path)
        self.tracker = None

    def reinitialize(self, frame, obstacles):
        """ Reinitializes a multiple obstacle tracker.

        Args:
            frame (:py:class:`~pylot.perception.camera_frame.CameraFrame`):
                Frame to reinitialize with.
            obstacles: List of perception.detection.utils.DetectedObstacle.
        """
        # update tracker with new detections
        self.track(frame, obstacles)

    def track(self, frame, obstacles=None):
        """ Tracks obstacles in a frame.

        Args:
            frame (:py:class:`~pylot.perception.camera_frame.CameraFrame`):
                Frame to track in.
        """
        if obstacles:
            bboxes = [
                obstacle.bounding_box.as_width_height_bbox()
                for obstacle in obstacles
            ]
            confidence_scores = [obstacle.confidence for obstacle in obstacles]
            self.tracker, detections_class = self._deepsort.run_deep_sort(
                frame.frame, confidence_scores, bboxes)
        if self.tracker:
            obstacles = []
            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                # Converts x, y, w, h bbox to tlbr bbox (top left and bottom
                # right coords).
                bbox = track.to_tlbr()
                # Converts to xmin, xmax, ymin, ymax format.
                obstacles.append(
                    DetectedObstacle(
                        BoundingBox2D(int(bbox[0]), int(bbox[2]), int(bbox[1]),
                                      int(bbox[3])), 0, "", track.track_id))
            return True, obstacles
        return False, []
