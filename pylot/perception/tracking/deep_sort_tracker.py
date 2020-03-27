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
        bboxes, labels, confidence_scores = [], [], []
        for obstacle in obstacles:
            bboxes.append(obstacle.bounding_box.as_width_height_bbox())
            labels.append(obstacle.label)
            confidence_scores.append(obstacle.confidence)
        # TODO: Remove this if-check and figure out why it breaks the deepsort
        # matching. If removed, calling tracker.predict() without an update
        # messes up tracker ages and matches are no longer made, just new tracks
        # each time step
        if obstacles:
            # If obstacles is empty, this should move existing tracks one step
            # ahead. Otherwise, it will update trackers with detections.
            self._deepsort.run_deep_sort(
                frame.frame, confidence_scores, bboxes, labels)
        tracked_obstacles = []
        for track in self._deepsort.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            # Converts x, y, w, h bbox to tlbr bbox (top left and bottom
            # right coords).
            bbox = track.to_tlbr()
            # Converts to xmin, xmax, ymin, ymax format.
            bbox_2d = BoundingBox2D(
                int(bbox[0]), int(bbox[2]), int(bbox[1]), int(bbox[3]))
            tracked_obstacles.append(
                DetectedObstacle(bbox_2d, 0, track.label, track.track_id))
        return True, tracked_obstacles
