import cv2

from pylot.perception.detection.utils import BoundingBox2D, DetectedObstacle
from pylot.perception.tracking.multi_object_tracker import MultiObjectTracker


class MultiObjectCV2Tracker(MultiObjectTracker):
    def __init__(self, flags):
        self._tracker = cv2.MultiTracker_create()

    def reinitialize(self, frame, obstacles):
        """ Reinitializes a multiple obstacle tracker.

        Args:
            frame: perception.camera_frame.CameraFrame to reinitialize with.
            obstacles: List of perception.detection.utils.DetectedObstacle.
        """
        self._tracker = cv2.MultiTracker_create()
        for obstacle in obstacles:
            self._tracker.add(cv2.TrackerKCF_create(), frame.frame,
                              obstacle.as_width_height_bbox())
            # self._tracker.add(cv2.TrackerMOSSE_create(), frame.frame, bbox)

    def track(self, frame):
        """ Tracks obstacles in a frame.

        Args:
            frame: perception.camera_frame.CameraFrame to track in.
        """
        ok, bboxes = self._tracker.update(frame.frame)
        if not ok:
            return False, []
        obstacles = []
        for (xmin, ymin, w, h) in bboxes:
            obstacles.append(
                DetectedObstacle(BoundingBox2D(xmin, xmin + w, ymin, ymin + h),
                                 "", 0))
        return True, obstacles
