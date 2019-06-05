import cv2

from pylot.perception.tracking.multi_object_tracker import MultiObjectTracker


class MultiObjectCV2Tracker(MultiObjectTracker):
    def __init__(self, flags):
        self._tracker = cv2.MultiTracker_create()

    def reinitialize(self, frame, bboxes):
        self._tracker = cv2.MultiTracker_create()
        for (xmin, xmax, ymin, ymax) in bboxes:
            bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
            self._tracker.add(cv2.TrackerKCF_create(), frame, bbox)
            #self._tracker.add(cv2.TrackerMOSSE_create(), frame, bbox)

    def track(self, frame):
        ok, bboxes = self._tracker.update(frame)
        if not ok:
            return False, []
        corners = []
        for (xmin, ymin, w, h) in bboxes:
            corners.append((xmin, xmin + w, ymin, ymin + h))
        return True, corners
