from dependencies.conv_reg_vot.simgeo import Rect
import dependencies.conv_reg_vot.tracker as tracker

from perception.tracking.multi_object_tracker import MultiObjectTracker


class SingleObjectCRVTracker(object):
    def __init__(self, frame, bbox):
        width = bbox[1] - bbox[0]
        height = bbox[3] - bbox[2]
        rect_coordinates = (int(bbox[0]), int(bbox[2]), int(width),
                            int(height))
        # TODO(ionel): Check if calling init is sufficient to
        # reinitialize the tracker.
        self._tracker = tracker.ConvRegTracker()
        self._tracker.init(frame, Rect(*rect_coordinates))

    def track(self, frame):
        bb_pred = self._tracker.track(frame)
        xmax = bb_pred.x + bb_pred.w
        ymax = bb_pred.y + bb_pred.h
        bbox = (bb_pred.x, xmax, bb_pred.y, ymax)
        return bbox


class MultiObjectCRVTracker(MultiObjectTracker):
    def __init__(self, flags):
        pass

    def reinitialize(self, frame, bboxes):
        # Create a tracker for each bbox.
        self._trackers = []
        for bbox in bboxes:
            self._trackers.append(SingleObjectCRVTracker(frame, bbox))
