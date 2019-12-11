import numpy as np
import torch

from nanonets_object_tracking.deepsort import deepsort_rbc
from pylot.perception.detection.utils import DetectedObject
from pylot.perception.tracking.multi_object_tracker import MultiObjectTracker


class MultiObjectDeepSORTTracker(MultiObjectTracker):
    def __init__(self, flags, logger):
        # Initialize the deepsort object, which has a tracker object within it
        self._deepsort = deepsort_rbc(wt_path=flags.deep_sort_tracker_pedestrian_weights_path)
        self.tracker = None

    def reinitialize(self, frame, bboxes, confidence_scores, ids):
        # update tracker with new detections
        self.track(frame, confidence_scores=confidence_scores, bboxes=bboxes)

    def track(self, frame, confidence_scores=None, bboxes=None):
        if bboxes:
            detections = self.convert_detections_for_deepsort_alg(bboxes)
            self.tracker, detections_class = self._deepsort.run_deep_sort(frame, confidence_scores, detections)
        if self.tracker:
            tracked_objects = []
            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr() # converts x, y, w, h bbox to tlbr bbox (top left and bottom right coords)
                corners = (int(bbox[0]), int(bbox[2]), int(bbox[1]), int(bbox[3])) # converts to xmin, xmax, ymin, ymax format
                tracked_objects.append(DetectedObject(corners, 0, "", track.track_id))
            return True, tracked_objects
        return False, []

    def convert_detections_for_deepsort_alg(self, bboxes):
        converted_bboxes = []
        for bbox in bboxes:
            x1, x2, y1, y2 = bbox # comes from detected objects self.corners (see detection utils)
            bbox_x, bbox_y = min(x1, x2), min(y1, y2)
            bbox_w, bbox_h = abs(x2 - x1), abs(y2 - y1)
            converted_bboxes.append([bbox_x, bbox_y, bbox_w, bbox_h])
        return converted_bboxes

