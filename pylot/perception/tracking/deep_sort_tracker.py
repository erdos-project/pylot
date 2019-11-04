import numpy as np
import torch

from nanonets_object_tracking.deepsort import deepsort_rbc
from pylot.perception.tracking.multi_object_tracker import MultiObjectTracker


class MultiObjectDeepSORTTracker(MultiObjectTracker):
    def __init__(self, flags, logger):
        # Initialize the deepsort object, which has a tracker object within it
        self._deepsort = deepsort_rbc(wt_path="/home/erdos/workspace/forks/pylot/pylot/perception/tracking/nanonets_object_tracking/ped_feature_extractor")
        self.tracker = None

    def reinitialize(self, frame, bboxes, confidence_scores):
        detections = self.convert_detections_for_deepsort_alg(bboxes)
        self.track(frame, confidence_scores=confidence_scores, detections=detections)

    def track(self, frame, confidence_scores=None, detections=None):
        if detections:
            #detections = self.convert_detections_for_deepsort_alg(detections)
            self.tracker, detections_class = self._deepsort.run_deep_sort(frame, confidence_scores, detections)
        if self.tracker:
            bboxes = []
            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr() # converts x, y, w, h bbox to tlbr bbox (top left and bottom right coords)
                id_num = str(track.track_id)
                bboxes.append((bbox, id_num))
            return True, bboxes
        return False, []

    def convert_detections_for_deepsort_alg(self, bboxes):
        converted_bboxes = []
        for bbox in bboxes:
            x1, x2, y1, y2 = bbox # comes from detected objects self.corners (see detection utils)
            bbox_x, bbox_y = min(x1, x2), min(y1, y2)
            bbox_w, bbox_h = abs(x2 - x1), abs(y2 - y1)
            converted_bboxes.append([bbox_x, bbox_y, bbox_w, bbox_h])
        return converted_bboxes

