import numpy as np
import torch

from sort.sort import *
from pylot.perception.detection.utils import DetectedObject
from pylot.perception.tracking.multi_object_tracker import MultiObjectTracker


class MultiObjectSORTTracker(MultiObjectTracker):
    def __init__(self, flags):
        self.tracker = Sort()

    def reinitialize(self, frame, bboxes, confidence_scores):
        detections = self.convert_detections_for_sort_alg(bboxes, confidence_scores)
        self.tracker.update(detections)

    def track(self, frame):
        # each track in tracks has format ([xmin, ymin, xmax, ymax], id)
        tracked_objects = []
        for track in self.tracker.trackers:
            coords = track.predict()[0].tolist()
            # changing to xmin, xmax, ymin, ymax format
            coords = (int(coords[0]), int(coords[2]), int(coords[1]), int(coords[3]))
            tracked_objects.append(DetectedObject(coords, 0, "", track.id))
        return True, tracked_objects

    def convert_detections_for_sort_alg(self, bboxes, confidence_scores):
        # for each bbox, convert from x, y, w, h to xmin, ymin, xmax, ymax (top-left and bottom-right)
        # then append detection confidence score to the end, return n x 5 np array
        converted_detections = []
        for i in range(len(bboxes)):
            x1, x2, y1, y2 = list(bboxes[i]) # comes from detected objects self.corners (see detection utils)
            score = confidence_scores[i]
            bbox = [x1, y1, x2, y2, score]
            converted_detections.append(bbox)
        return np.array(converted_detections)

