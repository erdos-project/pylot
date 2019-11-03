import numpy as np
import torch

from sort.sort import *
from pylot.perception.tracking.multi_object_tracker import MultiObjectTracker


class MultiObjectSORTTracker(MultiObjectTracker):
    def __init__(self, flags):
        self.tracker = Sort()

    def reinitialize(self, frame, bboxes, deep_sort_logs):
        detections = self.get_detections_and_scores_from_logs(deep_sort_logs)
        self.tracker.update(detections)

    def track(self, frame):
        tracks = [(track.predict(), track.id) for track in self.tracker.trackers]
        bboxes = []
        for track_info in tracks:
            bboxes.append(track_info) # track_info = ([xmin, ymin, xmax, ymax], id)
        return True, bboxes

    def get_detections_and_scores_from_logs(self, deep_sort_logs):
        detections = []
        for i in range(len(deep_sort_logs)):
            info = list(deep_sort_logs[i].split(','))
            info = list(map(float,info))
            coords_and_conf = list(info[2:7]) # x, y, w, h, conf
            # convert to x1, y1, x2, y2 format
            coords_and_conf[2] += coords_and_conf[0]
            coords_and_conf[3] += coords_and_conf[1]
            detections.append(coords_and_conf)
        return np.array(detections)

