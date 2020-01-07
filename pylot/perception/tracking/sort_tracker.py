import numpy as np

from sort.sort import *
from pylot.perception.detection.utils import BoundingBox2D, DetectedObstacle
from pylot.perception.tracking.multi_object_tracker import MultiObjectTracker


class MultiObjectSORTTracker(MultiObjectTracker):
    def __init__(self, flags):
        self.tracker = Sort()

    def reinitialize(self, frame, bboxes, confidence_scores, ids):
        detections = self.convert_detections_for_sort_alg(
            bboxes, confidence_scores)
        self.tracker.update(detections)

    def track(self, frame):
        # each track in tracks has format ([xmin, ymin, xmax, ymax], id)
        obstacles = []
        for track in self.tracker.trackers:
            coords = track.predict()[0].tolist()
            # changing to xmin, xmax, ymin, ymax format
            bbox = BoundingBox2D(int(coords[0]), int(coords[2]),
                                 int(coords[1]), int(coords[3]))
            obstacles.append(DetectedObstacle(bbox, 0, "", track.id))
        return True, obstacles

    def convert_detections_for_sort_alg(self, bboxes, confidence_scores):
        converted_detections = []
        for i in range(len(bboxes)):
            score = confidence_scores[i]
            bbox = [
                bboxes[i].x_min, bboxes[i].y_min, bboxes[i].x_max,
                bboxes[i].y_max, score
            ]
            converted_detections.append(bbox)
        return np.array(converted_detections)
