from absl import flags

from nanonets_object_tracking.deepsort import deepsort_rbc
from pylot.perception.detection.utils import BoundingBox2D, DetectedObstacle
from pylot.perception.tracking.multi_object_tracker import MultiObjectTracker

flags.DEFINE_string(
    'deep_sort_tracker_pedestrian_weights_path',
    'dependencies/nanonets_object_tracking/ped_feature_extractor',
    'Path to weights for pedestrian feature extractor model')


class MultiObjectDeepSORTTracker(MultiObjectTracker):
    def __init__(self, flags, logger):
        # Initialize the deepsort object, which has a tracker object within it
        self._deepsort = deepsort_rbc(
            wt_path=flags.deep_sort_tracker_pedestrian_weights_path)
        self.tracker = None

    def reinitialize(self, frame, bboxes, confidence_scores, ids):
        # update tracker with new detections
        self.track(frame, confidence_scores=confidence_scores, bboxes=bboxes)

    def track(self, frame, confidence_scores=None, bboxes=None):
        if bboxes:
            detections = self.convert_detections_for_deepsort_alg(bboxes)
            self.tracker, detections_class = self._deepsort.run_deep_sort(
                frame, confidence_scores, detections)
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

    def convert_detections_for_deepsort_alg(self, bboxes):
        converted_bboxes = []
        for bbox in bboxes:
            converted_bboxes.append(bbox.as_width_height_bbox())
        return converted_bboxes
