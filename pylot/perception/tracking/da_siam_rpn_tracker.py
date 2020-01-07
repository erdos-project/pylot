from absl import flags
import cv2
import erdos
import numpy as np
from skimage.measure import compare_ssim
from sklearn.utils.linear_assignment_ import linear_assignment
import torch

from DaSiamRPN.code.net import SiamRPNvot
from DaSiamRPN.code.run_SiamRPN import SiamRPN_init, SiamRPN_track

from pylot.perception.detection.utils import BoundingBox2D, DetectedObstacle
from pylot.perception.tracking.multi_object_tracker import MultiObjectTracker

flags.DEFINE_string('da_siam_rpn_model_path',
                    'dependencies/models/SiamRPNVOT.model',
                    'Path to the model')

MAX_TRACKER_AGE = 5


class SingleObjectDaSiamRPNTracker(object):
    def __init__(self, frame, bbox, siam_net, id=-1):
        self.bbox = bbox
        self.id = id
        self.missed_det_updates = 0
        target_pos = np.array(bbox.get_center_point())
        target_size = np.array([bbox.get_width(), bbox.get_height()])
        self._tracker = SiamRPN_init(frame, target_pos, target_size, siam_net)

    def track(self, frame):
        self._tracker = SiamRPN_track(self._tracker, frame)
        target_pos = self._tracker['target_pos']
        target_sz = self._tracker['target_sz']
        self.bbox = BoundingBox2D(int(target_pos[0] - target_sz[0] / 2.0),
                                  int(target_pos[0] + target_sz[0] / 2.0),
                                  int(target_pos[1] - target_sz[1] / 2.0),
                                  int(target_pos[1] + target_sz[1] / 2.0))
        return DetectedObstacle(self.bbox, 0, "")


class MultiObjectDaSiamRPNTracker(MultiObjectTracker):
    def __init__(self, flags):
        # Initialize the siam network.
        self._logger = erdos.utils.setup_logging(
            'multi_object_da_siam_rpn_trakcer', self.flags.log_file_name)
        self._siam_net = SiamRPNvot()
        self._siam_net.load_state_dict(torch.load(
            flags.da_siam_rpn_model_path))
        self._siam_net.eval().cuda()

    def reinitialize(self, frame, bboxes, confidence_scores, ids):
        # Create a tracker for each bbox.
        self._trackers = []
        for bbox in bboxes:
            self._trackers.append(
                SingleObjectDaSiamRPNTracker(frame, bbox, self._siam_net))

    def reinitialize_new(self, frame, bboxes, confidence_scores, ids):
        # Create matrix of similarities between detection and tracker bboxes.
        cost_matrix = self._create_hungarian_cost_matrix(frame, bboxes)
        # Run sklearn linear assignment (Hungarian Algo) with matrix
        assignments = linear_assignment(cost_matrix)

        updated_trackers = []
        # Add matched trackers to updated_trackers
        for bbox_idx, tracker_idx in assignments:
            updated_trackers.append(
                SingleObjectDaSiamRPNTracker(
                    frame, bboxes[bbox_idx], self._siam_net,
                    self._trackers[tracker_idx].obj_id))
        # Add 1 to age of any unmatched trackers, filter old ones
        if len(self._trackers) > len(bboxes):
            for i, tracker in enumerate(self._trackers):
                if i not in assignments[:, 1]:
                    tracker.missed_det_updates += 1
                    if tracker.missed_det_updates < MAX_TRACKER_AGE:
                        updated_trackers.append(tracker)
        # Create new trackers for new bboxes
        elif len(bboxes) > len(self._trackers):
            for i, bbox in enumerate(bboxes):
                if i not in assignments[:, 0]:
                    updated_trackers.append(
                        SingleObjectDaSiamRPNTracker(frame, bbox,
                                                     self._siam_net, ids[i]))

        self._trackers = updated_trackers

    def _create_hungarian_cost_matrix(self, frame, bboxes):
        # Create cost matrix with shape (num_bboxes, num_trackers)
        cost_matrix = [[0 for _ in range(len(self._trackers))]
                       for __ in range(len(bboxes))]
        for i, bbox in enumerate(bboxes):
            for j, tracker in enumerate(self._trackers):
                tracker_bbox = tracker.bbox
                # Get crops from frame
                self._logger.debug(bbox, tracker_bbox)
                bbox_crop = frame[bbox.y_min:bbox.y_max, bbox.x_min:bbox.x_max]
                tracker_bbox_crop = frame[
                    tracker_bbox.y_min:tracker_bbox.y_max,
                    tracker_bbox.x_min:tracker_bbox.x_max]
                # Resize larger crop to same shape as smaller one
                bbox_area = np.prod(bbox_crop.shape[:2])
                tracker_bbox_area = np.prod(tracker_bbox_crop.shape[:2])
                if bbox_area < tracker_bbox_area:
                    self._logger.debug(tracker_bbox_crop.shape)
                    tracker_bbox_crop = cv2.resize(
                        tracker_bbox_crop,
                        bbox_crop.shape[:2]
                        [::-1],  # cv2 needs width, then height
                        interpolation=cv2.INTER_AREA)
                else:
                    self._logger.debug(bbox_crop.shape)
                    bbox_crop = cv2.resize(
                        bbox_crop,
                        tracker_bbox_crop.shape[:2]
                        [::-1],  # cv2 needs width, then height
                        interpolation=cv2.INTER_AREA)
                # Use SSIM as metric for crop similarity, assign to matrix
                self._logger.debug(
                    bbox_crop.shape,
                    tracker_bbox_crop.transpose((1, 0, 2)).shape)
                cost_matrix[i][j] = compare_ssim(bbox_crop,
                                                 tracker_bbox_crop,
                                                 multichannel=True)
        return np.array(cost_matrix)
