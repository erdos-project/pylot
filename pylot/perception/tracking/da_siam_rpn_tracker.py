import numpy as np
import torch

from DaSiamRPN.code.net import SiamRPNvot
from DaSiamRPN.code.run_SiamRPN import SiamRPN_init, SiamRPN_track

from pylot.perception.detection.utils import DetectedObject
from pylot.perception.tracking.multi_object_tracker import MultiObjectTracker


class SingleObjectDaSiamRPNTracker(object):
    def __init__(self, frame, bbox, siam_net):
        width = bbox[1] - bbox[0]
        height = bbox[3] - bbox[2]
        target_pos = np.array([(bbox[0] + bbox[1]) / 2.0,
                               (bbox[2] + bbox[3]) / 2.0])
        target_size = np.array([width, height])
        self._tracker = SiamRPN_init(frame, target_pos, target_size, siam_net)

    def track(self, frame):
        self._tracker = SiamRPN_track(self._tracker, frame)
        target_pos = self._tracker['target_pos']
        target_sz = self._tracker['target_sz']
        bbox = (int(target_pos[0] - target_sz[0] / 2.0),
                int(target_pos[0] + target_sz[0] / 2.0),
                int(target_pos[1] - target_sz[1] / 2.0),
                int(target_pos[1] + target_sz[1] / 2.0))
        return DetectedObject(bbox, "", 0)


class MultiObjectDaSiamRPNTracker(MultiObjectTracker):
    def __init__(self, flags):
        # Initialize the siam network.
        self._siam_net = SiamRPNvot()
        self._siam_net.load_state_dict(
            torch.load(flags.da_siam_rpn_model_path))
        self._siam_net.eval().cuda()

    def reinitialize(self, frame, bboxes, confidence_scores):
        # Create a tracker for each bbox.
        self._trackers = []
        for bbox in bboxes:
            self._trackers.append(
                SingleObjectDaSiamRPNTracker(frame, bbox, self._siam_net))
