from DaSiamRPN.code.net import SiamRPNvot
from DaSiamRPN.code.run_SiamRPN import SiamRPN_init, SiamRPN_track

from lapsolver import solve_dense

import numpy as np

from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.detection.utils import BoundingBox2D
from pylot.perception.tracking.multi_object_tracker import MultiObjectTracker

import torch


class SingleObjectDaSiamRPNTracker(object):
    def __init__(self, frame, obstacle, siam_net):
        """ Construct a single obstacle tracker.

        Args:
            frame (:py:class:`~pylot.perception.camera_frame.CameraFrame`):
                Frame to reinitialize with.
            obstacle: perception.detection.obstacle.Obstacle.
        """
        self.obstacle = obstacle
        self.missed_det_updates = 0
        center_point = obstacle.bounding_box_2D.get_center_point()
        target_pos = np.array([center_point.x, center_point.y])
        target_size = np.array([
            obstacle.bounding_box_2D.get_width(),
            obstacle.bounding_box_2D.get_height()
        ])
        self._tracker = SiamRPN_init(frame.frame, target_pos, target_size,
                                     siam_net)

    def track(self, frame):
        """ Tracks obstacles in a frame.

        Args:
            frame (:py:class:`~pylot.perception.camera_frame.CameraFrame`):
                Frame to track in.
        """
        self._tracker = SiamRPN_track(self._tracker, frame.frame)
        target_pos = self._tracker['target_pos']
        target_sz = self._tracker['target_sz']
        self.obstacle.bounding_box_2D = BoundingBox2D(
            int(target_pos[0] - target_sz[0] / 2.0),
            int(target_pos[0] + target_sz[0] / 2.0),
            int(target_pos[1] - target_sz[1] / 2.0),
            int(target_pos[1] + target_sz[1] / 2.0))
        return Obstacle(self.obstacle.bounding_box_2D,
                        self.obstacle.confidence, self.obstacle.label,
                        self.obstacle.id)

    def reset_bbox(self, bbox: BoundingBox2D):
        """Resets tracker's bounding box with a new bounding box."""
        center = bbox.get_center_point()
        self._tracker['target_pos'] = np.array([center.x, center.y])
        self._tracker['target_sz'] = np.array(
            [bbox.get_width(), bbox.get_height()])


class MultiObjectDaSiamRPNTracker(MultiObjectTracker):
    def __init__(self, flags, logger):
        # Initialize the siam network.
        self._logger = logger
        self._siam_net = SiamRPNvot()
        self._siam_net.load_state_dict(torch.load(
            flags.da_siam_rpn_model_path))
        self._siam_net.eval().cuda()
        self._trackers = []
        self._min_matching_iou = flags.min_matching_iou
        self._max_missed_detections = flags.obstacle_track_max_age

    def initialize(self, frame, obstacles):
        """ Initializes a multiple obstacle tracker.

        Args:
            frame: perception.camera_frame.CameraFrame to reinitialize with.
            obstacles: List of perception.detection.obstacle.Obstacle.
        """
        # Create a tracker for each obstacle.
        for obstacle in obstacles:
            self._trackers.append(
                SingleObjectDaSiamRPNTracker(frame, obstacle, self._siam_net))

    def reinitialize(self, frame, obstacles):
        """ Renitializes a multiple obstacle tracker.

        Args:
            frame: perception.camera_frame.CameraFrame to reinitialize with.
            obstacles: List of perception.detection.obstacle.Obstacle.
        """
        # If empty obstacles passed in, continue existing tracks and exit.
        if not obstacles:
            self.track(frame)
            return

        if self._trackers == []:
            self.initialize(frame, obstacles)
        else:
            # Update the bounding boxes so that the matching happens between
            # bounding boxes computed on the same frame.
            self.track(frame, False)
        # Create matrix of similarities between detection and tracker bboxes.
        cost_matrix = self._create_hungarian_cost_matrix(obstacles)
        # Run linear assignment (Hungarian Algo) with matrix.
        row_ids, col_ids = solve_dense(cost_matrix)
        matched_map = {}
        for row_id, col_id in zip(row_ids, col_ids):
            matched_map[self._trackers[col_id].obstacle.id] = row_id
        matched_obstacle_indices, matched_tracker_indices = set(row_ids), set(
            col_ids)

        updated_trackers = []
        # Separate matched and unmatched tracks, obstacles.
        unmatched_tracker_indices, matched_trackers, unmatched_trackers = \
            self._separate_matches_from_unmatched(self._trackers,
                                                  matched_tracker_indices)
        unmatched_obstacle_indices, matched_obstacles, unmatched_obstacles = \
            self._separate_matches_from_unmatched(obstacles,
                                                  matched_obstacle_indices)

        # Add successfully matched trackers to updated_trackers.
        for tracker in matched_trackers:
            tracker.missed_det_updates = 0
            # Update the tracker's internal bounding box. If we don't do
            # this, the tracker's bounding box degrades across the frames until
            # it doesn't overlap with the bounding box the detector outputs.
            tracker.reset_bbox(
                obstacles[matched_map[tracker.obstacle.id]].bounding_box_2D)
            updated_trackers.append(tracker)
        # Add 1 to age of any unmatched trackers, filter old ones.
        for tracker in unmatched_trackers:
            tracker.missed_det_updates += 1
            if tracker.missed_det_updates <= self._max_missed_detections:
                updated_trackers.append(tracker)
            else:
                self._logger.debug("Dropping tracker with id {}".format(
                    tracker.obstacle.id))
        # Initialize trackers for unmatched obstacles.
        for obstacle in unmatched_obstacles:
            updated_trackers.append(
                SingleObjectDaSiamRPNTracker(frame, obstacle, self._siam_net))
        # Keep one tracker per obstacle id; prefer trackers with recent
        # detection updates.
        unique_updated_trackers = {}
        for tracker in updated_trackers:
            if tracker.obstacle.id not in unique_updated_trackers:
                unique_updated_trackers[tracker.obstacle.id] = tracker
            elif (unique_updated_trackers[tracker.obstacle.id].
                  missed_det_updates > tracker.missed_det_updates):
                unique_updated_trackers[tracker.obstacle.id] = tracker

        self._trackers = list(unique_updated_trackers.values())

    def track(self, frame, missed_detection: bool = True):
        """ Tracks obstacles in a frame.

        Args:
            frame: perception.camera_frame.CameraFrame to track in.
        """
        tracked_obstacles = []
        for tracker in self._trackers:
            tracked_obstacles.append(tracker.track(frame))
            if missed_detection:
                tracker.missed_det_updates += 1
        self._trackers = [
            tracker for tracker in self._trackers
            if tracker.missed_det_updates <= self._max_missed_detections
        ]
        return True, tracked_obstacles

    def _create_hungarian_cost_matrix(self, obstacles):
        # Create cost matrix with shape (num_bboxes, num_trackers)
        cost_matrix = [[0 for _ in range(len(self._trackers))]
                       for __ in range(len(obstacles))]
        for i, obstacle in enumerate(obstacles):
            for j, tracker in enumerate(self._trackers):
                obstacle_bbox = obstacle.bounding_box_2D
                tracker_bbox = tracker.obstacle.bounding_box_2D
                iou = obstacle_bbox.calculate_iou(tracker_bbox)
                # If track too far from det, mark pair impossible with np.nan
                if iou >= self._min_matching_iou:
                    # Negative since high IoU indicates lower cost
                    cost_matrix[i][j] = -iou
                else:
                    cost_matrix[i][j] = np.nan
        return np.array(cost_matrix)

    def _separate_matches_from_unmatched(self, obstacles,
                                         matched_obstacle_indices):
        unmatched_obstacle_indices = \
            set(range(len(obstacles))) - matched_obstacle_indices
        matched_obstacles = [obstacles[i] for i in matched_obstacle_indices]
        unmatched_obstacles = [
            obstacles[i] for i in unmatched_obstacle_indices
        ]
        return (unmatched_obstacle_indices, matched_obstacles,
                unmatched_obstacles)
