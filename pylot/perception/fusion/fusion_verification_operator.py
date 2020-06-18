from collections import deque

import erdos

import numpy as np


class FusionVerificationOperator(erdos.Operator):
    def __init__(self, ground_obstacles_stream, fusion_stream):
        ground_obstacles_stream.add_callback(self.on_obstacles_update)
        fusion_stream.add_callback(self.on_fusion_update)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self.vehicles = deque()

    @staticmethod
    def connect(ground_obstacles_stream, fusion_stream):
        return []

    def on_obstacles_update(self, msg):
        vehicle_positions = []
        for obstacle in msg.obstacles:
            if obstacle.is_vehicle():
                position = np.array([
                    obstacle.transform.location.x,
                    obstacle.transform.location.y
                ])
                vehicle_positions.append(position)

        self.vehicles.append((msg.timestamp, vehicle_positions))

    def on_fusion_update(self, msg):
        while self.vehicles[0][0] < msg.timestamp:
            self.vehicles.popleft()

        truths = self.vehicles[0][1]
        min_errors = []
        for prediction in msg.obstacle_positions:
            min_error = float("inf")
            for truth in truths:
                error = np.linalg.norm(prediction - truth)
                min_error = min(error, min_error)
            min_errors.append(min_error)

        self._logger.info(
            "Fusion: min vehicle position errors: {}".format(min_errors))
