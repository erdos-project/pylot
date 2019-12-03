from collections import deque
import erdust
import numpy as np


class FusionVerificationOperator(erdust.Operator):
    def __init__(self,
                 ground_vehicles_stream,
                 fusion_stream,
                 name,
                 log_file_name=None):
        ground_vehicles_stream.add_callback(self.on_vehicles_update)
        fusion_stream.add_callback(self.on_fusion_update)
        self._logger = erdust.utils.setup_logging(name, log_file_name)
        self.vehicles = deque()

    @staticmethod
    def connect(ground_vehicles_stream, fusion_stream):
        return []

    def on_vehicles_update(self, msg):
        vehicle_positions = []
        for vehicle in msg.vehicles:
            position = np.array([vehicle.transform.location.x,
                                 vehicle.transform.location.y])
            vehicle_positions.append(position)

        self.vehicles.append((msg.timestamp, vehicle_positions))

    def on_fusion_update(self, msg):
        while self.vehicles[0][0] < msg.timestamp:
            self.vehicles.popleft()

        truths = self.vehicles[0][1]
        min_errors = []
        for prediction in msg.obj_positions:
            min_error = float("inf")
            for truth in truths:
                error = np.linalg.norm(prediction - truth)
                min_error = min(error, min_error)
            min_errors.append(min_error)

        self._logger.info(
            "Fusion: min vehicle position errors: {}".format(min_errors))
