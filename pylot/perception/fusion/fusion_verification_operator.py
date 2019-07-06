from collections import deque
import numpy as np

from erdos.op import Op
from erdos.utils import setup_logging

from pylot.utils import is_fusion_stream, is_ground_vehicles_stream


class FusionVerificationOperator(Op):
    def __init__(self, name, log_file_name=None):
        super(FusionVerificationOperator, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self.vehicles = deque()

    @staticmethod
    def setup_streams(input_streams):
        input_streams.filter(is_ground_vehicles_stream).add_callback(
            FusionVerificationOperator.on_vehicles_update)
        input_streams.filter(is_fusion_stream).add_callback(
            FusionVerificationOperator.on_fusion_update)
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

    def execute(self):
        self.spin()
