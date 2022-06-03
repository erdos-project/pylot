import erdos
from erdos.operator import TwoInOneOut
from erdos.context import TwoInOneOutContext

import pylot.utils

from pylot.perception.messages import ObstaclesMessageTuple


class TimeToDecisionOperator(TwoInOneOut):
    def __init__(self, flags):
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._last_obstacles_msg = None

    def on_left_data(self, context: TwoInOneOutContext,
                     data: pylot.utils.Pose):
        self._logger.debug('@{}: {} received pose message'.format(
            context.timestamp, self.config.name))
        ttd = TimeToDecisionOperator.time_to_decision(data.transform,
                                                      data.forward_speed, None)
        context.write_stream.send(erdos.Message(context.timestamp, ttd))

    def on_right_data(self, context: TwoInOneOutContext,
                      data: ObstaclesMessageTuple):
        self._last_obstacles_msg = data

    def destroy(self):
        self._logger.warn('destroying {}'.format(self.config.name))

    @staticmethod
    def time_to_decision(pose, forward_speed, obstacles):
        """Computes time to decision (in ms)."""
        # Time to deadline is 400 ms when driving at 10 m/s
        # Deadline decreases by 10ms for every 1m/s of extra speed.
        time_to_deadline = 400 - (forward_speed - 10) * 10
        # TODO: Include other environment information in the calculation.
        return time_to_deadline
