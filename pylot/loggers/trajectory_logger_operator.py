"""This module implements an operator that logs obstacle trajectories."""

import json
import os

import erdos


class TrajectoryLoggerOperator(erdos.Operator):
    """Logs tracked obstacles trajectories to files.

    Args:
        obstacles_tracking_stream (:py:class:`erdos.ReadStream`): The stream on
            which :py:class:`~pylot.perception.messages.ObstacleTrajectoriesMessage`  # noqa: E501
            are received.
        flags (absl.flags): Object to be used to access absl flags.

    Attributes:
        _logger (:obj:`logging.Logger`): Instance to be used to log messages.
        _flags (absl.flags): Object to be used to access absl flags.
        _msg_cnt (:obj:`int`): Number of messages received.
    """
    def __init__(self, obstacle_tracking_stream: erdos.ReadStream,
                 finished_indicator_stream: erdos.WriteStream, flags):
        obstacle_tracking_stream.add_callback(self.on_trajectories_msg)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags
        self._msg_cnt = 0

    @staticmethod
    def connect(obstacle_tracking_stream: erdos.ReadStream):
        finished_indicator_stream = erdos.WriteStream()
        return [finished_indicator_stream]

    def on_trajectories_msg(self, msg):
        """Logs obstacle trajectories to files.

        Invoked upon the receipt of a msg on the obstacles trajectories stream.

        Args:
            msg (:py:class:`~pylot.perception.messages.ObstacleTrajectoriesMessage`):  # noqa: E501
                Received message.
        """
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self.config.name))
        self._msg_cnt += 1
        if self._msg_cnt % self._flags.log_every_nth_message != 0:
            return
        trajectories = [
            str(trajectory) for trajectory in msg.obstacle_trajectories
        ]
        assert len(msg.timestamp.coordinates) == 1
        timestamp = msg.timestamp.coordinates[0]
        # Write the trajectories.
        file_name = os.path.join(self._flags.data_path,
                                 'trajectories-{}.json'.format(timestamp))
        with open(file_name, 'w') as outfile:
            json.dump(trajectories, outfile)
