"""This module implements an operator that logs tracked obstacles."""

import os

import erdos


class MultipleObjectTrackerLoggerOperator(erdos.Operator):
    """Operator that logs tracked obstacles in the MOT16 file format.

    Args:
        obstacles_stream (:py:class:`erdos.ReadStream`): The stream on which
            :py:class:`~pylot.perception.messages.ObstaclesMessage` are
            received.
        flags (absl.flags): Object to be used to access absl flags.

    Attributes:
        _logger (:obj:`logging.Logger`): Instance to be used to log messages.
        _flags (absl.flags): Object to be used to access absl flags.
        _msg_cnt (:obj:`int`): Number of messages received.
    """
    def __init__(self, obstacles_stream: erdos.ReadStream,
                 finished_indicator_stream: erdos.WriteStream, flags):
        # Register a callback on obstacles data stream.
        obstacles_stream.add_callback(self.on_obstacles_msg)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags
        self._msg_cnt = 0

    @staticmethod
    def connect(obstacles_stream: erdos.ReadStream):
        finished_indicator_stream = erdos.WriteStream()
        return [finished_indicator_stream]

    def on_obstacles_msg(self, msg):
        """Logs obstacles to files.

        Invoked upon the receipt of a msg on the obstacles stream.

        Args:
            msg (:py:class:`~pylot.perception.messages.ObstaclesMessage`):
                Received message.
        """
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self.config.name))
        self._msg_cnt += 1
        if self._msg_cnt % self._flags.log_every_nth_message != 0:
            return
        assert len(msg.timestamp.coordinates) == 1
        timestamp = msg.timestamp.coordinates[0]
        lines = []
        for obstacle in msg.obstacles:
            if obstacle.is_person() or obstacle.is_vehicle():
                lines.append(obstacle.as_mot16_str(timestamp))

        # Write the data, MOT16 style: https://motchallenge.net/instructions/
        file_name = os.path.join(self._flags.data_path,
                                 'mot-{}.txt'.format(timestamp))
        with open(file_name, 'w') as outfile:
            outfile.writelines(lines)
