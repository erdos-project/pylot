"""This module implements an operator that logs pose sensor readings."""

import json
import os

import erdos


class PoseLoggerOperator(erdos.Operator):
    """Subscribes to pose streams and logs pose measurements.

    This will log every frame to preserve linearization when approximating jerk
    in smoothness evaluation metrics.

    Args:
        pose_stream: (:py:class:`erdos.ReadStream`): The stream on which
            :py:class:`~pylot.localization.messages.PoseMessage` are received.
        flags (absl.flags): Object to be used to access absl flags.

    Attributes:
        _logger (:obj:`logging.Logger`): Instance to be used to log messages.
        _flags (absl.flags): Object to be used to access absl flags.
        _msg_cnt (:obj:`int`): Number of messages received.
        _data_path (:obj:`str`): Directory to which to log files.
    """
    def __init__(self, pose_stream: erdos.ReadStream,
                 finished_indicator_stream: erdos.WriteStream, flags):
        pose_stream.add_callback(self.on_pose_update)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags
        self._msg_cnt = 0
        self._data_path = os.path.join(self._flags.data_path, 'pose')
        os.makedirs(self._data_path, exist_ok=True)

    @staticmethod
    def connect(pose_stream: erdos.ReadStream):
        finished_indicator_stream = erdos.WriteStream()
        return [finished_indicator_stream]

    def on_pose_update(self, msg):
        """Invoked upon receipt of an Pose message.

        Args:
            msg (:py:class:`pylot.localization.messages.PoseMessage`): Message
                to be logged.
        """
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self.config.name))
        self._msg_cnt += 1
        if self._msg_cnt % self._flags.log_every_nth_message != 0:
            return
        assert len(msg.timestamp.coordinates) == 1
        timestamp = msg.timestamp.coordinates[0]
        file_name = os.path.join(self._data_path,
                                 'pose-{}.json'.format(timestamp))
        measurements = {
            "x": str(msg.data.transform.location.x),
            "y": str(msg.data.transform.location.y),
            "z": str(msg.data.transform.location.z),
            "pitch": str(msg.data.transform.rotation.pitch),
            "yaw": str(msg.data.transform.rotation.yaw),
            "roll": str(msg.data.transform.rotation.roll),
            "timestamp": str(timestamp)
        }
        with open(file_name, 'w') as outfile:
            json.dump(measurements, outfile)
