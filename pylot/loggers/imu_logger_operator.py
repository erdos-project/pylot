import erdos
import json


class IMULoggerOperator(erdos.Operator):
    """ Subscribes to IMU streams and logs IMU measurements. This will log
    every frame to preserve linearization when approximating jerk in smoothness
    evaluation metrics."""
    def __init__(self, imu_stream, name, flags, log_file_name=None):
        imu_stream.add_callback(self.on_imu_update)
        self._name = name
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._flags = flags
        self._msg_cnt = 0

    @staticmethod
    def connect(imu_stream):
        return []

    def on_imu_update(self, msg):
        """ The callback function that gets called upon receipt of the
        IMU message to be logged.

        Args:
            msg: A message of type `pylot.perception.messages.IMUMessage` to
                be logged.
        """
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self._name))
        self._msg_cnt += 1
        if self._msg_cnt % self._flags.log_every_nth_message != 0:
            return
        assert len(msg.timestamp.coordinates) == 1
        timestamp = msg.timestamp.coordinates[0]
        file_name = '{}imu-{}.json'.format(self._flags.data_path, timestamp)
        measurements = {
            "transform": str(msg.transform),
            "acceleration": str(msg.acceleration),
            "gyro": str(msg.gyro),
            "compass": str(msg.compass),
            "timestamp": str(timestamp)
        }
        with open(file_name, 'w') as outfile:
            json.dump(measurements, outfile)
