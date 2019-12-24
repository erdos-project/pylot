import erdust
import json


class IMULoggerOperator(erdust.Operator):
    """ Subscribes to IMU streams and logs IMU measurements. This will log
    every frame to preserve linearization when approximating jerk in smoothness
    evaluation metrics."""
    def __init__(self, imu_stream, name, flags, log_file_name=None):
        imu_stream.add_callback(self.on_imu_update)
        self._name = name
        self._logger = erdust.utils.setup_logging(name, log_file_name)
        self._flags = flags

    @staticmethod
    def connect(imu_stream):
        return []

    def on_imu_update(self, msg):
        """ The callback function that gets called upon receipt of the
        IMU message to be logged.

        Args:
            msg: A message of type `pylot.simulation.messages.IMUMessage` to
                be logged.
        """
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self._name))
        transform = msg.transform
        acceleration = msg.acceleration
        gyro = msg.gyro
        compass = msg.compass
        timestamp = msg.timestamp.coordinates[0]
        file_name = '{}imu-{}.json'.format(self._flags.data_path, timestamp)
        measurements = {
            "transform": str(transform),
            "acceleration": str(acceleration),
            "gyro": str(gyro),
            "compass": str(compass),
            "timestamp": str(timestamp)
        }
        with open(file_name, 'w') as outfile:
            json.dump(measurements, outfile)
