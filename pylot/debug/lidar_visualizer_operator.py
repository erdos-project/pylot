import erdos
import pptk


class LidarVisualizerOperator(erdos.Operator):
    """ Subscribes to point cloud streams and visualizes point clouds."""
    def __init__(self, point_cloud_stream, name, log_file_name=None):
        point_cloud_stream.add_callback(self.display_point_cloud)
        self._name = name
        self._logger = erdos.utils.setup_logging(name, log_file_name)

    @staticmethod
    def connect(point_cloud_stream):
        return []

    def display_point_cloud(self, msg):
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self._name))
        pptk.viewer(msg.point_cloud)
