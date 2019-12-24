import erdust
import pptk


class LidarVisualizerOperator(erdust.Operator):
    """ Subscribes to point cloud streams and visualizes point clouds."""
    def __init__(self, point_cloud_stream, name, log_file_name=None):
        point_cloud_stream.add_callback(self.display_point_cloud)
        self._name = name
        self._logger = erdust.utils.setup_logging(name, log_file_name)
        self._cnt = 0

    @staticmethod
    def connect(point_cloud_stream):
        return []

    def display_point_cloud(self, msg):
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self._name))
        #        filename = './carla-point-cloud{}.ply'.format(self._cnt)
        pptk.viewer(msg.point_cloud)
        # pcd = open3d.PointCloud()
        # pcd.points = open3d.Vector3dVector(msg.point_cloud)
        # open3d.draw_geometries([pcd])
