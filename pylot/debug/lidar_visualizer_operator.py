import open3d
import pptk

from erdos.op import Op
from erdos.utils import setup_logging

import pylot_utils


class LidarVisualizerOperator(Op):
    def __init__(self, name, flags, log_file_name=None):
        super(LidarVisualizerOperator, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._flags = flags
        self.cnt = 0

    @staticmethod
    def setup_streams(input_streams):
        input_streams.filter(pylot_utils.is_lidar_stream).add_callback(
            LidarVisualizerOperator.display_point_cloud)
        return []

    def display_point_cloud(self, msg):
        #        filename = './carla-point-cloud{}.ply'.format(self.cnt)
        pptk.viewer(msg.point_cloud)
        # pcd = open3d.PointCloud()
        # pcd.points = open3d.Vector3dVector(msg.point_cloud)
        # open3d.draw_geometries([pcd])

    def execute(self):
        self.spin()
