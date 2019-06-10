import numpy as np
import open3d as o3d

import pylot.utils
from pylot.perception.segmentation.utils import transform_to_cityscapes_palette

from erdos.op import Op
from erdos.utils import setup_csv_logging, setup_logging

import pickle

class LidarLoggerOp(Op):
    def __init__(self, name, flags, log_file_name=None, csv_file_name=None):
        super(LidarLoggerOp, self).__init__(name)
        self._flags = flags
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._last_lidar_timestamp = -1

    @staticmethod
    def setup_streams(input_streams):
        input_streams.filter(
            pylot.utils.is_lidar_stream).add_callback(
                LidarLoggerOp.on_lidar_frame)
        return []

    def on_lidar_frame(self, msg):
        # Ensure we didn't skip a frame.
        if self._last_lidar_timestamp != -1:
            assert (self._last_lidar_timestamp + 1 ==
                    msg.timestamp.coordinates[1])
        self._last_lidar_timestamp = msg.timestamp.coordinates[1]
        if self._last_lidar_timestamp % self._flags.log_every_nth_frame != 0:
            return
        # Write the lidar information.
        file_name = '{}carla-lidar-{}.ply'.format(
            self._flags.data_path, self._last_lidar_timestamp)
        pcd = o3d.PointCloud()
        pcd.points = o3d.Vector3dVector(msg.data.data)
        o3d.write_point_cloud(file_name, pcd)
