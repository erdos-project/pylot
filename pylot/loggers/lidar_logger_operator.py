import erdos
import open3d as o3d
import os


class LidarLoggerOperator(erdos.Operator):
    """ Logs point cloud messages."""

    def __init__(self, lidar_stream, name, flags, log_file_name=None):
        lidar_stream.add_callback(self.on_lidar_frame)
        self._name = name
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._flags = flags
        self._pc_msg_cnt = 0

    @staticmethod
    def connect(lidar_stream):
        return []

    def on_lidar_frame(self, msg):
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self._name))
        self._pc_msg_cnt += 1
        if self._pc_msg_cnt % self._flags.log_every_nth_frame != 0:
            return
        # Write the lidar information.
        file_name = os.path.join(
            self._flags.data_path,
            'carla-lidar-{}.ply'.format(msg.timestamp.coordinates[0]))
        pcd = o3d.PointCloud()
        pcd.points = o3d.Vector3dVector(msg.point_cloud)
        o3d.write_point_cloud(file_name, pcd)
