import erdust
import open3d as o3d
import os


class LidarLoggerOperator(erdust.Operator):
    """ Logs point cloud messages."""

    def __init__(self, lidar_stream, name, flags):
        lidar_stream.add_callback(self.on_lidar_frame)
        self._flags = flags
        self._pc_msg_cnt = 0

    @staticmethod
    def connect(lidar_stream):
        return []

    def on_lidar_frame(self, msg):
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
