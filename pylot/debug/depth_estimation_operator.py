from collections import deque
import threading

from erdos.op import Op
from erdos.utils import setup_logging

from pylot.utils import is_camera_stream, is_can_bus_stream, is_depth_camera_stream, is_lidar_stream
from pylot.simulation.utils import depth_to_local_point_cloud


class DepthEstimationOp(Op):
    def __init__(self, name, flags, log_file_name=None):
        super(DepthEstimationOp, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._flags = flags
        self._left_camera_msgs = deque()
        self._right_camera_msgs = deque()
        self._depth_msgs = deque()
        self._point_cloud_msgs = deque()
        self._can_bus_msgs = deque()
        self._lock = threading.Lock()

    @staticmethod
    def setup_streams(input_streams, left_camera_name, right_camera_name):
        input_streams.filter(is_depth_camera_stream).add_callback(
            DepthEstimationOp.on_depth_msg)
        input_streams.filter(is_lidar_stream).add_callback(
            DepthEstimationOp.on_point_cloud_msg)
        input_streams.filter(is_can_bus_stream).add_callback(
            DepthEstimationOp.on_can_bus_msg)
        camera_streams = input_streams.filter(is_camera_stream)
        camera_streams.filter_name(left_camera_name).add_callback(
            DepthEstimationOp.on_left_camera_msg)
        camera_streams.filter_name(right_camera_name).add_callback(
            DepthEstimationOp.on_right_camera_msg)

        input_streams.add_completion_callback(
            DepthEstimationOp.on_watermark)
        return []

    def on_watermark(self, msg):
        with self._lock:
            left_camera_msg = self._left_camera_msgs.popleft()
            right_camera_msg = self._right_camera_msgs.popleft()
            depth_msg = self._depth_msgs.popleft()
            point_cloud_msg = self._point_cloud_msgs.popleft()
            can_bus_msg = self._can_bus_msgs.popleft()

            assert (left_camera_msg.timestamp == right_camera_msg.timestamp ==
                    depth_msg.timestamp == point_cloud_msg.timestamp ==
                    can_bus_msg.timestamp)

            # self.compare_depth_lidar(
            #     can_bus_msg.transform, depth_msg, point_cloud_msg)

    def compare_depth_lidar(self,
                            vehicle_transform,
                            depth_msg,
                            point_cloud_msg):
        car_transform = vehicle_transform * point_cloud_msg.transform
        points = car_transform.transform_points(point_cloud_msg.point_cloud)

    def compare_depth_helper(self,
                             vehicle_transform,
                             point_cloud_msg,
                             depth_msg):
        # Transform point cloud to world coordinates
        car_transform = vehicle_transform * point_cloud_msg.transform
        point_cloud = car_transform.transform_points(
            point_cloud_msg.point_cloud).tolist()

        depth_point_cloud = depth_to_local_point_cloud(
            depth_msg.frame, depth_msg.width, depth_msg.height,
            depth_msg.fov, max_depth=1.0)
        car_transform = vehicle_transform * depth_msg.transform
        depth_point_cloud = car_transform.transform_points(
            depth_point_cloud).tolist()

        for (x, y, z) in depth_point_cloud:
            pcd = (x, y, z)
            for (px, py, pz) in point_cloud:
                if abs(px - x) < 2 and abs(py - y) < 2:
                    pc = (px, py, pz)
                    if abs(pz - z) < 1:
                        print("Same depth {} {} {}".format(
                            pc, pcd, depth_msg.timestamp))
                    else:
                        print("Different depth {} {} {}".format(
                            pc, pcd, depth_msg.timestamp))

    def on_left_camera_msg(self, msg):
        with self._lock:
            self._left_camera_msgs.append(msg)

    def on_right_camera_msg(self, msg):
        with self._lock:
            self._right_camera_msgs.append(msg)

    def on_can_bus_msg(self, msg):
        with self._lock:
            self._can_bus_msgs.append(msg)

    def on_depth_msg(self, msg):
        with self._lock:
            self._depth_msgs.append(msg)

    def on_point_cloud_msg(self, msg):
        with self._lock:
            self._point_cloud_msgs.append(msg)

    def execute(self):
        self.spin()
