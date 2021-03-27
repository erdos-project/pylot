from collections import deque

import erdos

import numpy as np

from pylot.perception.messages import ObstaclePositionsSpeedsMessage


class FusionOperator(erdos.Operator):
    """Fusion Operator

    Args:
        rgbd_max_range (:obj:`float`): Maximum distance of the rgbd frame
        camera_fov (:obj:`float`): Angular field of view in radians of the RGBD
            and RGB cameras used to infer depth info and generate bounding
            boxes respectively. Note that camera position, orientation, and
            FOV must be identical for both.
    """
    def __init__(self,
                 pose_stream,
                 obstacles_stream,
                 depth_camera_stream,
                 fused_stream,
                 flags,
                 camera_fov=np.pi / 4,
                 rgbd_max_range=1000):
        self.pose_stream = pose_stream
        self.obstacles_stream = obstacles_stream
        self.depth_camera_stream = depth_camera_stream
        pose_stream.add_callback(self.update_pos)
        obstacles_stream.add_callback(self.update_obstacles)
        depth_camera_stream.add_callback(self.update_distances)
        self._fused_stream = fused_stream
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags
        self._segments = []
        self._rgbd_max_range = rgbd_max_range
        # TODO(ionel): Check fov is same as the camera fov.
        self._camera_fov = camera_fov
        self._car_positions = deque()
        self._distances = deque()
        self._obstacles = deque()

    @staticmethod
    def connect(pose_stream, obstacles_stream, depth_camera_stream):
        fused_stream = erdos.WriteStream()
        return [fused_stream]

    def __calc_obstacle_positions(self, obstacle_bboxes, distances,
                                  car_position, car_orientation):
        obstacle_positions = []
        for bbox in obstacle_bboxes:
            bounding_box_center = np.average(
                [[bbox.x_min, bbox.x_max], [bbox.y_min, bbox.y_max]], axis=1)

            distance = np.median(distances[bbox.x_min:bbox.x_max,
                                           bbox.y_min:bbox.y_max])
            vertical_angle, horizontal_angle = (
                self._camera_fov * (bounding_box_center - distances.shape) /
                distances.shape)

            horizontal_diagonal = distance * np.cos(vertical_angle)

            forward_distance = horizontal_diagonal * np.cos(horizontal_angle)
            right_distance = horizontal_diagonal * np.sin(horizontal_angle)

            # TODO(peter): check that this is right
            position_x = car_position[0] + forward_distance * np.cos(
                car_orientation) - right_distance * np.sin(car_orientation)
            position_y = car_position[1] + forward_distance * np.sin(
                car_orientation) - right_distance * np.cos(car_orientation)

            obstacle_positions.append([position_x, position_y])

        return obstacle_positions

    def __discard_old_data(self):
        """Discards stored data that are too old to be used for fusion"""
        oldest_timestamp = min([
            self._car_positions[-1][0], self._distances[-1][0],
            self._obstacles[-1][0]
        ])
        for queue in [self._car_positions, self._distances, self._obstacles]:
            while queue[0][0] < oldest_timestamp:
                queue.popleft()

    @erdos.profile_method()
    def fuse(self):
        # Return if we don't have car position, distances or obstacles.
        if min(
                map(len,
                    [self._car_positions, self._distances, self._obstacles
                     ])) == 0:
            return
        self.__discard_old_data()
        obstacle_positions = self.__calc_obstacle_positions(
            self._obstacles[0][1], self._distances[0][1],
            self._car_positions[0][1][0],
            np.arccos(self._car_positions[0][1][1][0]))
        timestamp = self._obstacles[0][0]

        output_msg = ObstaclePositionsSpeedsMessage(timestamp,
                                                    obstacle_positions)
        self._fused_stream.send(output_msg)

    def update_pos(self, msg):
        vehicle_pos = ((msg.data.transform.location.x,
                        msg.data.transform.location.y,
                        msg.data.transform.location.z),
                       (msg.data.transform.forward_vector.x,
                        msg.data.transform.forward_vector.y,
                        msg.data.transform.forward_vector.z))
        self._car_positions.append((msg.timestamp, vehicle_pos))

    def update_obstacles(self, msg):
        # Filter obstacles
        self._logger.info("Received update obstacles")
        vehicle_bounds = []
        for obstacle in msg.obstacles:
            self._logger.info("%s received: %s ", self.config.name, obstacle)
            # TODO(ionel): Deal with different types of labels.
            if obstacle.label in {"truck", "car"}:
                vehicle_bounds.append(obstacle.bounding_box_2D)
        self._obstacles.append((msg.timestamp, vehicle_bounds))

    def update_distances(self, msg):
        self._distances.append((msg.timestamp, msg.frame.as_numpy_array()))

    def run(self):
        while True:
            pose_msg = self.pose_stream.read()
            obstacles_msg = self.obstacles_stream.read()
            depth_camera_msg = self.depth_camera_stream.read()
            self.update_pos(pose_msg)
            self.update_obstacles(obstacles_msg)
            self.update_distances(depth_camera_msg)
            self.fuse()
