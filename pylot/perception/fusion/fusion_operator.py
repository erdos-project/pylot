from collections import deque
import erdust
import numpy as np
import time

from pylot.perception.messages import ObjPositionsSpeedsMessage
from pylot.utils import time_epoch_ms


class FusionOperator(erdust.Operator):
    """Fusion Operator

    Args:
        rgbd_max_range (float): Maximum distance of the rgbd frame
        camera_fov (float): Angular field of view in radians of the RGBD and
            RGB cameras used to infer depth information and generate bounding
            boxes respectively. Note that camera position, orientation, and
            FOV must be identical for both.
    """
    def __init__(self,
                 can_bus_stream,
                 obstacles_stream,
                 depth_camera_stream,
                 fused_stream,
                 name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None,
                 camera_fov=np.pi / 4,
                 rgbd_max_range=1000):
        self.can_bus_stream = can_bus_stream
        self.obstacles_stream = obstacles_stream
        self.depth_camera_stream = depth_camera_stream
        can_bus_stream.add_callback(self.update_pos)
        obstacles_stream.add_callback(self.update_objects)
        depth_camera_stream.add_callback(self.update_distances)
        self._fused_stream = fused_stream
        self._name = name
        self._logger = erdust.utils.setup_logging(name, log_file_name)
        self._csv_logger = erdust.utils.setup_csv_logging(
            name + '-csv', csv_file_name)
        self._flags = flags
        self._segments = []
        self._objs = []
        self._rgbd_max_range = rgbd_max_range
        # TODO(ionel): Check fov is same as the camera fov.
        self._camera_fov = camera_fov
        self._car_positions = deque()
        self._distances = deque()
        self._objects = deque()

    @staticmethod
    def connect(can_bus_stream, obstacles_stream, depth_camera_stream):
        fused_stream = erdust.WriteStream()
        return [fused_stream]

    def __calc_object_positions(self, object_bounds, distances, car_position,
                                car_orientation):
        object_positions = []
        for bounds in object_bounds:
            i_min, i_max, j_min, j_max = bounds

            bounding_box_center = np.average([[i_min, i_max], [j_min, j_max]],
                                             axis=1)

            distance = np.median(distances[i_min:i_max, j_min:j_max])
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

            object_positions.append([position_x, position_y])

        return object_positions

    def __discard_old_data(self):
        """Discards stored data that are too old to be used for fusion"""
        oldest_timestamp = min([
            self._car_positions[-1][0], self._distances[-1][0],
            self._objects[-1][0]
        ])
        for queue in [self._car_positions, self._distances, self._objects]:
            while queue[0][0] < oldest_timestamp:
                queue.popleft()

    def fuse(self):
        # Return if we don't have car position, distances or objects.
        start_time = time.time()
        if min(map(
                len,
            [self._car_positions, self._distances, self._objects])) == 0:
            return
        self.__discard_old_data()
        object_positions = self.__calc_object_positions(
            self._objects[0][1], self._distances[0][1],
            self._car_positions[0][1][0],
            np.arccos(self._car_positions[0][1][1][0]))
        timestamp = self._objects[0][0]

        # Get runtime in ms.
        runtime = (time.time() - start_time) * 1000
        self._csv_logger.info('{},{},{}'.format(time_epoch_ms(), self._name,
                                                runtime))

        output_msg = ObjPositionsSpeedsMessage(object_positions, timestamp)
        self._fused_stream.send(output_msg)

    def update_pos(self, msg):
        vehicle_pos = ((msg.data.transform.location.x,
                        msg.data.transform.location.y,
                        msg.data.transform.location.z),
                       (msg.data.transform.forward_vector.x,
                        msg.data.transform.forward_vector.y,
                        msg.data.transform.forward_vector.z))
        self._car_positions.append((msg.timestamp, vehicle_pos))

    def update_objects(self, msg):
        # Filter objects
        self._logger.info("Received update objects")
        vehicle_bounds = []
        for detected_object in msg.detected_objects:
            self._logger.info("%s received: %s ", self._name, detected_object)
            # TODO(ionel): Deal with different types of labels.
            if detected_object.label in {"truck", "car"}:
                vehicle_bounds.append(detected_object.corners)
        self._objects.append((msg.timestamp, vehicle_bounds))

    def update_distances(self, msg):
        self._distances.append((msg.timestamp, msg.frame))

    def run(self):
        while True:
            can_bus_msg = self.can_bus_stream.read()
            obstacles_msg = self.obstacles_stream.read()
            depth_camera_msg = self.depth_camera_stream.read()
            self.update_pos(can_bus_msg)
            self.update_objects(obstacles_msg)
            self.update_distances(depth_camera_msg)
            self.fuse()
