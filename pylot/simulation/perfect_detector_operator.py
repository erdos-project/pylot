from collections import deque
import erdos

from pylot.perception.detection.utils import DetectedObstacle
from pylot.perception.messages import ObstaclesMessage
import pylot.simulation.utils


class PerfectDetectorOperator(erdos.Operator):
    """ Operator that transforms information it receives from Carla into
    perfect bounding boxes.

    Attributes:
        _bgr_imgs: Buffer of received ground BGR image messages.
        _can_bus_msgs: Buffer of received ground can bus messages.
        _depth_frame_msgs: Buffer of received depth frame messages.
        _pedestrians: Buffer of pedestrian messages received from Carla.
        _vehicles: Buffer of pedestrian messages received from Carla.
        _segmented_imgs: Buffer of segmented frame msgs received from Carla.
    """
    def __init__(self,
                 depth_camera_stream,
                 center_camera_stream,
                 segmented_camera_stream,
                 can_bus_stream,
                 ground_obstacles_stream,
                 ground_speed_limit_signs_stream,
                 ground_stop_signs_stream,
                 obstacles_stream,
                 name,
                 bgr_camera_setup,
                 flags,
                 log_file_name=None):
        """ Initializes the operator.

        Args:
            bgr_camera_setup: A simulation.sensor_setup.CameraSetup.
        """
        depth_camera_stream.add_callback(self.on_depth_camera_update)
        center_camera_stream.add_callback(self.on_bgr_camera_update)
        segmented_camera_stream.add_callback(self.on_segmented_frame)
        can_bus_stream.add_callback(self.on_can_bus_update)
        ground_obstacles_stream.add_callback(self.on_obstacles_update)
        ground_speed_limit_signs_stream.add_callback(
            self.on_speed_limit_signs_update)
        ground_stop_signs_stream.add_callback(self.on_stop_signs_update)
        # Register a completion watermark callback. The callback is invoked
        # after all the messages with a given timestamp have been received.
        erdos.add_watermark_callback([
            depth_camera_stream, center_camera_stream, segmented_camera_stream,
            can_bus_stream, ground_obstacles_stream,
            ground_speed_limit_signs_stream, ground_stop_signs_stream
        ], [obstacles_stream], self.on_watermark)

        self._name = name
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._flags = flags
        # Queues of incoming data.
        self._bgr_imgs = deque()
        self._can_bus_msgs = deque()
        self._depth_frame_msgs = deque()
        self._obstacles = deque()
        self._segmented_imgs = deque()
        self._speed_limit_signs = deque()
        self._stop_signs = deque()
        self._camera_setup = bgr_camera_setup
        self._frame_cnt = 0

    @staticmethod
    def connect(depth_camera_stream, center_camera_stream,
                segmented_camera_stream, can_bus_stream,
                ground_obstacles_stream, ground_speed_limit_signs_stream,
                ground_stop_signs_stream):
        obstacles_stream = erdos.WriteStream()
        # Stream on which to output bounding boxes.
        return [obstacles_stream]

    def on_watermark(self, timestamp, obstacles_stream):
        self._logger.debug('@{}: received watermark'.format(timestamp))
        depth_msg = self._depth_frame_msgs.popleft()
        bgr_msg = self._bgr_imgs.popleft()
        segmented_msg = self._segmented_imgs.popleft()
        can_bus_msg = self._can_bus_msgs.popleft()
        obstacles_msg = self._obstacles.popleft()
        speed_limit_signs_msg = self._speed_limit_signs.popleft()
        stop_signs_msg = self._stop_signs.popleft()
        self._frame_cnt += 1
        if (hasattr(self._flags, 'log_every_nth_message')
                and self._frame_cnt % self._flags.log_every_nth_message != 0):
            # There's no point to run the perfect detector if collecting
            # data, and only logging every nth frame.
            obstacles_stream.send(ObstaclesMessage([], timestamp))
            return
        vehicle_transform = can_bus_msg.data.transform

        # The camera setup sent with the image is relative to the car, we need
        # to transform it relative to the world.
        depth_msg.frame.camera_setup.set_transform(
            vehicle_transform * depth_msg.frame.camera_setup.transform)

        det_obstacles = self.__get_obstacles(obstacles_msg.obstacles,
                                             vehicle_transform,
                                             depth_msg.frame,
                                             segmented_msg.frame)

        det_speed_limits = pylot.simulation.utils.get_detected_speed_limits(
            speed_limit_signs_msg.speed_signs, depth_msg.frame,
            segmented_msg.frame)

        det_stop_signs = pylot.simulation.utils.get_detected_traffic_stops(
            stop_signs_msg.stop_signs, depth_msg.frame)

        det_obstacles = det_obstacles + det_speed_limits + det_stop_signs

        # Send the detected obstacles.
        obstacles_stream.send(ObstaclesMessage(det_obstacles, timestamp))

        if (self._flags.visualize_ground_obstacles
                or self._flags.log_detector_output):
            bgr_msg.frame.annotate_with_bounding_boxes(bgr_msg.timestamp,
                                                       det_obstacles)
            if self._flags.visualize_ground_obstacles:
                bgr_msg.frame.visualize(self._name)
            if self._flags.log_detector_output:
                bgr_msg.frame.save(bgr_msg.timestamp.coordinates[0],
                                   self._flags.data_path, 'perfect-detector')

    def on_can_bus_update(self, msg):
        self._logger.debug('@{}: received can bus message'.format(
            msg.timestamp))
        self._can_bus_msgs.append(msg)

    def on_speed_limit_signs_update(self, msg):
        self._logger.debug('@{}: received ground speed limits update'.format(
            msg.timestamp))
        self._speed_limit_signs.append(msg)

    def on_stop_signs_update(self, msg):
        self._logger.debug('@{}: received ground stop signs update'.format(
            msg.timestamp))
        self._stop_signs.append(msg)

    def on_obstacles_update(self, msg):
        self._logger.debug('@{}: received ground obstacles update'.format(
            msg.timestamp))
        self._obstacles.append(msg)

    def on_depth_camera_update(self, msg):
        self._logger.debug('@{}: received depth frame'.format(msg.timestamp))
        self._depth_frame_msgs.append(msg)

    def on_bgr_camera_update(self, msg):
        self._logger.debug('@{}: received BGR frame'.format(msg.timestamp))
        self._bgr_imgs.append(msg)

    def on_segmented_frame(self, msg):
        self._logger.debug('@{}: received segmented frame'.format(
            msg.timestamp))
        self._segmented_imgs.append(msg)

    def __get_obstacles(self, obstacles, vehicle_transform, depth_frame,
                        segmented_frame):
        """ Transforms obstacles into detected obstacles.

        Args:
            obstacles: List of pylot.perception.detection.obstacle.Obstacle.
            vehicle_transform: The transform of the ego vehicle.
            depth_frame: perception.depth_frame.DepthFrame taken at the
                time when obstacles were collected.
            segmented_frame: perception.segmentation.segmented_frame.SegmentedFrame
                taken at the time when the obstacles were collected.
        """
        det_obstacles = []
        for obstacle in obstacles:
            # Calculate the distance of the obstacle from the vehicle, and
            # convert to camera view if it is less than 125 metres away.
            if obstacle.distance(vehicle_transform) > 125:
                bbox = None
            else:
                bbox = obstacle.to_camera_view(depth_frame, segmented_frame)
                if bbox:
                    det_obstacles.append(
                        DetectedObstacle(bbox, 1.0, obstacle.label,
                                         obstacle.id))
        return det_obstacles
