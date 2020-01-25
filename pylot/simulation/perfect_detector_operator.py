from collections import deque
import erdos

from pylot.perception.detection.utils import DetectedObstacle
from pylot.perception.messages import ObstaclesMessage
import pylot.simulation.utils


class PerfectDetectorOperator(erdos.Operator):
    """Uses info from the simulator to perfectly detect obstacles, stop and
    speed limit signs.

    Args:
        depth_camera_stream (:py:class:`erdos.ReadStream`): Stream on
            which :py:class:`~pylot.perception.messages.DepthFrameMessage` are
            received.
        center_camera_stream (:py:class:`erdos.ReadStream`): Stream on which
            RGB camera frames are received.
        segmented_camera__stream (:py:class:`erdos.ReadStream`): Stream on
            which :py:class:`~pylot.perception.messages.SegmentedFrameMessage`
            are received.
        can_bus_stream (:py:class:`erdos.ReadStream`):
            Stream on which can bus info is received.
        ground_obstacles_stream (:py:class:`erdos.ReadStream`): Stream on which
            :py:class:`~pylot.simulation.GroundObstaclesMessage` messages are
            received.
        ground_speed_limit_signs_stream (:py:class:`erdos.ReadStream`): Stream
            on which :py:class:`~pylot.simulation.GroundSpeedSignsMessage`
            messages are received.
        ground_stop_signs_stream (:py:class:`erdos.ReadStream`): Stream on
            which :py:class:`~pylot.simulation.GroundStopSignsMessage` messages
            are received.
        obstacles_stream (:py:class:`erdos.WriteStream`): Stream on which the
            operator publishes
            :py:class:`~pylot.perception.messages.ObstaclesMessage` messages
            for detected obstacles.
        name (:obj:`str`): The name of the operator.
        flags (absl.flags): Object to be used to access absl flags.
        log_file_name (:obj:`str`, optional): Name of file where log messages
            are written to. If None, then messages are written to stdout.

    Attributes:
        _bgr_msgs (:obj:`collections.deque`): Buffer of received ground BGR
            image messages.
        _can_bus_msgs (:obj:`collections.deque`): Buffer of received ground
            can bus messages.
        _depth_frame_msgs (:obj:`collections.deque`): Buffer of received depth
            frame messages.
        _obstacles (:obj:`collections.deque`): Buffer of obstacle messages
            received from Carla.
        _segmented_msgs (:obj:`collections.deque`): Buffer of segmented frame
            msgs received from Carla.
        _speed_limit_signs (:obj:`collections.deque`): Buffer of speed limit
            sign msgs.
        _stop_signs (:obj:`collections.deque`): Buffer of stop sign msgs
            received from Carla.
        _frame_cnt (:obj:`int`): Number of messages received.
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
                 flags,
                 log_file_name=None):
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
        self._bgr_msgs = deque()
        self._can_bus_msgs = deque()
        self._depth_frame_msgs = deque()
        self._obstacles = deque()
        self._segmented_msgs = deque()
        self._speed_limit_signs = deque()
        self._stop_signs = deque()
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
        """Invoked when all input streams have received a watermark.

        Args:
            timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of
                the watermark.
        """
        self._logger.debug('@{}: received watermark'.format(timestamp))
        depth_msg = self._depth_frame_msgs.popleft()
        bgr_msg = self._bgr_msgs.popleft()
        segmented_msg = self._segmented_msgs.popleft()
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

        if (self._flags.visualize_detected_obstacles
                or self._flags.log_detector_output):
            bgr_msg.frame.annotate_with_bounding_boxes(bgr_msg.timestamp,
                                                       det_obstacles)
            if self._flags.visualize_detected_obstacles:
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
        self._bgr_msgs.append(msg)

    def on_segmented_frame(self, msg):
        self._logger.debug('@{}: received segmented frame'.format(
            msg.timestamp))
        self._segmented_msgs.append(msg)

    def __get_obstacles(self, obstacles, vehicle_transform, depth_frame,
                        segmented_frame):
        """Transforms obstacles into detected obstacles.

        Args:
            obstacles (list(:py:class:`~pylot.perception.detection.obstacle.Obstacle`)):
                List of obstacles.
            vehicle_transform (:py:class:`~pylot.utils.Transform`): Transform of
                the ego vehicle.
            depth_frame (:py:class:`~pylot.perception.depth_frame.DepthFrame`):
                Depth frame taken at the time when obstacles were collected.
            segmented_frame (:py:class:`~pylot.perception.segmentation.segmented_frame.SegmentedFrame`):
                Segmented frame taken at the time when the obstacles were
                collected.

        Returns:
            list(:py:class:`~pylot.perception.detection.utils.DetectedObstacle`):
            List of detected obstacles.
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
                                         obstacle.id,
                                         obstacle.transform.location))
        return det_obstacles
