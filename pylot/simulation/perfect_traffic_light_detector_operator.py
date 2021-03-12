from collections import deque

import erdos
from erdos import Message, ReadStream, Timestamp, WriteStream

from pylot.perception.messages import TrafficLightsMessage
from pylot.simulation.utils import get_map, get_traffic_lights_obstacles


class PerfectTrafficLightDetectorOperator(erdos.Operator):
    """Uses info from the simulator to perfectly detect traffic lights.

    Args:
        ground_traffic_lights_stream (:py:class:`erdos.ReadStream`): Stream on
            which :py:class:`~pylot.perception.messages.TrafficLightsMessage`
            messages are received.
        tl_camera_stream (:py:class:`erdos.ReadStream`): Stream on which BGR
            frames are received.
        depth_camera_stream (:py:class:`erdos.ReadStream`): Stream on which
            :py:class:`~pylot.perception.messages.DepthFrameMessage` are
            received.
        segmented_camera__stream (:py:class:`erdos.ReadStream`):
            Stream on which
            :py:class:`~pylot.perception.messages.SegmentedFrameMessage`
            are received.
        pose_stream (:py:class:`erdos.ReadStream`): Stream on which pose
            info is received.
        traffic_lights_stream (:py:class:`erdos.WriteStream`): Stream on which
            the operator publishes
            :py:class:`~pylot.perception.messages.TrafficLightsMessage`
            messages for traffic lights.
        flags (absl.flags): Object to be used to access absl flags.

    Attributes:
        _town_name (:obj:`str`): Name of the simulator town.
        _traffic_lights (:obj:`collections.deque`): Buffer of ground traffic
            lights messages.
        _bgr_msgs (:obj:`collections.deque`): Buffer of ground camera messages.
        _depth_frame_msgs (:obj:`collections.deque`): Buffer of ground camera
            depth messages.
        _segmented_msgs (:obj:`collections.deque`): Buffer of ground segmented
            messages.
        _pose_msgs (:obj:`collections.deque`): Buffer of pose messages.
        _frame_cnt (:obj:`int`): Number of messages received.
    """
    def __init__(self, ground_traffic_lights_stream: ReadStream,
                 tl_camera_stream: ReadStream, depth_camera_stream: ReadStream,
                 segmented_camera_stream: ReadStream, pose_stream: ReadStream,
                 traffic_lights_stream: WriteStream, flags):
        ground_traffic_lights_stream.add_callback(self.on_traffic_light_update)
        tl_camera_stream.add_callback(self.on_bgr_camera_update)
        depth_camera_stream.add_callback(self.on_depth_camera_update)
        segmented_camera_stream.add_callback(self.on_segmented_frame)
        pose_stream.add_callback(self.on_pose_update)
        erdos.add_watermark_callback([
            ground_traffic_lights_stream, tl_camera_stream,
            depth_camera_stream, segmented_camera_stream, pose_stream
        ], [traffic_lights_stream], self.on_watermark)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags
        self._traffic_lights = deque()
        self._bgr_msgs = deque()
        self._depth_frame_msgs = deque()
        self._segmented_msgs = deque()
        self._pose_msgs = deque()
        self._frame_cnt = 0

    @staticmethod
    def connect(ground_traffic_lights_stream: ReadStream,
                tl_camera_stream: ReadStream, depth_camera_stream: ReadStream,
                segmented_camera_stream: ReadStream, pose_stream: ReadStream):
        traffic_lights_stream = erdos.WriteStream()
        return [traffic_lights_stream]

    def destroy(self):
        self._logger.warn('destroying {}'.format(self.config.name))

    def run(self):
        # Run method is invoked after all operators finished initializing.
        # Thus, we're sure the world is up-to-date here.
        world_map = get_map(self._flags.simulator_host,
                            self._flags.simulator_port,
                            self._flags.simulator_timeout)
        self._town_name = world_map.name

    @erdos.profile_method()
    def on_watermark(self, timestamp: Timestamp,
                     traffic_lights_stream: WriteStream):
        """Invoked when all input streams have received a watermark.

        Args:
            timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of
                the watermark.
        """
        self._logger.debug('@{}: received watermark'.format(timestamp))
        if timestamp.is_top:
            return
        traffic_light_msg = self._traffic_lights.popleft()
        bgr_msg = self._bgr_msgs.popleft()
        depth_msg = self._depth_frame_msgs.popleft()
        segmented_msg = self._segmented_msgs.popleft()
        pose_msg = self._pose_msgs.popleft()
        vehicle_transform = pose_msg.data.transform
        self._frame_cnt += 1
        if (hasattr(self._flags, 'log_every_nth_message')
                and self._frame_cnt % self._flags.log_every_nth_message != 0):
            # There's no point to run the perfect detector if collecting
            # data, and only logging every nth frame.
            traffic_lights_stream.send(TrafficLightsMessage(timestamp, []))
            return

        # The camera setup sent with the image is relative to the car, we need
        # to transform it relative to the world to detect traffic lights.
        depth_msg.frame.camera_setup.set_transform(
            vehicle_transform * depth_msg.frame.camera_setup.transform)

        det_traffic_lights = get_traffic_lights_obstacles(
            traffic_light_msg.obstacles, depth_msg.frame, segmented_msg.frame,
            self._town_name)

        # Filter out traffic lights that are far away.
        det_traffic_lights = [
            tl for tl in det_traffic_lights
            if tl.transform.location.distance(vehicle_transform.location) <=
            self._flags.static_obstacle_distance_threshold
        ]

        # Send the detected traffic lights.
        traffic_lights_stream.send(
            TrafficLightsMessage(timestamp, det_traffic_lights))

        if self._flags.log_detector_output:
            bgr_msg.frame.annotate_with_bounding_boxes(bgr_msg.timestamp,
                                                       det_traffic_lights,
                                                       vehicle_transform)
            bgr_msg.frame.save(bgr_msg.timestamp.coordinates[0],
                               self._flags.data_path, 'perfect-detector')

    def on_pose_update(self, msg: Message):
        self._logger.debug('@{}: received pose message'.format(msg.timestamp))
        self._pose_msgs.append(msg)

    def on_traffic_light_update(self, msg: Message):
        self._logger.debug('@{}: received ground traffic lights update'.format(
            msg.timestamp))
        self._traffic_lights.append(msg)

    def on_bgr_camera_update(self, msg: Message):
        self._logger.debug('@{}: received BGR frame'.format(msg.timestamp))
        self._bgr_msgs.append(msg)

    def on_depth_camera_update(self, msg: Message):
        self._logger.debug('@{}: received depth frame'.format(msg.timestamp))
        self._depth_frame_msgs.append(msg)

    def on_segmented_frame(self, msg: Message):
        self._logger.debug('@{}: received segmented frame'.format(
            msg.timestamp))
        self._segmented_msgs.append(msg)
