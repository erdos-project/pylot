from collections import deque
import erdos

from pylot.perception.messages import ObstaclesMessage
from pylot.simulation.utils import get_world, get_traffic_lights_obstacles


class PerfectTrafficLightDetectorOperator(erdos.Operator):
    """ Operator that uses info received from the simulator to perfectly detect
    traffic lights.

    Attributes:
        _town_name: Name of the Carla town.
        _traffic_lights: Buffer of ground traffic lights messages.
        _bgr_msgs: Buffer of ground camera messages.
        _depth_frame_msgs: Buffer of ground camera depth messages.
        _segmented_imgs: Buffer of ground segmented messages.
        _can_bus_msgs: Buffer of can bus messages.
    """
    def __init__(self,
                 ground_traffic_lights_stream,
                 tl_camera_stream,
                 depth_camera_stream,
                 segmented_camera_stream,
                 can_bus_stream,
                 traffic_lights_stream,
                 name,
                 flags,
                 log_file_name=None):
        ground_traffic_lights_stream.add_callback(self.on_traffic_light_update)
        tl_camera_stream.add_callback(self.on_bgr_camera_update)
        depth_camera_stream.add_callback(self.on_depth_camera_update)
        segmented_camera_stream.add_callback(self.on_segmented_frame)
        can_bus_stream.add_callback(self.on_can_bus_update)
        erdos.add_watermark_callback([
            ground_traffic_lights_stream, tl_camera_stream,
            depth_camera_stream, segmented_camera_stream, can_bus_stream
        ], [traffic_lights_stream], self.on_watermark)
        self._name = name
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._flags = flags
        _, world = get_world(self._flags.carla_host, self._flags.carla_port,
                             self._flags.carla_timeout)
        if world is None:
            raise ValueError("There was an issue connecting to the simulator.")
        self._town_name = world.get_map().name

        self._traffic_lights = deque()
        self._bgr_msgs = deque()
        self._depth_frame_msgs = deque()
        self._segmented_imgs = deque()
        self._can_bus_msgs = deque()
        self._frame_cnt = 0

    @staticmethod
    def connect(ground_traffic_lights_stream, tl_camera_stream,
                depth_camera_stream, segmented_camera_stream, can_bus_stream):
        traffic_lights_stream = erdos.WriteStream()
        return [traffic_lights_stream]

    def on_watermark(self, timestamp, traffic_lights_stream):
        self._logger.debug('@{}: received watermark'.format(timestamp))
        traffic_light_msg = self._traffic_lights.popleft()
        bgr_msg = self._bgr_msgs.popleft()
        depth_msg = self._depth_frame_msgs.popleft()
        segmented_msg = self._segmented_imgs.popleft()
        can_bus_msg = self._can_bus_msgs.popleft()
        vehicle_transform = can_bus_msg.data.transform
        self._frame_cnt += 1
        if (hasattr(self._flags, 'log_every_nth_message')
                and self._frame_cnt % self._flags.log_every_nth_message != 0):
            # There's no point to run the perfect detector if collecting
            # data, and only logging every nth frame.
            traffic_lights_stream.send(ObstaclesMessage([], timestamp))
            return

        # The camera setup sent with the image is relative to the car, we need
        # to transform it relative to the world to detect traffic lights.
        depth_msg.frame.camera_setup.set_transform(
            vehicle_transform * depth_msg.frame.camera_setup.transform)

        det_traffic_lights = get_traffic_lights_obstacles(
            traffic_light_msg.traffic_lights, depth_msg.frame,
            segmented_msg.frame.as_numpy_array(), self._town_name)

        if (self._flags.visualize_ground_obstacles
                or self._flags.log_detector_output):
            bgr_msg.frame.annotate_with_bounding_boxes(bgr_msg.timestamp,
                                                       det_traffic_lights)
            if self._flags.visualize_ground_obstacles:
                bgr_msg.frame.visualize(self._name)
            if self._flags.log_detector_output:
                bgr_msg.frame.save(bgr_msg.timestamp.coordinates[0],
                                   self._flags.data_path, 'perfect-detector')

        # Send the detected traffic lights.
        traffic_lights_stream.send(
            ObstaclesMessage(det_traffic_lights, timestamp))

    def on_can_bus_update(self, msg):
        self._logger.debug('@{}: received can bus message'.format(
            msg.timestamp))
        self._can_bus_msgs.append(msg)

    def on_traffic_light_update(self, msg):
        self._logger.debug('@{}: received ground traffic lights update'.format(
            msg.timestamp))
        self._traffic_lights.append(msg)

    def on_bgr_camera_update(self, msg):
        self._logger.debug('@{}: received BGR frame'.format(msg.timestamp))
        self._bgr_msgs.append(msg)

    def on_depth_camera_update(self, msg):
        self._logger.debug('@{}: received depth frame'.format(msg.timestamp))
        self._depth_frame_msgs.append(msg)

    def on_segmented_frame(self, msg):
        self._logger.debug('@{}: received segmented frame'.format(
            msg.timestamp))
        self._segmented_imgs.append(msg)
