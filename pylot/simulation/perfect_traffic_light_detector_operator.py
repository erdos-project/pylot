from collections import deque
import erdust

from pylot.perception.detection.utils import annotate_image_with_bboxes,\
    save_image, visualize_image
from pylot.perception.messages import DetectorMessage
from pylot.simulation.carla_utils import get_world
import pylot.simulation.utils


class PerfectTrafficLightDetectorOperator(erdust.Operator):
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
        erdust.add_watermark_callback(
            [ground_traffic_lights_stream,
             tl_camera_stream,
             depth_camera_stream,
             segmented_camera_stream,
             can_bus_stream],
            [traffic_lights_stream],
            self.on_watermark)
        self._name = name
        self._logger = erdust.utils.setup_logging(name, log_file_name)
        self._flags = flags
        _, world = get_world(self._flags.carla_host,
                             self._flags.carla_port,
                             self._flags.carla_timeout)
        if world is None:
            raise ValueError("There was an issue connecting to the simulator.")
        self._town_name = world.get_map().name

        self._traffic_lights = deque()
        self._bgr_imgs = deque()
        self._depth_imgs = deque()
        self._segmented_imgs = deque()
        self._can_bus_msgs = deque()
        self._frame_cnt = 0

    @staticmethod
    def connect(ground_traffic_lights_stream,
                tl_camera_stream,
                depth_camera_stream,
                segmented_camera_stream,
                can_bus_stream):
        traffic_lights_stream = erdust.WriteStream()
        return [traffic_lights_stream]

    def on_watermark(self, timestamp, traffic_lights_stream):
        traffic_light_msg = self._traffic_lights.popleft()
        bgr_msg = self._bgr_imgs.popleft()
        depth_msg = self._depth_imgs.popleft()
        segmented_msg = self._segmented_imgs.popleft()
        can_bus_msg = self._can_bus_msgs.popleft()
        vehicle_transform = can_bus_msg.data.transform
        self._frame_cnt += 1
        if (hasattr(self._flags, 'log_every_nth_frame') and
            self._frame_cnt % self._flags.log_every_nth_frame != 0):
            # There's no point to run the perfect detector if collecting
            # data, and only logging every nth frame.
            traffic_light_msg.send(DetectorMessage([], 0, timestamp))
            return

        det_traffic_lights = pylot.simulation.utils.get_traffic_light_det_objs(
            traffic_light_msg.traffic_lights,
            vehicle_transform * depth_msg.transform,
            depth_msg.frame,
            segmented_msg.frame.as_numpy_array(),
            depth_msg.width,
            depth_msg.height,
            self._town_name,
            fov=depth_msg.fov)

        if (self._flags.visualize_ground_obstacles or
            self._flags.log_detector_output):
            annotate_image_with_bboxes(
                bgr_msg.timestamp, bgr_msg.frame, det_traffic_lights)
            if self._flags.visualize_ground_obstacles:
                visualize_image(self.name, bgr_msg.frame)
            if self._flags.log_detector_output:
                save_image(pylot.utils.bgr_to_rgb(bgr_msg.frame),
                           bgr_msg.timestamp,
                           self._flags.data_path,
                           'perfect-detector')

        # Send the detected traffic lights.
        traffic_lights_stream.send(
            DetectorMessage(det_traffic_lights, 0, timestamp))

    def on_can_bus_update(self, msg):
        self._can_bus_msgs.append(msg)

    def on_traffic_light_update(self, msg):
        self._traffic_lights.append(msg)

    def on_bgr_camera_update(self, msg):
        self._bgr_imgs.append(msg)

    def on_depth_camera_update(self, msg):
        self._depth_imgs.append(msg)

    def on_segmented_frame(self, msg):
        self._segmented_imgs.append(msg)
