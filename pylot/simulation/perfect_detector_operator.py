from collections import deque
import threading

from erdos.message import WatermarkMessage
from erdos.op import Op
from erdos.utils import setup_csv_logging, setup_logging

import pylot.utils
from pylot.perception.detection.utils import DetectedObject,\
    TrafficLightColor, get_bounding_boxes_from_segmented,\
    visualize_ground_bboxes
from pylot.perception.messages import DetectorMessage
from pylot.perception.segmentation.utils import get_traffic_sign_pixels
from pylot.simulation.utils import get_2d_bbox_from_3d_box,\
    batch_get_3d_world_position_with_depth_map,\
    match_bboxes_with_traffic_lights


class PerfectDetectorOp(Op):
    """ Operator that transforms information it receives from Carla into
    perfect bounding boxes.

    Attributes:
        _output_stream_name: Name of the stream to output detected objects to.
        _bgr_imgs: Buffer of received ground BGR image messages.
        _can_bus_msgs: Buffer of received ground can bus messages.
        _depth_imgs: Buffer of received depth image messages.
        _pedestrians: Buffer of pedestrian messages received from Carla.
        _vehicles: Buffer of pedestrian messages received from Carla.
        _segmented_imgs: Buffer of segmented frame msgs received from Carla.
    """
    def __init__(self,
                 name,
                 output_stream_name,
                 bgr_camera_setup,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        """ Initializes the operator.

        Args:
            bgr_camera_setup: A simulation.utils.CameraSetup object
        """
        super(PerfectDetectorOp, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._flags = flags
        self._output_stream_name = output_stream_name
        # Queues of incoming data.
        self._bgr_imgs = deque()
        self._can_bus_msgs = deque()
        self._depth_imgs = deque()
        self._pedestrians = deque()
        self._segmented_imgs = deque()
        self._traffic_lights = deque()
        self._vehicles = deque()
        self._speed_limit_signs = deque()
        self._stop_signs = deque()
        self._bgr_intrinsic = bgr_camera_setup.get_intrinsic()
        self._bgr_transform = bgr_camera_setup.get_unreal_transform()
        self._bgr_img_size = (bgr_camera_setup.width, bgr_camera_setup.height)
        self._lock = threading.Lock()

    @staticmethod
    def setup_streams(input_streams, output_stream_name):
        # Register a callback on depth frames data stream.
        input_streams.filter(pylot.utils.is_depth_camera_stream).add_callback(
            PerfectDetectorOp.on_depth_camera_update)
        # Register a callback on BGR frames data stream.
        input_streams.filter(pylot.utils.is_center_camera_stream).add_callback(
            PerfectDetectorOp.on_bgr_camera_update)
        # Register a callback on segmented frames data stream.
        input_streams.filter(
            pylot.utils.is_segmented_camera_stream).add_callback(
                PerfectDetectorOp.on_segmented_frame)
        # Register a callback on can bus messages data stream.
        input_streams.filter(
            pylot.utils.is_can_bus_stream).add_callback(
                PerfectDetectorOp.on_can_bus_update)
        # Register a callback to receive pedestrian updates from Carla.
        input_streams.filter(
            pylot.utils.is_ground_pedestrians_stream).add_callback(
                PerfectDetectorOp.on_pedestrians_update)
        # Register a callback to receive vehicle updates from Carla.
        input_streams.filter(
            pylot.utils.is_ground_vehicles_stream).add_callback(
                PerfectDetectorOp.on_vehicles_update)
        # Register a completion watermark callback. The callback is invoked
        # after all the messages with a given timestamp have been received.
        input_streams.add_completion_callback(
            PerfectDetectorOp.on_notification)
        # Register a callback on traffic lights data stream.
        input_streams.filter(
            pylot.utils.is_ground_traffic_lights_stream).add_callback(
                PerfectDetectorOp.on_traffic_light_update)
        # Register a callback on speed limits data stream.
        input_streams.filter(
            pylot.utils.is_ground_speed_limit_signs_stream).add_callback(
                PerfectDetectorOp.on_speed_limit_signs_update)
        # Register a callback on stop signs data stream.
        input_streams.filter(
            pylot.utils.is_ground_stop_signs_stream).add_callback(
                PerfectDetectorOp.on_stop_signs_update)
        # Stream on which to output bounding boxes.
        return [pylot.utils.create_obstacles_stream(output_stream_name)]

    def synchronize_msg_buffers(self, timestamp, buffers):
        for buffer in buffers:
            while (len(buffer) > 0 and buffer[0].timestamp < timestamp):
                buffer.popleft()
            if len(buffer) == 0:
                return False
            assert buffer[0].timestamp == timestamp
        return True

    def on_notification(self, msg):
        # Pop the oldest message from each buffer.
        with self._lock:
            if not self.synchronize_msg_buffers(
                    msg.timestamp,
                    [self._depth_imgs, self._bgr_imgs, self._segmented_imgs,
                     self._can_bus_msgs, self._pedestrians, self._vehicles,
                     self._traffic_lights, self._speed_limit_signs,
                     self._stop_signs]):
                return
            depth_msg = self._depth_imgs.popleft()
            bgr_msg = self._bgr_imgs.popleft()
            segmented_msg = self._segmented_imgs.popleft()
            can_bus_msg = self._can_bus_msgs.popleft()
            pedestrians_msg = self._pedestrians.popleft()
            vehicles_msg = self._vehicles.popleft()
            traffic_light_msg = self._traffic_lights.popleft()
            speed_limit_signs_msg = self._speed_limit_signs.popleft()
            stop_signs_msg = self._stop_signs.popleft()

        self._logger.info('Timestamps {} {} {} {} {} {}'.format(
            depth_msg.timestamp, bgr_msg.timestamp, segmented_msg.timestamp,
            can_bus_msg.timestamp, pedestrians_msg.timestamp,
            vehicles_msg.timestamp, traffic_light_msg.timestamp))

        # The popper messages should have the same timestamp.
        assert (depth_msg.timestamp == bgr_msg.timestamp ==
                segmented_msg.timestamp == can_bus_msg.timestamp ==
                pedestrians_msg.timestamp == vehicles_msg.timestamp ==
                traffic_light_msg.timestamp)

        depth_array = depth_msg.frame
        vehicle_transform = can_bus_msg.data.transform

        det_ped = self.__get_pedestrians(
            pedestrians_msg.pedestrians, vehicle_transform, depth_array)

        det_vec = self.__get_vehicles(
            vehicles_msg.vehicles, vehicle_transform, depth_array)

        det_traffic_signs = self.__get_traffic_signs(segmented_msg.frame)
        det_traffic_lights = self.__get_traffic_lights(
            traffic_light_msg.traffic_lights,
            vehicle_transform,
            depth_msg,
            segmented_msg.frame)
        # Send the detected obstacles.
        output_msg = DetectorMessage(det_ped + det_vec + det_traffic_signs +
                                     det_traffic_lights,
                                     0, msg.timestamp)

        self.get_output_stream(self._output_stream_name).send(output_msg)
        # Send watermark on the output stream because operators do not
        # automatically forward watermarks when they've registed an
        # on completion callback.
        self.get_output_stream(self._output_stream_name)\
            .send(WatermarkMessage(msg.timestamp))

        if self._flags.visualize_ground_obstacles:
            ped_bboxes = [det_obj.corners for det_obj in det_ped]
            vec_bboxes = [det_obj.corners for det_obj in det_vec]
            traffic_sign_bboxes = [det_obj.corners
                                   for det_obj in det_traffic_signs]
            traffic_light_bboxes = [(det_obj.corners, det_obj.label)
                                    for det_obj in det_traffic_lights]
            visualize_ground_bboxes(self.name,
                                    bgr_msg.timestamp,
                                    bgr_msg.frame,
                                    ped_bboxes,
                                    vec_bboxes,
                                    traffic_sign_bboxes,
                                    traffic_light_bboxes)

    def on_can_bus_update(self, msg):
        with self._lock:
            self._can_bus_msgs.append(msg)

    def on_traffic_light_update(self, msg):
        with self._lock:
            self._traffic_lights.append(msg)

    def on_speed_limit_signs_update(self, msg):
        with self._lock:
            self._speed_limit_signs.append(msg)

    def on_stop_signs_update(self, msg):
        with self._lock:
            self._stop_signs.append(msg)

    def on_pedestrians_update(self, msg):
        with self._lock:
            self._pedestrians.append(msg)

    def on_vehicles_update(self, msg):
        with self._lock:
            self._vehicles.append(msg)

    def on_depth_camera_update(self, msg):
        with self._lock:
            self._depth_imgs.append(msg)

    def on_bgr_camera_update(self, msg):
        with self._lock:
            self._bgr_imgs.append(msg)

    def on_segmented_frame(self, msg):
        with self._lock:
            self._segmented_imgs.append(msg)

    def execute(self):
        self.spin()

    def __get_pedestrians(self, pedestrians, vehicle_transform, depth_array):
        """ Transforms pedestrians into detected objects.
        Args:
            pedestrians: List of Pedestrian objects.
            vehicle_transform: Ego-vehicle transform.
            depth_array: Depth frame taken at the time when pedestrians were
                         collected.
        """
        det_objs = []
        for pedestrian in pedestrians:
            bbox = get_2d_bbox_from_3d_box(
                depth_array, vehicle_transform, pedestrian.transform,
                pedestrian.bounding_box, self._bgr_transform,
                self._bgr_intrinsic, self._bgr_img_size, 1.5, 3.0)
            if bbox is not None:
                det_objs.append(DetectedObject(bbox, 1.0, 'pedestrian'))
        return det_objs

    def __get_vehicles(self, vehicles, vehicle_transform, depth_array):
        """ Transforms vehicles into detected objects.
        Args:
            vehicles: List of Vehicle objects.
            vehicle_transform: Ego-vehicle transform.
            depth_array: Depth frame taken at the time when pedestrians were
                         collected.
        """
        det_objs = []
        for vehicle in vehicles:
            bbox = get_2d_bbox_from_3d_box(
                depth_array, vehicle_transform, vehicle.transform,
                vehicle.bounding_box, self._bgr_transform, self._bgr_intrinsic,
                self._bgr_img_size, 3.0, 3.0)
            if bbox is not None:
                det_objs.append(DetectedObject(bbox, 1.0, 'vehicle'))
        return det_objs

    def __get_traffic_signs(self, segmented_frame):
        """ Extracts traffic signs bounding boxes out of a segmented frame."""
        traffic_signs_frame = get_traffic_sign_pixels(segmented_frame)
        bboxes = get_bounding_boxes_from_segmented(traffic_signs_frame)
        det_objs = [DetectedObject(bbox, 1.0, 'traffic sign')
                    for bbox in bboxes]
        return det_objs

    def __get_traffic_lights(self,
                             traffic_lights,
                             vehicle_transform,
                             depth_msg,
                             segmented_frame):
        # Get 3d world positions for all traffic signs (some of which are
        # traffic lights).
        traffic_signs_frame = get_traffic_sign_pixels(segmented_frame)
        bboxes = get_bounding_boxes_from_segmented(traffic_signs_frame)

        # Get the positions of the bounding box centers.
        x_mids = [(bbox[0] + bbox[1]) / 2 for bbox in bboxes]
        y_mids = [(bbox[2] + bbox[3]) / 2 for bbox in bboxes]
        pos_3d = batch_get_3d_world_position_with_depth_map(
            x_mids, y_mids, depth_msg.frame, depth_msg.width, depth_msg.height,
            depth_msg.fov, vehicle_transform * depth_msg.transform)
        pos_and_bboxes = zip(pos_3d, bboxes)

        # Map traffic lights to bounding boxes based on 3d world position.
        tl_bboxes = match_bboxes_with_traffic_lights(
            vehicle_transform, pos_and_bboxes, traffic_lights)
        det_objs = []

        for bbox, color in tl_bboxes:
            if color == TrafficLightColor.GREEN:
                det_objs.append(
                    DetectedObject(bbox, 1.0, 'green traffic light'))
            elif color == TrafficLightColor.YELLOW:
                det_objs.append(
                    DetectedObject(bbox, 1.0, 'yellow traffic light'))
            elif color == TrafficLightColor.RED:
                det_objs.append(
                    DetectedObject(bbox, 1.0, 'red traffic light'))
            else:
                det_objs.append(
                    DetectedObject(bbox, 1.0, 'off traffic light'))
        return det_objs
