from collections import deque
import threading

from erdos.op import Op
from erdos.utils import setup_csv_logging, setup_logging

import pylot.utils
from pylot.perception.detection.utils import DetectedObject, get_bounding_boxes_from_segmented, visualize_ground_bboxes
from pylot.perception.messages import DetectorMessage
from pylot.perception.segmentation.utils import get_traffic_sign_pixels
from pylot.simulation.utils import get_2d_bbox_from_3d_box


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
        # Queue of incoming data.
        self._bgr_imgs = deque()
        self._can_bus_msgs = deque()
        self._depth_imgs = deque()
        self._pedestrians = deque()
        self._segmented_imgs = deque()
        self._vehicles = deque()
        self._bgr_intrinsic = bgr_camera_setup.get_intrinsic()
        self._bgr_transform = bgr_camera_setup.get_unreal_transform()
        self._bgr_img_size = (bgr_camera_setup.width, bgr_camera_setup.height)
        self._notification_cnt = 0
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
            pylot.utils.is_ground_segmented_camera_stream).add_callback(
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
        # Stream on which to output bounding boxes.
        return [pylot.utils.create_obstacles_stream(output_stream_name)]

    def on_notification(self, msg):
        # Pop the oldest message from each buffer.
        with self._lock:
            depth_msg = self._depth_imgs.popleft()
            bgr_msg = self._bgr_imgs.popleft()
            segmented_msg = self._segmented_imgs.popleft()
            can_bus_msg = self._can_bus_msgs.popleft()
            pedestrians_msg = self._pedestrians.popleft()
            vehicles_msg = self._vehicles.popleft()

        self._logger.info('Timestamps {} {} {} {} {} {}'.format(
            depth_msg.timestamp, bgr_msg.timestamp, segmented_msg.timestamp,
            can_bus_msg.timestamp, pedestrians_msg.timestamp,
            vehicles_msg.timestamp))

        # The popper messages should have the same timestamp.
        assert (depth_msg.timestamp == bgr_msg.timestamp ==
                segmented_msg.timestamp == can_bus_msg.timestamp ==
                pedestrians_msg.timestamp == vehicles_msg.timestamp)

        self._notification_cnt += 1
        if self._notification_cnt % self._flags.log_every_nth_frame != 0:
            return

        depth_array = depth_msg.frame
        vehicle_transform = can_bus_msg.data.transform

        det_ped = self.__get_pedestrians(
            pedestrians_msg.pedestrians, vehicle_transform, depth_array)

        det_vec = self.__get_vehicles(
            vehicles_msg.vehicles, vehicle_transform, depth_array)

        det_traffic_signs = self.__get_traffic_signs(segmented_msg.frame)

        # Send the detected obstacles.
        output_msg = DetectorMessage(det_ped + det_vec + det_traffic_signs,
                                     0, msg.timestamp)
        self.get_output_stream(self._output_stream_name).send(output_msg)

        if self._flags.visualize_ground_obstacles:
            ped_bboxes = [det_obj.corners for det_obj in det_ped]
            vec_bboxes = [det_obj.corners for det_obj in det_vec]
            traffic_sign_bboxes = [det_obj.corners
                                   for det_obj in det_traffic_signs]
            visualize_ground_bboxes(self.name,
                                    bgr_msg.timestamp,
                                    bgr_msg.frame,
                                    ped_bboxes,
                                    vec_bboxes,
                                    traffic_sign_bboxes)

    def on_can_bus_update(self, msg):
        with self._lock:
            self._can_bus_msgs.append(msg)

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
        traffic_signs_frame = get_traffic_sign_pixels(segmented_frame)
        bboxes = get_bounding_boxes_from_segmented(traffic_signs_frame)
        det_objs = [DetectedObject(bbox, 1.0, 'traffic sign')
                    for bbox in bboxes]
        return det_objs
