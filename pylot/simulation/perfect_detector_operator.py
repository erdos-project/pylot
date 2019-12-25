from collections import deque
import erdust

import pylot.utils
from pylot.perception.detection.utils import DetectedObject,\
    annotate_image_with_bboxes, save_image, visualize_image
from pylot.perception.messages import DetectorMessage
from pylot.simulation.sensor_setup import DepthCameraSetup, RGBCameraSetup


class PerfectDetectorOperator(erdust.Operator):
    """ Operator that transforms information it receives from Carla into
    perfect bounding boxes.

    Attributes:
        _bgr_imgs: Buffer of received ground BGR image messages.
        _can_bus_msgs: Buffer of received ground can bus messages.
        _depth_imgs: Buffer of received depth image messages.
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
            bgr_camera_setup: A simulation.sensor_setup.CameraSetup object
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
        erdust.add_watermark_callback([
            depth_camera_stream, center_camera_stream, segmented_camera_stream,
            can_bus_stream, ground_obstacles_stream,
            ground_speed_limit_signs_stream, ground_stop_signs_stream
        ], [obstacles_stream], self.on_watermark)

        self._name = name
        self._logger = erdust.utils.setup_logging(name, log_file_name)
        self._flags = flags
        # Queues of incoming data.
        self._bgr_imgs = deque()
        self._can_bus_msgs = deque()
        self._depth_imgs = deque()
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
        obstacles_stream = erdust.WriteStream()
        # Stream on which to output bounding boxes.
        return [obstacles_stream]

    def on_watermark(self, timestamp, obstacles_stream):
        self._logger.debug('@{}: received watermark'.format(timestamp))
        depth_msg = self._depth_imgs.popleft()
        bgr_msg = self._bgr_imgs.popleft()
        segmented_msg = self._segmented_imgs.popleft()
        can_bus_msg = self._can_bus_msgs.popleft()
        obstacles_msg = self._obstacles.popleft()
        speed_limit_signs_msg = self._speed_limit_signs.popleft()
        stop_signs_msg = self._stop_signs.popleft()
        self._frame_cnt += 1
        if (hasattr(self._flags, 'log_every_nth_frame')
                and self._frame_cnt % self._flags.log_every_nth_frame != 0):
            # There's no point to run the perfect detector if collecting
            # data, and only logging every nth frame.
            obstacles_stream.send(DetectorMessage([], 0, timestamp))
            return
        vehicle_transform = can_bus_msg.data.transform

        det_obstacles = self.__get_obstacles(obstacles_msg.obstacles,
                                             vehicle_transform,
                                             depth_msg.frame,
                                             segmented_msg.frame)

        # The camera setup sent with the image is relative to the car, we need
        # to transform it relative to the world.
        transformed_camera_setup = DepthCameraSetup(
            depth_msg.camera_setup.name,
            depth_msg.camera_setup.width,
            depth_msg.camera_setup.height,
            vehicle_transform * depth_msg.camera_setup.transform,
            fov=depth_msg.camera_setup.fov)

        det_speed_limits = pylot.simulation.utils.get_speed_limit_det_objs(
            speed_limit_signs_msg.speed_signs, vehicle_transform,
            depth_msg.frame, segmented_msg.frame, transformed_camera_setup)

        det_stop_signs = pylot.simulation.utils.get_traffic_stop_det_objs(
            stop_signs_msg.stop_signs, depth_msg.frame,
            transformed_camera_setup)

        det_objs = det_obstacles + det_speed_limits + det_stop_signs

        # Send the detected obstacles.
        obstacles_stream.send(DetectorMessage(det_objs, 0, timestamp))

        if (self._flags.visualize_ground_obstacles
                or self._flags.log_detector_output):
            annotate_image_with_bboxes(bgr_msg.timestamp, bgr_msg.frame,
                                       det_objs)
            if self._flags.visualize_ground_obstacles:
                visualize_image(self._name, bgr_msg.frame)
            if self._flags.log_detector_output:
                save_image(pylot.utils.bgr_to_rgb(bgr_msg.frame),
                           bgr_msg.timestamp, self._flags.data_path,
                           'perfect-detector')

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
        self._depth_imgs.append(msg)

    def on_bgr_camera_update(self, msg):
        self._logger.debug('@{}: received BGR frame'.format(msg.timestamp))
        self._bgr_imgs.append(msg)

    def on_segmented_frame(self, msg):
        self._logger.debug('@{}: received segmented frame'.format(
            msg.timestamp))
        self._segmented_imgs.append(msg)

    def __get_obstacles(self, obstacles, vehicle_transform, depth_array,
                        segmented_frame):
        """ Transforms obstacles into detected objects.
        Args:
            obstacles: List of pylot.simulation.util.Obstacle objects.
            vehicle_transform: Ego-vehicle transform.
            depth_array: The depth array taken at the time when obstacles were
                collected.
            segmented_image: SegmentedFrame taken at the time when the
                obstacles were collected.
        """
        det_objs = []
        for obstacle in obstacles:
            # We have a static camera setup, need to transform it with respect
            # to the location of the ego vehicle, before doing detection.
            transformed_camera_setup = RGBCameraSetup(
                self._camera_setup.name,
                self._camera_setup.width,
                self._camera_setup.height,
                vehicle_transform * self._camera_setup.get_transform(),
                fov=self._camera_setup.get_fov())

            # Calculate the distance of the obstacle from the vehicle, and
            # convert to camera view if it is less than 125 metres away.
            if obstacle.distance(vehicle_transform) > 125:
                bbox = None
            else:
                bbox = obstacle.to_camera_view(transformed_camera_setup,
                                               depth_array, segmented_frame)
            if bbox is not None:
                det_objs.append(
                    DetectedObject(bbox, 1.0, obstacle.label, obstacle.id))
        return det_objs
