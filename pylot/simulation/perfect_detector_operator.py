from collections import deque
import erdust

import pylot.utils
from pylot.perception.detection.utils import DetectedObject,\
    annotate_image_with_bboxes, save_image, visualize_image
from pylot.perception.messages import DetectorMessage
from pylot.simulation.utils import get_2d_bbox_from_3d_box,\
    camera_to_unreal_transform
from pylot.simulation.carla_utils import get_world


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
                 ground_pedestrians_stream,
                 ground_vehicles_stream,
                 ground_traffic_lights_stream,
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
        ground_pedestrians_stream.add_callback(self.on_pedestrians_update)
        ground_vehicles_stream.add_callback(self.on_vehicles_update)
        ground_traffic_lights_stream.add_callback(self.on_traffic_light_update)
        ground_speed_limit_signs_stream.add_callback(
            self.on_speed_limit_signs_update)
        ground_stop_signs_stream.add_callback(self.on_stop_signs_update)
        # Register a completion watermark callback. The callback is invoked
        # after all the messages with a given timestamp have been received.
        erdust.add_watermark_callback(
            [depth_camera_stream,
             center_camera_stream,
             segmented_camera_stream,
             can_bus_stream,
             ground_pedestrians_stream,
             ground_vehicles_stream,
             ground_traffic_lights_stream,
             ground_speed_limit_signs_stream,
             ground_stop_signs_stream],
            [obstacles_stream],
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
        self._frame_cnt = 0

    @staticmethod
    def connect(depth_camera_stream,
                center_camera_stream,
                segmented_camera_stream,
                can_bus_stream,
                ground_pedestrians_stream,
                ground_vehicles_stream,
                ground_traffic_lights_stream,
                ground_speed_limit_signs_stream,
                ground_stop_signs_stream):
        obstacles_stream = erdust.WriteStream()
        # Stream on which to output bounding boxes.
        return [obstacles_stream]

    def on_watermark(self, timestamp, obstacles_stream):
        depth_msg = self._depth_imgs.popleft()
        bgr_msg = self._bgr_imgs.popleft()
        segmented_msg = self._segmented_imgs.popleft()
        can_bus_msg = self._can_bus_msgs.popleft()
        pedestrians_msg = self._pedestrians.popleft()
        vehicles_msg = self._vehicles.popleft()
        traffic_light_msg = self._traffic_lights.popleft()
        speed_limit_signs_msg = self._speed_limit_signs.popleft()
        stop_signs_msg = self._stop_signs.popleft()
        self._frame_cnt += 1
        if (hasattr(self._flags, 'log_every_nth_frame') and
            self._frame_cnt % self._flags.log_every_nth_frame != 0):
            # There's no point to run the perfect detector if collecting
            # data, and only logging every nth frame.
            obstacles_stream.send(DetectorMessage([], 0, timestamp))
            return
        depth_array = depth_msg.frame
        segmented_image = segmented_msg.frame
        vehicle_transform = can_bus_msg.data.transform

        det_ped = self.__get_pedestrians(pedestrians_msg.pedestrians,
                                         vehicle_transform, depth_array,
                                         segmented_image)

        det_vec = self.__get_vehicles(vehicles_msg.vehicles, vehicle_transform,
                                      depth_array, segmented_image)

        det_traffic_lights = pylot.simulation.utils.get_traffic_light_det_objs(
            traffic_light_msg.traffic_lights,
            vehicle_transform * camera_to_unreal_transform(depth_msg.transform),
            depth_msg.frame,
            segmented_image,
            depth_msg.width,
            depth_msg.height,
            self._town_name,
            fov=depth_msg.fov)

        det_speed_limits = pylot.simulation.utils.get_speed_limit_det_objs(
            speed_limit_signs_msg.speed_signs,
            vehicle_transform,
            vehicle_transform * depth_msg.transform,
            depth_msg.frame, depth_msg.width, depth_msg.height,
            depth_msg.fov, segmented_msg.frame)

        det_stop_signs = pylot.simulation.utils.get_traffic_stop_det_objs(
            stop_signs_msg.stop_signs,
            vehicle_transform * camera_to_unreal_transform(depth_msg.transform),
            depth_msg.frame, depth_msg.width, depth_msg.height, depth_msg.fov)

        det_objs = (det_ped + det_vec + det_traffic_lights +
                    det_speed_limits + det_stop_signs)

        # Send the detected obstacles.
        obstacles_stream.send(DetectorMessage(det_objs, 0, timestamp))

        if (self._flags.visualize_ground_obstacles or
            self._flags.log_detector_output):
            annotate_image_with_bboxes(
                bgr_msg.timestamp, bgr_msg.frame, det_objs)
            if self._flags.visualize_ground_obstacles:
                visualize_image(self.name, bgr_msg.frame)
            if self._flags.log_detector_output:
                save_image(pylot.utils.bgr_to_rgb(bgr_msg.frame),
                           bgr_msg.timestamp,
                           self._flags.data_path,
                           'perfect-detector')

    def on_can_bus_update(self, msg):
        self._can_bus_msgs.append(msg)

    def on_traffic_light_update(self, msg):
        self._traffic_lights.append(msg)

    def on_speed_limit_signs_update(self, msg):
        self._speed_limit_signs.append(msg)

    def on_stop_signs_update(self, msg):
        self._stop_signs.append(msg)

    def on_pedestrians_update(self, msg):
        self._pedestrians.append(msg)

    def on_vehicles_update(self, msg):
        self._vehicles.append(msg)

    def on_depth_camera_update(self, msg):
        self._depth_imgs.append(msg)

    def on_bgr_camera_update(self, msg):
        self._bgr_imgs.append(msg)

    def on_segmented_frame(self, msg):
        self._segmented_imgs.append(msg)

    def __get_pedestrians(self, pedestrians, vehicle_transform, depth_array,
                          segmented_image):
        """ Transforms pedestrians into detected objects.
        Args:
            pedestrians: List of Pedestrian objects.
            vehicle_transform: Ego-vehicle transform.
            depth_array: Depth frame taken at the time when pedestrians were
                         collected.
            segmented_image: The segmentation frame taken at the time when
                the pedestrians were collected.
        """
        det_objs = []
        for pedestrian in pedestrians:
            bbox = get_2d_bbox_from_3d_box(
                vehicle_transform, pedestrian.transform,
                pedestrian.bounding_box, self._bgr_transform,
                self._bgr_intrinsic, self._bgr_img_size, depth_array,
                segmented_image, 4)
            if bbox is not None:
                det_objs.append(DetectedObject(bbox, 1.0, 'pedestrian', pedestrian.id))
        return det_objs

    def __get_vehicles(self, vehicles, vehicle_transform, depth_array,
                       segmented_image):
        """ Transforms vehicles into detected objects.
        Args:
            vehicles: List of Vehicle objects.
            vehicle_transform: Ego-vehicle transform.
            depth_array: Depth frame taken at the time when vehicles were
                         collected.
            segmented_image: The segmentation frame taken at the time when
                the vehicles were collected.
        """
        det_objs = []
        for vehicle in vehicles:
            bbox = get_2d_bbox_from_3d_box(
                vehicle_transform, vehicle.transform, vehicle.bounding_box,
                self._bgr_transform, self._bgr_intrinsic, self._bgr_img_size,
                depth_array, segmented_image, 10)
            if bbox is not None:
                det_objs.append(DetectedObject(bbox, 1.0, 'vehicle', vehicle.id))
        return det_objs
