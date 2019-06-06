import json
import numpy as np
import PIL.Image as Image
from absl import app
from absl import flags
from collections import deque

import pylot.config
from pylot.perception.detection.utils import get_bounding_boxes_from_segmented, visualize_ground_bboxes
from pylot.perception.segmentation.utils import get_traffic_sign_pixels, transform_to_cityscapes_palette
import pylot.simulation.utils
import pylot.operator_creator
import pylot.utils

import erdos.graph
from erdos.op import Op
from erdos.utils import setup_csv_logging, setup_logging

FLAGS = flags.FLAGS
CENTER_CAMERA_NAME = 'front_rgb_camera'
DEPTH_CAMERA_NAME = 'front_depth_camera'
SEGMENTED_CAMERA_NAME = 'front_semantic_camera'

# Flags that control what data is recorded.
flags.DEFINE_string('data_path', 'data/',
                    'Path where to store Carla camera images')
flags.DEFINE_integer('log_every_nth_frame', 1,
                     'Control how often the script logs frames')


class CameraLoggerOp(Op):
    def __init__(self, name, flags, log_file_name=None, csv_file_name=None):
        super(CameraLoggerOp, self).__init__(name)
        self._flags = flags
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._last_bgr_timestamp = -1
        self._last_segmented_timestamp = -1

    @staticmethod
    def setup_streams(input_streams):
        input_streams.filter(pylot.utils.is_camera_stream).add_callback(
            CameraLoggerOp.on_bgr_frame)
        input_streams.filter(
            pylot.utils.is_ground_segmented_camera_stream).add_callback(
                CameraLoggerOp.on_segmented_frame)
        return []

    def on_bgr_frame(self, msg):
        # Ensure we didn't skip a frame.
        if self._last_bgr_timestamp != -1:
            assert self._last_bgr_timestamp + 1 == msg.timestamp.coordinates[1]
        self._last_bgr_timestamp = msg.timestamp.coordinates[1]
        if self._last_bgr_timestamp % self._flags.log_every_nth_frame != 0:
            return
        # Write the image.
        assert msg.encoding == 'BGR', 'Expects BGR frames'
        rgb_array = pylot.utils.bgr_to_rgb(msg.frame)
        file_name = '{}carla-{}.png'.format(
            self._flags.data_path, self._last_bgr_timestamp)
        rgb_img = Image.fromarray(np.uint8(rgb_array))
        rgb_img.save(file_name)

    def on_segmented_frame(self, msg):
        # Ensure we didn't skip a frame.
        if self._last_segmented_timestamp != -1:
            assert self._last_segmented_timestamp + 1 == msg.timestamp.coordinates[1]
        self._last_segmented_timestamp = msg.timestamp.coordinates[1]
        if self._last_bgr_timestamp % self._flags.log_every_nth_frame != 0:
            return
        frame = transform_to_cityscapes_palette(msg.frame)
        # Write the segmented image.
        img = Image.fromarray(np.uint8(frame))
        file_name = '{}carla-segmented-{}.png'.format(
            self._flags.data_path, self._last_segmented_timestamp)
        img.save(file_name)


class GroundTruthObjectLoggerOp(Op):
    def __init__(self,
                 name,
                 bgr_camera_setup,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        super(GroundTruthObjectLoggerOp, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._flags = flags
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
        self._last_notification = -1

    @staticmethod
    def setup_streams(input_streams):
        input_streams.filter(pylot.utils.is_depth_camera_stream).add_callback(
            GroundTruthObjectLoggerOp.on_depth_camera_update)
        input_streams.filter(pylot.utils.is_camera_stream).add_callback(
            GroundTruthObjectLoggerOp.on_bgr_camera_update)
        input_streams.filter(
            pylot.utils.is_ground_segmented_camera_stream).add_callback(
                GroundTruthObjectLoggerOp.on_segmented_frame)
        input_streams.filter(
            pylot.utils.is_can_bus_stream).add_callback(
                GroundTruthObjectLoggerOp.on_can_bus_update)
        input_streams.filter(
            pylot.utils.is_ground_pedestrians_stream).add_callback(
                GroundTruthObjectLoggerOp.on_pedestrians_update)
        input_streams.filter(
            pylot.utils.is_ground_vehicles_stream).add_callback(
                GroundTruthObjectLoggerOp.on_vehicles_update)
        input_streams.add_completion_callback(
            GroundTruthObjectLoggerOp.on_notification)
        return []

    def on_notification(self, msg):
        # Check that we didn't skip any notification. We only skip
        # notifications if messages or watermarks are lost.
        if self._last_notification != -1:
            assert self._last_notification + 1 == msg.timestamp.coordinates[1]
        self._last_notification = msg.timestamp.coordinates[1]

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

        assert (depth_msg.timestamp == bgr_msg.timestamp ==
                segmented_msg.timestamp == can_bus_msg.timestamp ==
                pedestrians_msg.timestamp == vehicles_msg.timestamp)

        if self._last_notification % self._flags.log_every_nth_frame != 0:
            return

        depth_array = depth_msg.frame
        vehicle_transform = can_bus_msg.data.transform

        ped_bboxes = self.__get_pedestrians_bboxes(
            pedestrians_msg.pedestrians, vehicle_transform, depth_array)

        vec_bboxes = self.__get_vehicles_bboxes(
            vehicles_msg.vehicles, vehicle_transform, depth_array)

        traffic_sign_bboxes = self.__get_traffic_sign_bboxes(
            segmented_msg.frame)

        bboxes = ped_bboxes + vec_bboxes + traffic_sign_bboxes
        # Write the bounding boxes.
        file_name = '{}bboxes-{}.json'.format(
            self._flags.data_path, self._last_notification)
        with open(file_name, 'w') as outfile:
            json.dump(bboxes, outfile)

        if self._flags.visualize_ground_obstacles:
            ped_vis_bboxes = [(xmin, xmax, ymin, ymax)
                              for (_, ((xmin, ymin), (xmax, ymax))) in ped_bboxes]
            vec_vis_bboxes = [(xmin, xmax, ymin, ymax)
                              for (_, ((xmin, ymin), (xmax, ymax))) in vec_bboxes]
            traffic_sign_vis_bboxes = [(xmin, xmax, ymin, ymax)
                                        for (_, ((xmin, ymin), (xmax, ymax))) in traffic_sign_bboxes]
            visualize_ground_bboxes(self.name, bgr_msg.timestamp, bgr_msg.frame,
                                    ped_vis_bboxes, vec_vis_bboxes,
                                    traffic_sign_vis_bboxes)

    def on_can_bus_update(self, msg):
        self._can_bus_msgs.append(msg)

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

    def execute(self):
        self.spin()

    def __get_pedestrians_bboxes(self, pedestrians, vehicle_transform,
                                 depth_array):
        bboxes = []
        for pedestrian in pedestrians:
            bbox = pylot.simulation.utils.get_2d_bbox_from_3d_box(
                depth_array, vehicle_transform, pedestrian.transform,
                pedestrian.bounding_box, self._bgr_transform, self._bgr_intrinsic,
                self._bgr_img_size, 1.5, 3.0)
            if bbox is not None:
                (xmin, xmax, ymin, ymax) = bbox
                bboxes.append(('pedestrian', ((xmin, ymin), (xmax, ymax))))
        return bboxes

    def __get_vehicles_bboxes(self, vehicles, vehicle_transform, depth_array):
        vec_bboxes = []
        for vehicle in vehicles:
            bbox = pylot.simulation.utils.get_2d_bbox_from_3d_box(
                depth_array, vehicle_transform, vehicle.transform,
                vehicle.bounding_box, self._bgr_transform, self._bgr_intrinsic,
                self._bgr_img_size, 3.0, 3.0)
            if bbox is not None:
                (xmin, xmax, ymin, ymax) = bbox
                vec_bboxes.append(('vehicle', ((xmin, ymin), (xmax, ymax))))
        return vec_bboxes

    def __get_traffic_sign_bboxes(self, segmented_frame):
        traffic_signs_frame = get_traffic_sign_pixels(segmented_frame)
        bboxes = get_bounding_boxes_from_segmented(traffic_signs_frame)
        traffic_sign_bboxes = [('traffic sign', bbox) for bbox in bboxes]
        return traffic_sign_bboxes


def create_camera_setups():
    location = pylot.simulation.utils.Location(2.0, 0.0, 1.4)
    rotation = pylot.simulation.utils.Rotation(0, 0, 0)
    transform = pylot.simulation.utils.Transform(location, rotation)
    rgb_camera_setup = pylot.simulation.utils.CameraSetup(
        CENTER_CAMERA_NAME,
        'sensor.camera.rgb',
        FLAGS.carla_camera_image_width,
        FLAGS.carla_camera_image_height,
        transform)
    depth_camera_setup = pylot.simulation.utils.CameraSetup(
        DEPTH_CAMERA_NAME,
        'sensor.camera.depth',
        FLAGS.carla_camera_image_width,
        FLAGS.carla_camera_image_height,
        transform)
    segmented_camera_setup = pylot.simulation.utils.CameraSetup(
        SEGMENTED_CAMERA_NAME,
        'sensor.camera.semantic_segmentation',
        FLAGS.carla_camera_image_width,
        FLAGS.carla_camera_image_height,
        transform)
    return (rgb_camera_setup, depth_camera_setup, segmented_camera_setup)


def main(argv):

    # Define graph
    graph = erdos.graph.get_current_graph()

    bgr_camera_setup, depth_camera_setup, segmented_camera_setup = create_camera_setups()
    camera_setups = [bgr_camera_setup,
                     depth_camera_setup,
                     segmented_camera_setup]

    logging_ops = []
    # Add an operator that logs BGR frames and segmented frames.
    camera_logger_op = graph.add(
        CameraLoggerOp,
        name='camera_logger_op',
        init_args={'flags': FLAGS,
                   'log_file_name': FLAGS.log_file_name,
                   'csv_file_name': FLAGS.csv_log_file_name})
    logging_ops.append(camera_logger_op)

    # Add operator that converts from 3D bounding boxes
    # to 2D bouding boxes.
    ground_object_logger_op = graph.add(
        GroundTruthObjectLoggerOp,
        name='ground_truth_obj_logger',
        init_args={'bgr_camera_setup': bgr_camera_setup,
                   'flags': FLAGS,
                   'log_file_name': FLAGS.log_file_name,
                   'csv_file_name': FLAGS.csv_log_file_name})
    logging_ops.append(ground_object_logger_op)

    # Add operator that interacts with the Carla simulator.
    carla_op = None
    if '0.8' in FLAGS.carla_version:
        carla_op = pylot.operator_creator.create_carla_legacy_op(
            graph, camera_setups, [])
        camera_ops = [carla_op]
    elif '0.9' in FLAGS.carla_version:
        carla_op = pylot.operator_creator.create_carla_op(graph)
        camera_ops = [pylot.operator_creator.create_camera_driver_op(graph, cs)
                      for cs in camera_setups]
        graph.connect([carla_op], camera_ops)
    else:
        raise ValueError(
            'Unexpected Carla version {}'.format(FLAGS.carla_version))
    graph.connect([carla_op] + camera_ops, logging_ops)

    agent_op = pylot.operator_creator.create_ground_agent_op(graph)
    graph.connect([carla_op], [agent_op])
    graph.connect([agent_op], [carla_op])

    # Add agent that uses ground data to drive around.
    goal_location = (234.269989014, 59.3300170898, 39.4306259155)
    goal_orientation = (1.0, 0.0, 0.22)

    if '0.8' in FLAGS.carla_version:
        waypointer_op = pylot.operator_creator.create_waypointer_op(
            graph, goal_location, goal_orientation)
        graph.connect([carla_op], [waypointer_op])
        graph.connect([waypointer_op], [agent_op])
    elif '0.9' in FLAGS.carla_version:
        planning_op = pylot.operator_creator.create_planning_op(
            graph, goal_location)
        graph.connect([carla_op], [planning_op])
        graph.connect([planning_op], [agent_op])

    graph.execute(FLAGS.framework)


if __name__ == '__main__':
    app.run(main)
