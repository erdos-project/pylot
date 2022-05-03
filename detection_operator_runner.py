import glob
import os
import sys
import threading
from pylot.perception import depth_estimation

from setuptools import setup

try:
    sys.path.append(
        glob.glob('../carla/dist/carla-*%d.%d-%s.egg' %
                  (sys.version_info.major, sys.version_info.minor,
                   'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random
import time

import erdos
import logging
from absl import flags, app

import pylot.flags
import pylot.utils
import pylot.simulation.utils
from pylot.perception.camera_frame import CameraFrame
from pylot.perception.depth_frame import DepthFrame
from pylot.perception.segmentation.segmented_frame import SegmentedFrame
from pylot.drivers.sensor_setup import RGBCameraSetup, DepthCameraSetup, SegmentedCameraSetup

_lock = threading.Lock()

FLAGS = flags.FLAGS
flags.DEFINE_enum(
    'test_operator',
    'detection_operator', [
        'detection_operator', 'detection_eval', 'detection_decay',
        'traffic_light', 'efficient_det', 'lanenet', 'canny_lane',
        'depth_estimation', 'qd_track', 'segmentation_decay',
        'segmentation_drn', 'segmentation_eval', 'collision_sensor'
    ],
    help='Operator of choice to test')

CENTER_CAMERA_LOCATION = pylot.utils.Location(1.0, 0.0, 1.8)


def setup_camera(world, camera_setup, vehicle):
    """Sets up camera given world, camera_setup, and vehicle to attach to."""
    bp = world.get_blueprint_library().find(camera_setup.camera_type)
    bp.set_attribute('image_size_x', str(camera_setup.width))
    bp.set_attribute('image_size_y', str(camera_setup.height))
    bp.set_attribute('fov', str(camera_setup.fov))
    bp.set_attribute('sensor_tick', str(1.0 / 20))

    transform = camera_setup.get_transform().as_simulator_transform()
    print("Spawning a {} camera: {}".format(camera_setup.name, camera_setup))
    return world.spawn_actor(bp, transform, attach_to=vehicle)


def add_carla_callback(carla_sensor, setup, stream):
    def callback(simulator_data):
        if getattr(setup, 'camera_type') == 'sensor.camera.rgb':
            """Invoked when an rgb image is received from the simulator."""
            game_time = int(simulator_data.timestamp * 1000)
            timestamp = erdos.Timestamp(coordinates=[game_time])
            watermark_msg = erdos.WatermarkMessage(timestamp)

            with _lock:
                msg = None
                msg = erdos.Message(timestamp=timestamp,
                                    data=CameraFrame.from_simulator_frame(
                                        simulator_data, setup))
                stream.send(msg)
                # ttd_ingest_stream.send(erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))  Panics on internal msg call
        elif getattr(setup, 'camera_type') == 'sensor.camera.depth':
            """Invoked when a depth image is received from the simulator."""
            game_time = int(simulator_data.timestamp * 1000)
            timestamp = erdos.Timestamp(coordinates=[game_time])
            watermark_msg = erdos.WatermarkMessage(timestamp)

            with _lock:
                msg = erdos.Message(
                    timestamp=timestamp,
                    data=DepthFrame.from_simulator_frame(
                        simulator_data,
                        setup,
                        save_original_frame=FLAGS.visualize_depth_camera))
                stream.send(msg)
        elif getattr(setup,
                     'camera_type') == 'sensor.camera.semantic_segmentation':
            """Invoked when a segmented image is received from the simulator."""
            game_time = int(simulator_data.timestamp * 1000)
            timestamp = erdos.Timestamp(coordinates=[game_time])
            watermark_msg = erdos.WatermarkMessage(timestamp)

            with _lock:
                msg = erdos.Message(timestamp=timestamp,
                                    data=SegmentedFrame.from_simulator_image(
                                        simulator_data, setup))
                stream.send(msg)
        else:
            assert False, 'camera_type {} not supported'.format(
                getattr(setup, 'camera_type'))

    carla_sensor.listen(callback)


def main(args):
    actor_list = []

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()

        bp = world.get_blueprint_library().filter('vehicle.lincoln.mkz2017')[0]

        # Get random spawn position
        transform = random.choice(world.get_map().get_spawn_points())

        # Spawn lincoln vehicle
        vehicle = world.spawn_actor(bp, transform)

        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        # Let's put the vehicle to drive around
        vehicle.set_autopilot(True)

        transform = pylot.utils.Transform(CENTER_CAMERA_LOCATION,
                                          pylot.utils.Rotation())

        rgb_camera_setup = RGBCameraSetup('center_camera',
                                          FLAGS.camera_image_width,
                                          FLAGS.camera_image_height, transform,
                                          FLAGS.camera_fov)

        depth_camera_setup = DepthCameraSetup('depth_center_camera',
                                              FLAGS.camera_image_width,
                                              FLAGS.camera_image_height,
                                              transform, FLAGS.camera_fov)

        seg_camera_setup = SegmentedCameraSetup('seg_center_camera',
                                                FLAGS.camera_image_width,
                                                FLAGS.camera_image_height,
                                                transform, FLAGS.camera_fov)

        (left_camera_setup, right_camera_setup) = \
        pylot.drivers.sensor_setup.create_left_right_camera_setups(
            'camera',
            transform.location,
            FLAGS.camera_image_width,
            FLAGS.camera_image_height,
            FLAGS.offset_left_right_cameras,
            FLAGS.camera_fov)

        rgb_camera = setup_camera(world, rgb_camera_setup, vehicle)
        depth_camera = setup_camera(world, depth_camera_setup, vehicle)
        seg_camera = setup_camera(world, seg_camera_setup, vehicle)

        left_camera = setup_camera(world, left_camera_setup, vehicle)
        right_camera = setup_camera(world, right_camera_setup, vehicle)

        rgb_camera_ingest_stream = erdos.streams.IngestStream(
            name='rgb_camera')
        depth_camera_ingest_stream = erdos.streams.IngestStream(
            name='depth_camera')
        seg_camera_ingest_stream = erdos.streams.IngestStream(
            name='seg_camera')
        left_camera_ingest_stream = erdos.streams.IngestStream(
            name='left_camera')
        right_camera_ingest_stream = erdos.streams.IngestStream(
            name='right_camera')
        ttd_ingest_stream = erdos.streams.IngestStream(name='ttd')
        ground_obstacles_stream = erdos.streams.IngestStream(
            name='ground_obstacles_stream')
        vehicle_id_stream = erdos.streams.IngestStream(name='vehicle_id')

        DETECTOR = FLAGS.test_operator
        if DETECTOR == 'detection_operator':
            from pylot.perception.detection.detection_operator import DetectionOperator
            detection_op_cfg = erdos.operator.OperatorConfig(
                name='detection_op')
            obstacles_stream = erdos.connect_two_in_one_out(
                DetectionOperator,
                detection_op_cfg,
                rgb_camera_ingest_stream,
                ttd_ingest_stream,
                model_path=FLAGS.obstacle_detection_model_paths[0],
                flags=FLAGS)
        if DETECTOR == 'detection_eval':
            from pylot.perception.detection.detection_operator import DetectionOperator
            detection_op_cfg = erdos.operator.OperatorConfig(
                name='detection_op')
            obstacles_stream = erdos.connect_two_in_one_out(
                DetectionOperator,
                detection_op_cfg,
                rgb_camera_ingest_stream,
                ttd_ingest_stream,
                model_path=FLAGS.obstacle_detection_model_paths[0],
                flags=FLAGS)

            from pylot.perception.detection.detection_eval_operator import DetectionEvalOperator
            detection_eval_cfg = erdos.operator.OperatorConfig(
                name='detection_eval_op')
            metrics_stream = erdos.connect_two_in_one_out(
                DetectionEvalOperator,
                detection_eval_cfg,
                obstacles_stream,
                ground_obstacles_stream,
                evaluate_timely=False,
                matching_policy='ceil',
                frame_gap=None,
                flags=FLAGS)
        if DETECTOR == 'detection_decay':
            from pylot.perception.detection.detection_operator import DetectionOperator
            detection_op_cfg = erdos.operator.OperatorConfig(
                name='detection_op')
            obstacles_stream = erdos.connect_two_in_one_out(
                DetectionOperator,
                detection_op_cfg,
                rgb_camera_ingest_stream,
                ttd_ingest_stream,
                model_path=FLAGS.obstacle_detection_model_paths[0],
                flags=FLAGS)

            flags.DEFINE_integer(
                'decay_max_latency', 400,
                'Max latency to evaluate in ground truth experiments')

            from pylot.perception.detection.detection_decay_operator import DetectionDecayOperator
            detection_decay_op_cfg = erdos.operator.OperatorConfig(
                name='detection_decay_op')
            map_results = erdos.connect_one_in_one_out(DetectionDecayOperator,
                                                       detection_decay_op_cfg,
                                                       obstacles_stream,
                                                       flags=FLAGS)
        if DETECTOR == 'traffic_light':
            from pylot.perception.detection.traffic_light_det_operator import TrafficLightDetOperator
            traffic_light_op_cfg = erdos.operator.OperatorConfig(
                name='traffic_light_op')
            traffic_light_stream = erdos.connect_two_in_one_out(
                TrafficLightDetOperator,
                traffic_light_op_cfg,
                rgb_camera_ingest_stream,
                ttd_ingest_stream,
                flags=FLAGS)
        if DETECTOR == 'efficient_det':
            from pylot.perception.detection.efficientdet_operator import EfficientDetOperator
            model_names = ['efficientdet-d4']
            model_paths = [
                'dependencies/models/obstacle_detection/efficientdet/efficientdet-d4/efficientdet-d4_frozen.pb'
            ]
            efficient_det_op_cfg = erdos.operator.OperatorConfig(
                name='efficientdet_operator')
            efficient_det_stream = erdos.connect_two_in_one_out(
                EfficientDetOperator,
                efficient_det_op_cfg,
                rgb_camera_ingest_stream,
                ttd_ingest_stream,
                model_names=model_names,
                model_paths=model_paths,
                flags=FLAGS)
        if DETECTOR == 'lanenet':
            from pylot.perception.detection.lanenet_detection_operator import LanenetDetectionOperator
            lanenet_lane_detection_op_cfg = erdos.operator.OperatorConfig(
                name='lanenet_lane_detection')
            detected_lanes_stream = erdos.connect_one_in_one_out(
                LanenetDetectionOperator,
                lanenet_lane_detection_op_cfg,
                rgb_camera_ingest_stream,
                flags=FLAGS)
        if DETECTOR == 'canny_lane':
            from pylot.perception.detection.lane_detection_canny_operator import CannyEdgeLaneDetectionOperator
            lane_detection_canny_op_cfg = erdos.operator.OperatorConfig(
                name='lane_detection_canny_op')
            detected_lanes_stream = erdos.connect_one_in_one_out(
                CannyEdgeLaneDetectionOperator,
                lane_detection_canny_op_cfg,
                rgb_camera_ingest_stream,
                flags=FLAGS)
        if DETECTOR == 'depth_estimation':
            from pylot.perception.depth_estimation.depth_estimation_operator import DepthEstimationOperator
            depth_estimation_op_cfg = erdos.operator.OperatorConfig(
                name='depth_estimation_op')
            _ = erdos.connect_two_in_one_out(
                DepthEstimationOperator,
                depth_estimation_op_cfg,
                left_camera_ingest_stream,
                right_camera_ingest_stream,
                transform=depth_camera_setup.get_transform(),
                fov=FLAGS.camera_fov,
                flags=FLAGS)
        if DETECTOR == 'qd_track':
            from pylot.perception.tracking.qd_track_operator import QdTrackOperator
            qd_track_op_cfg = erdos.operator.OperatorConfig(name='qd_track_op')
            obstacles_stream = erdos.connect_one_in_one_out(
                QdTrackOperator, qd_track_op_cfg, rgb_camera_ingest_stream,
                FLAGS, rgb_camera_setup)
        if DETECTOR == 'segmentation_decay':
            from pylot.perception.segmentation.segmentation_decay_operator import SegmentationDecayOperator
            flags.DEFINE_integer(
                'decay_max_latency', 400,
                'Max latency to evaluate in ground truth experiments')
            segmentation_decay_op_cfg = erdos.operator.OperatorConfig(
                name='segmentation_decay_op')
            iou_stream = erdos.connect_one_in_one_out(
                SegmentationDecayOperator,
                segmentation_decay_op_cfg,
                seg_camera_ingest_stream,
                flags=FLAGS)
        if DETECTOR == 'segmentation_drn':
            from pylot.perception.segmentation.segmentation_drn_operator import SegmentationDRNOperator
            segmentation_drn_op_cfg = erdos.operator.OperatorConfig(
                name='segmentation_drn_op')
            segmented_stream = erdos.connect_one_in_one_out(
                SegmentationDRNOperator,
                segmentation_drn_op_cfg,
                rgb_camera_ingest_stream,
                flags=FLAGS)
        if DETECTOR == 'segmentation_eval':
            from pylot.perception.segmentation.segmentation_drn_operator import SegmentationDRNOperator
            segmentation_drn_op_cfg = erdos.operator.OperatorConfig(
                name='segmentation_drn_op')
            segmented_stream = erdos.connect_one_in_one_out(
                SegmentationDRNOperator,
                segmentation_drn_op_cfg,
                rgb_camera_ingest_stream,
                flags=FLAGS)
            from pylot.perception.segmentation.segmentation_eval_operator import SegmentationEvalOperator
            segmentation_eval_op_cfg = erdos.operator.OperatorConfig(
                name='segmentation_eval')
            _ = erdos.connect_two_in_one_out(SegmentationEvalOperator,
                                             segmentation_eval_op_cfg,
                                             seg_camera_ingest_stream,
                                             segmented_stream,
                                             flags=FLAGS)
        if DETECTOR == 'collision_sensor':
            from pylot.drivers.carla_collision_sensor_operator import CarlaCollisionSensorDriverOperator
            collision_op_cfg = erdos.operator.OperatorConfig(name='collision')
            collision_stream = erdos.connect_one_in_one_out(
                CarlaCollisionSensorDriverOperator,
                collision_op_cfg,
                vehicle_id_stream,
                flags=FLAGS)

        erdos.run_async()

        vehicle_id_stream.send(
            erdos.Message(erdos.Timestamp(coordinates=[0]), vehicle.id))
        vehicle_id_stream.send(
            erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

        # Register camera frame callbacks
        add_carla_callback(rgb_camera, rgb_camera_setup,
                           rgb_camera_ingest_stream)
        add_carla_callback(depth_camera, depth_camera_setup,
                           depth_camera_ingest_stream)
        add_carla_callback(seg_camera, seg_camera_setup,
                           seg_camera_ingest_stream)
        add_carla_callback(left_camera, left_camera_setup,
                           left_camera_ingest_stream)
        add_carla_callback(right_camera, right_camera_setup,
                           right_camera_ingest_stream)

        # Spawn 20 test vehicles
        pylot.simulation.utils.spawn_vehicles(client, world, 8000, 20,
                                              logging.Logger(name="test"))

        # Spawn 100 people
        pylot.simulation.utils.spawn_people(client, world, 100,
                                            logging.Logger(name="test2"))

        time.sleep(5)

    finally:
        print('destroying actors')
        rgb_camera.destroy()
        depth_camera.destroy()
        seg_camera.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')


if __name__ == '__main__':
    app.run(main)
