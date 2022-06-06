import glob
import os
import sys
import threading

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
import pylot.operator_creator
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
        'segmentation_drn', 'segmentation_eval', 'bounding_box_logger',
        'camera_logger', 'multiple_object_logger', 'collision_sensor',
<<<<<<< HEAD
        'object_tracker', 'gnss_sensor', 'imu_sensor', 'lane_invasion_sensor'
=======
        'object_tracker', 'linear_predictor'
>>>>>>> 70b1a6704194e466564b62f3ba966b85a82b7973
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
                stream.send(watermark_msg)
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
                stream.send(watermark_msg)
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
                stream.send(watermark_msg)
        else:
            assert False, 'camera_type {} not supported'.format(
                getattr(setup, 'camera_type'))

    carla_sensor.listen(callback)


def send_pose_message(stream: erdos.WriteStream, timestamp: erdos.Timestamp,
                      vehicle):
    vec_transform = pylot.utils.Transform.from_simulator_transform(
        vehicle.get_transform())
    velocity_vector = pylot.utils.Vector3D.from_simulator_vector(
        vehicle.get_velocity())
    forward_speed = velocity_vector.magnitude()
    pose = pylot.utils.Pose(vec_transform, forward_speed, velocity_vector,
                            timestamp.coordinates[0])
    stream.send(erdos.Message(timestamp, pose))
    stream.send(erdos.WatermarkMessage(timestamp))


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
        pose_stream = erdos.streams.IngestStream(name='pose_stream')

        if FLAGS.test_operator == 'detection_operator' or FLAGS.test_operator == 'object_tracker':
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
        if FLAGS.test_operator == 'detection_eval':
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
        if FLAGS.test_operator == 'detection_decay':
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
        if FLAGS.test_operator == 'traffic_light':
            from pylot.perception.detection.traffic_light_det_operator import TrafficLightDetOperator
            traffic_light_op_cfg = erdos.operator.OperatorConfig(
                name='traffic_light_op')
            traffic_light_stream = erdos.connect_two_in_one_out(
                TrafficLightDetOperator,
                traffic_light_op_cfg,
                rgb_camera_ingest_stream,
                ttd_ingest_stream,
                flags=FLAGS)
        if FLAGS.test_operator == 'efficient_det':
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
        if FLAGS.test_operator == 'lanenet':
            from pylot.perception.detection.lanenet_detection_operator import LanenetDetectionOperator
            lanenet_lane_detection_op_cfg = erdos.operator.OperatorConfig(
                name='lanenet_lane_detection')
            detected_lanes_stream = erdos.connect_one_in_one_out(
                LanenetDetectionOperator,
                lanenet_lane_detection_op_cfg,
                rgb_camera_ingest_stream,
                flags=FLAGS)
        if FLAGS.test_operator == 'canny_lane':
            from pylot.perception.detection.lane_detection_canny_operator import CannyEdgeLaneDetectionOperator
            lane_detection_canny_op_cfg = erdos.operator.OperatorConfig(
                name='lane_detection_canny_op')
            detected_lanes_stream = erdos.connect_one_in_one_out(
                CannyEdgeLaneDetectionOperator,
                lane_detection_canny_op_cfg,
                rgb_camera_ingest_stream,
                flags=FLAGS)
        if FLAGS.test_operator == 'depth_estimation':
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
        if FLAGS.test_operator == 'qd_track':
            from pylot.perception.tracking.qd_track_operator import QdTrackOperator
            qd_track_op_cfg = erdos.operator.OperatorConfig(name='qd_track_op')
            obstacles_stream = erdos.connect_one_in_one_out(
                QdTrackOperator, qd_track_op_cfg, rgb_camera_ingest_stream,
                FLAGS, rgb_camera_setup)
        if FLAGS.test_operator == 'object_tracker':
            pylot.operator_creator.add_obstacle_tracking(
                obstacles_stream, rgb_camera_ingest_stream, ttd_ingest_stream)
        if FLAGS.test_operator == 'segmentation_decay':
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
        if FLAGS.test_operator == 'segmentation_drn':
            from pylot.perception.segmentation.segmentation_drn_operator import SegmentationDRNOperator
            segmentation_drn_op_cfg = erdos.operator.OperatorConfig(
                name='segmentation_drn_op')
            segmented_stream = erdos.connect_one_in_one_out(
                SegmentationDRNOperator,
                segmentation_drn_op_cfg,
                rgb_camera_ingest_stream,
                flags=FLAGS)
        if FLAGS.test_operator == 'segmentation_eval':
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
        if FLAGS.test_operator == 'bounding_box_logger':
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

            from pylot.loggers.bounding_box_logger_operator import BoundingBoxLoggerOperator
            detection_logger_cfg = erdos.operator.OperatorConfig(
                name='detection_logger_op')
            finished_indicator_stream = erdos.connect_one_in_one_out(
                BoundingBoxLoggerOperator, detection_logger_cfg,
                obstacles_stream, FLAGS, './')
        if FLAGS.test_operator == 'camera_logger':
            from pylot.loggers.camera_logger_operator import CameraLoggerOperator
            camera_logger_op_cfg = erdos.operator.OperatorConfig(
                name='camera_logger_op')
            finished_indicator_stream = erdos.connect_one_in_one_out(
                CameraLoggerOperator, camera_logger_op_cfg,
                rgb_camera_ingest_stream, FLAGS, 'testing')
        if FLAGS.test_operator == 'multiple_object_logger':
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

            from pylot.loggers.multiple_object_tracker_logger_operator import MultipleObjectTrackerLoggerOperator
            multiple_object_logger_cfg = erdos.operator.OperatorConfig(
                name='multiple_object_logger_op')
            finished_indicator_stream = erdos.connect_one_in_one_out(
                MultipleObjectTrackerLoggerOperator,
                multiple_object_logger_cfg, obstacles_stream, FLAGS)
        if FLAGS.test_operator == 'collision_sensor':
            from pylot.drivers.carla_collision_sensor_operator import CarlaCollisionSensorDriverOperator
            collision_op_cfg = erdos.operator.OperatorConfig(name='collision')
            collision_stream = erdos.connect_one_in_one_out(
                CarlaCollisionSensorDriverOperator,
                collision_op_cfg,
                vehicle_id_stream,
                flags=FLAGS)
        if FLAGS.test_operator == 'gnss_sensor':
            from pylot.drivers.carla_gnss_driver_operator import CarlaGNSSDriverOperator
            gnss_op_cfg = erdos.operator.OperatorConfig(name='gnss')
            gnss_setup = pylot.drivers.sensor_setup.GNSSSetup('gnss', transform)
            gnss_stream = erdos.connect_one_in_one_out(
                CarlaGNSSDriverOperator,
                gnss_op_cfg,
                vehicle_id_stream,
                gnss_setup,
                flags=FLAGS)
        if FLAGS.test_operator == 'imu_sensor':
            from pylot.drivers.carla_imu_driver_operator import CarlaIMUDriverOperator
            imu_op_cfg = erdos.operator.OperatorConfig(name='imu')
            imu_setup = pylot.drivers.sensor_setup.IMUSetup('imu', transform)
            imu_stream = erdos.connect_one_in_one_out(
                CarlaIMUDriverOperator,
                imu_op_cfg,
                vehicle_id_stream,
                imu_setup,
                flags=FLAGS)
        if FLAGS.test_operator == 'lane_invasion_sensor':
            from pylot.drivers.carla_lane_invasion_sensor_operator import CarlaLaneInvasionSensorDriverOperator
            lane_invasion_op_cfg = erdos.operator.OperatorConfig(name='simulator_lane_invasion_sensor_operator')
            lane_invasion_stream = erdos.connect_one_in_one_out(
                CarlaLaneInvasionSensorDriverOperator,
                lane_invasion_op_cfg,
                vehicle_id_stream,
                flags=FLAGS)
        if FLAGS.test_operator == 'linear_predictor':
            time_to_decision_loop_stream = erdos.streams.LoopStream()

            from pylot.perception.detection.detection_operator import DetectionOperator
            detection_op_cfg = erdos.operator.OperatorConfig(
                name='detection_op')
            obstacles_stream = erdos.connect_two_in_one_out(
                DetectionOperator,
                detection_op_cfg,
                rgb_camera_ingest_stream,
                time_to_decision_loop_stream,
                model_path=FLAGS.obstacle_detection_model_paths[0],
                flags=FLAGS)

            tracked_obstacles = pylot.operator_creator.add_obstacle_location_history(
                obstacles_stream, depth_camera_ingest_stream, pose_stream,
                depth_camera_setup)

            time_to_decision_stream = pylot.operator_creator.add_time_to_decision(
                pose_stream, obstacles_stream)
            time_to_decision_loop_stream.connect_loop(time_to_decision_stream)

            linear_prediction_stream = pylot.operator_creator.add_linear_prediction(
                tracked_obstacles, time_to_decision_loop_stream)

        erdos.run_async()

        ttd_ingest_stream.send(
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

        vehicle_id_stream.send(
            erdos.Message(erdos.Timestamp(coordinates=[0]), vehicle.id))
        vehicle_id_stream.send(
            erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

        def _send_pose_message(simulator_data):
            with _lock:
                game_time = int(simulator_data.elapsed_seconds * 1000)
                timestamp = erdos.Timestamp(coordinates=[game_time])

                send_pose_message(pose_stream, timestamp, vehicle)

        world.on_tick(_send_pose_message)

        time.sleep(5)

    finally:
        print('destroying actors')
        rgb_camera.destroy()
        depth_camera.destroy()
        seg_camera.destroy()
        left_camera.destroy()
        right_camera.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')


if __name__ == '__main__':
    app.run(main)
