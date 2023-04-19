from __future__ import annotations  # Nicer syntax for Union types (PEP 604)

import logging
from typing import List, Optional, Tuple

from absl import flags

from erdos import Stream
from pylot import prediction

import pylot.operator_creator
from pylot.drivers.sensor_setup import DepthCameraSetup, IMUSetup, LidarSetup, RGBCameraSetup, \
    SegmentedCameraSetup, CameraSetup
from pylot.perception.camera_frame import CameraFrame
from pylot.perception.depth_frame import DepthFrame
from pylot.perception.detection import obstacle
from pylot.perception.detection.lane import Lane
from pylot.perception.messages import (ObstacleTrajectoriesMessageTuple,
                                       ObstaclesMessageTuple,
                                       SegmentedMessageTuple,
                                       SpeedSignsMessage, StopSignsMessage)
from pylot.perception.point_cloud import PointCloud
from pylot.perception.detection.traffic_light import TrafficLight

FLAGS = flags.FLAGS

logger = logging.getLogger(__name__)


def add_obstacle_detection(
    center_camera_stream: Stream[CameraFrame],
    center_camera_setup: Optional[CameraSetup] = None,
    pose_stream: Optional[Stream[pylot.utils.Pose]] = None,
    depth_stream: Optional[Stream[DepthFrame] | Stream[PointCloud]] = None,
    depth_camera_stream: Optional[Stream[DepthFrame]] = None,
    segmented_camera_stream: Optional[Stream[SegmentedMessageTuple]] = None,
    ground_obstacles_stream: Optional[Stream[ObstaclesMessageTuple]] = None,
    ground_speed_limit_signs_stream: Optional[
        Stream[SpeedSignsMessage]] = None,
    ground_stop_signs_stream: Optional[Stream[StopSignsMessage]] = None,
    time_to_decision_stream: Optional[Stream[float]] = None
) -> Tuple[Stream[ObstaclesMessageTuple],
           Optional[Stream[ObstaclesMessageTuple]]]:
    """Adds operators for obstacle detection to the data-flow.

    If the `--perfect_obstacle_detection` flag is set, the method adds a
    perfect detector operator, and returns a stream of perfect obstacles.
    Otherwise, if the `--obstacle_detection` flag is set, the method returns
    a stream of obstacles detected using a trained model.

    Args:
        center_camera_stream: Stream on which BGR frames are received.
        center_camera_setup: The setup of the center camera. This setup is used
            to calculate the real-world location of the obstacles.
        pose_stream: Stream on which pose info is received.
        depth_stream: Stream on which point cloud or depth frame messages are
            received.
        depth_camera_stream: Stream on which depth frames are received.
        segmented_camera__stream: Stream on which segmented frames are
            received.
        ground_obstacles_stream: Stream on which obstacles messages are
            received.
        ground_speed_limit_signs_stream: Stream on which speed signs messages
            are received.
        ground_stop_signs_stream: Stream on which stop signs messages are
            received.

    Returns:
        Stream of detected obstacles and, if the
            ``--perfect_obstacle_detection`` flag is set, a stream containing
            all obstacles in the camera frame.
    """
    obstacles_stream = None
    perfect_obstacles_stream = None
    if FLAGS.obstacle_detection:
        obstacles_stream_wo_depth = None
        if any('efficientdet' in model
               for model in FLAGS.obstacle_detection_model_names):
            logger.debug('Using EfficientDet obstacle detector...')
            obstacles_stream = pylot.operator_creator.\
                add_efficientdet_obstacle_detection(
                    center_camera_stream, time_to_decision_stream)
            obstacles_stream_wo_depth = obstacles_stream
        else:
            logger.debug('Using obstacle detector...')
            # TODO: Only returns the first obstacles stream.
            obstacles_streams = pylot.operator_creator.add_obstacle_detection(
                center_camera_stream, time_to_decision_stream)
            obstacles_stream_wo_depth = obstacles_streams[0]
        if FLAGS.planning_type == 'waypoint':
            logger.debug('Adding obstacle location finder...')
            # Adds an operator that finds the world locations of the obstacles.
            obstacles_stream = \
                pylot.operator_creator.add_obstacle_location_finder(
                    obstacles_stream_wo_depth, depth_stream, pose_stream,
                    center_camera_setup)
        else:
            obstacles_stream = obstacles_stream_wo_depth

    if FLAGS.perfect_obstacle_detection or FLAGS.evaluate_obstacle_detection:
        assert (pose_stream is not None and depth_camera_stream is not None
                and segmented_camera_stream is not None
                and ground_obstacles_stream is not None
                and ground_speed_limit_signs_stream is not None
                and ground_stop_signs_stream is not None)
        logger.debug('Using perfect obstacle detector...')
        perfect_obstacles_stream = pylot.operator_creator.add_perfect_detector(
            depth_camera_stream, center_camera_stream, segmented_camera_stream,
            pose_stream, ground_obstacles_stream,
            ground_speed_limit_signs_stream, ground_stop_signs_stream)
        if FLAGS.evaluate_obstacle_detection:
            logger.debug('Adding obstacle detection evaluation...')
            pylot.operator_creator.add_detection_evaluation(
                obstacles_stream_wo_depth,
                perfect_obstacles_stream,
                evaluate_timely=False,
                matching_policy='ceil',
                name='sync_detection_eval_operator')
            pylot.operator_creator.add_detection_evaluation(
                obstacles_stream_wo_depth,
                perfect_obstacles_stream,
                evaluate_timely=True,
                matching_policy='ceil',
                name='timely_detection_eval_operator')
        if FLAGS.perfect_obstacle_detection:
            logger.debug('Using perfect obstacle detector...')
            obstacles_stream = perfect_obstacles_stream

    if FLAGS.simulator_obstacle_detection:
        logger.debug('Using ground obstacles from the simulator...')
        obstacles_stream = ground_obstacles_stream

    return obstacles_stream, perfect_obstacles_stream


def add_traffic_light_detection(
    tl_transform: pylot.utils.Transform,
    vehicle_id_stream: Stream[int],
    release_sensor_stream: Stream[None],
    pose_stream: Optional[Stream[pylot.utils.Pose]] = None,
    depth_stream: Optional[Stream[DepthFrame] | Stream[PointCloud]] = None,
    ground_traffic_lights_stream: Optional[Stream[List[TrafficLight]]] = None,
    time_to_decision_stream: Optional[Stream[float]] = None
) -> Tuple[Stream[List[TrafficLight]], Optional[Stream[CameraFrame]]]:
    """Adds traffic light detection operators.

    The traffic light detectors use a camera with a narrow field of view.

    If the `--perfect_traffic_light_detection` flag is set, the method adds a
    perfect traffic light detector operator, and returns a stream of perfect
    traffic lights. Otherwise, if the `--traffic_light_detection` flag is
    set it returns a stream of traffic lights detected using a trained model.

    Args:
        tl_transform: Transform of the traffic light camera relative to the ego
            vehicle.
        vehicle_id_stream: A stream on which the simulator publishes simulator
            ego-vehicle id.
        release_sensor_stream: Sends a watermark to synchronize the sending of
            sensor data.
        pose_stream: A stream on which pose info is received.
        depth_stream: Stream on which point clouds or depth frames are
            received.

    Returns:
        A stream of traffic lights, and a stream of camera frames from the
            camera defined by :param tl_transform:, unless
            the ``--simulator_traffic_light_detection`` flag is set.
    """
    tl_camera_stream = None
    if FLAGS.traffic_light_detection or FLAGS.perfect_traffic_light_detection:
        logger.debug('Adding traffic light camera...')
        # Only add the TL camera if traffic light detection is enabled.
        tl_camera_setup = RGBCameraSetup('traffic_light_camera',
                                         FLAGS.camera_image_width,
                                         FLAGS.camera_image_height,
                                         tl_transform, 45)
        (tl_camera_stream,
         _) = pylot.operator_creator.add_camera_driver(tl_camera_setup,
                                                       vehicle_id_stream,
                                                       release_sensor_stream)

    traffic_lights_stream = None
    if FLAGS.traffic_light_detection:
        logger.debug('Using traffic light detection...')
        traffic_lights_stream = \
            pylot.operator_creator.add_traffic_light_detector(
                tl_camera_stream, time_to_decision_stream)
        # Adds operator that finds the world locations of the traffic lights.
        traffic_lights_stream = \
            pylot.operator_creator.add_obstacle_location_finder(
                traffic_lights_stream, depth_stream, pose_stream,
                tl_camera_setup)

    if FLAGS.perfect_traffic_light_detection:
        assert (pose_stream is not None
                and ground_traffic_lights_stream is not None)
        logger.debug('Using perfect traffic light detection...')
        # Add segmented and depth cameras with fov 45. These cameras are needed
        # by the perfect traffic light detector.
        tl_depth_camera_setup = DepthCameraSetup('traffic_light_depth_camera',
                                                 FLAGS.camera_image_width,
                                                 FLAGS.camera_image_height,
                                                 tl_transform, 45)
        (tl_depth_camera_stream, _) = \
            pylot.operator_creator.add_camera_driver(tl_depth_camera_setup,
                                                     vehicle_id_stream,
                                                     release_sensor_stream)

        segmented_tl_camera_setup = SegmentedCameraSetup(
            'traffic_light_segmented_camera', FLAGS.camera_image_width,
            FLAGS.camera_image_height, tl_transform, 45)
        (tl_segmented_camera_stream, _) = \
            pylot.operator_creator.add_camera_driver(
                segmented_tl_camera_setup, vehicle_id_stream,
                release_sensor_stream)

        traffic_lights_stream = \
            pylot.operator_creator.add_perfect_traffic_light_detector(
                ground_traffic_lights_stream, tl_camera_stream,
                tl_depth_camera_stream, tl_segmented_camera_stream,
                pose_stream)

    if FLAGS.simulator_traffic_light_detection:
        logger.debug('Using ground traffic lights from the simulator...')
        traffic_lights_stream = ground_traffic_lights_stream

    return traffic_lights_stream, tl_camera_stream


def add_depth(
    transform: pylot.utils.Transform, vehicle_id_stream: Stream[id],
    center_camera_setup: CameraSetup, depth_camera_stream: Stream[DepthFrame],
    release_sensor_stream: Stream[None]
) -> Tuple[Stream[DepthFrame], Stream[None], Stream[None]]:
    """Adds operators for depth estimation.

    The operator returns depth frames from the simulator if the
    `--perfect_depth_estimation` flag is set.

    Args:
        transform: Transform of the center camera relative to the ego vehicle.
        vehicle_id_stream: A stream on which the simulator publishes simulator
            ego-vehicle id.
        center_camera_setup: The setup of the center camera.
        depth_camera_stream: Stream on which depth frames are received.
        release_sensor_stream: Sends a watermark to synchronize the sending of
            sensor data.

    Returns:
        A stream of depth frames and two streams notifying when the added
            stereo cameras are ready to send data.
    """
    depth_stream = None
    if FLAGS.depth_estimation:
        logger.debug('Adding left and right cameras for depth estimation...')
        (left_camera_stream,
         right_camera_stream,
         notify_left_camera_stream,
         notify_right_camera_stream) = \
            pylot.operator_creator.add_left_right_cameras(
                transform, vehicle_id_stream, release_sensor_stream)
        logger.debug('Using camera depth estimation...')
        depth_stream = pylot.operator_creator.add_depth_estimation(
            left_camera_stream, right_camera_stream, center_camera_setup)
    if FLAGS.perfect_depth_estimation:
        logger.debug('Using perfect depth estimation...')
        depth_stream = depth_camera_stream
    return depth_stream, notify_left_camera_stream, notify_right_camera_stream


def add_lane_detection(
        center_camera_stream: Stream[CameraFrame],
        pose_stream: Optional[Stream[pylot.utils.Pose]] = None,
        open_drive_stream: Optional[Stream[str]] = None) -> Stream[List[Lane]]:
    """Adds operators for lane detection.

    If the `--perfect_lane_detection` flag is set, the method adds a perfect
    lane detection operator, and returns a stream of perfect lanes. Otherwise,
    if the `--lane_detection` flag is set the method returns a stream of lanes
    detected using a trained model.

    Args:
        center_camera_stream: Contains frames from the center camera.
        pose_stream: Contains pose information. Required for perfect lane
            detection.
        open_drive_stream: Stream on which open drive string representations
            are received. The operator can construct HDMaps out of the open
            drive strings. Required for perfect lane detection.

    Returns:
        Stream containing a list of lanes.
    """
    lane_detection_stream = None
    if FLAGS.lane_detection:
        if FLAGS.lane_detection_type == 'canny':
            logger.debug('Using Canny Edge lane detector...')
            lane_detection_stream = \
                pylot.operator_creator.add_canny_edge_lane_detection(
                    center_camera_stream)
        elif FLAGS.lane_detection_type == 'lanenet':
            logger.debug('Using Lanenet lane detector...')
            lane_detection_stream = \
                pylot.operator_creator.add_lanenet_detection(
                    center_camera_stream)
        else:
            raise ValueError('Unexpected lane detection type {}'.format(
                FLAGS.lane_detection_type))
    if FLAGS.perfect_lane_detection:
        assert pose_stream is not None, \
            'Cannot added perfect lane detection without a post stream'
        logger.debug('Using perfect lane detector...')
        lane_detection_stream = \
            pylot.operator_creator.add_perfect_lane_detector(
                pose_stream, open_drive_stream, center_camera_stream)
    return lane_detection_stream


def add_obstacle_tracking(
    center_camera_stream: Stream[CameraFrame],
    center_camera_setup: CameraSetup,
    obstacles_stream: Stream[ObstaclesMessageTuple],
    depth_stream: Optional[Stream[DepthFrame]
                           | Stream[PointCloud]] = None,
    vehicle_id_stream: Optional[Stream[int]] = None,
    pose_stream: Optional[Stream[pylot.utils.Pose]] = None,
    ground_obstacles_stream: Optional[Stream[ObstaclesMessageTuple]] = None,
    time_to_decision_stream: Optional[Stream[float]] = None
) -> Stream[ObstacleTrajectoriesMessageTuple]:
    """Adds operators for obstacle tracking.

    If the `--perfect_obstacle_tracking` flag is setup, the method adds an
    operator which uses information from the simulator to perfectly track
    obstacles. Otherwise, if the '--obstacle_tracking' flag is set, the method
    adds operators that use algorithms and trained models to track obstacles.

    Args:
        center_camera_stream: Stream on which camera frames are received.
        center_camera_setup: The setup of the center camera. This setup is used
            to calculate the real-world location of the obstacles.
        obstacles_stream: Stream on which detected obstacles are received.
        depth_stream: Stream on which point cloud or depth frame messages are
            received.
        vehicle_id_stream: A stream on which the simulator publishes simulator
            ego-vehicle id.
        pose_stream: A stream on which pose info is received.
        ground_obstacles_stream: Stream on which obstacles are received.

    Returns:
        A stream of obstacles annotated with their trajectories.
    """
    obstacles_tracking_stream = None
    if FLAGS.obstacle_tracking:
        if FLAGS.tracker_type == 'center_track':
            logger.debug('Using CenterTrack obstacle tracker...')
            obstacles_wo_history_tracking_stream = \
                pylot.operator_creator.add_center_track_tracking(
                    center_camera_stream, center_camera_setup)
        elif FLAGS.tracker_type == 'qd_track':
            logger.debug('Using QDTrack obstacle tracker...')
            obstacles_wo_history_tracking_stream = \
                pylot.operator_creator.add_qd_track_tracking(
                    center_camera_stream, center_camera_setup)
        else:
            logger.debug('Using obstacle tracker...')
            obstacles_wo_history_tracking_stream = \
                pylot.operator_creator.add_obstacle_tracking(
                    obstacles_stream,
                    center_camera_stream,
                    time_to_decision_stream)
        logger.debug('Adding operator to compute obstacle location history...')
        obstacles_tracking_stream = \
            pylot.operator_creator.add_obstacle_location_history(
                obstacles_wo_history_tracking_stream, depth_stream,
                pose_stream, center_camera_setup)
    if FLAGS.perfect_obstacle_tracking:
        assert (pose_stream is not None
                and ground_obstacles_stream is not None)
        logger.debug('Using perfect obstacle tracker...')
        obstacles_tracking_stream = \
            pylot.operator_creator.add_perfect_tracking(
                vehicle_id_stream, ground_obstacles_stream, pose_stream)

    if FLAGS.evaluate_obstacle_tracking:
        # The operator uses the ground stream which should
        # contain obstacles with 2D bounding boxes if the
        # stream is generated by a perfect detector.
        # Note: the tracker eval operator cannot compute accuracy
        # if the obstacles do not contain 2D bounding boxes.
        logger.debug('Adding obstacle tracking evaluation...')
        pylot.operator_creator.add_tracking_evaluation(
            obstacles_wo_history_tracking_stream,
            ground_obstacles_stream,
            evaluate_timely=False,
            matching_policy='ceil',
            name='sync_tracking_eval_operator')
        pylot.operator_creator.add_tracking_evaluation(
            obstacles_wo_history_tracking_stream,
            ground_obstacles_stream,
            evaluate_timely=True,
            matching_policy='ceil',
            name='timely_tracking_eval_operator')

    return obstacles_tracking_stream


def add_segmentation(center_camera_stream: Stream[CameraFrame],
                     ground_segmented_stream: Optional[
                         Stream[SegmentedMessageTuple]] = None):
    """Adds operators for pixel semantic segmentation.

    If the `--perfect_segmentation` flag is set, the method returns a stream
    of perfectly frames. Otherwise, if the `--segmentation` flag is set, the
    method adds operators that use trained models.

    Args:
        center_camera_stream: Stream on which camera frames are received.
        ground_segmented_stream: Stream on which perfectly segmented frames are
            received. Required, and returned, if the
            ``--perfect_segmentation`` flag is set.

    Returns:
        Stream on which semantically segmented frames are published.
    """
    segmented_stream = None
    if FLAGS.segmentation:
        logger.debug('Using semantic segmentation...')
        segmented_stream = pylot.operator_creator.add_segmentation(
            center_camera_stream)
        if FLAGS.evaluate_segmentation:
            assert ground_segmented_stream is not None, \
                "Cannot evaluate segmentation without ground truth"
            logger.debug('Adding semantic segmentation evaluation...')
            pylot.operator_creator.add_segmentation_evaluation(
                ground_segmented_stream, segmented_stream)
    if FLAGS.perfect_segmentation:
        logger.debug('Using perfect semantic segmentation...')
        assert ground_segmented_stream is not None
        return ground_segmented_stream
    return segmented_stream


def add_prediction(obstacles_tracking_stream: Stream[ObstaclesMessageTuple],
                   vehicle_id_stream: Stream[int],
                   release_sensor_stream: Stream[None],
                   point_cloud_stream: Stream[PointCloud],
                   lidar_setup: Stream[LidarSetup],
                   pose_stream: Optional[Stream[pylot.utils.Pose]] = None,
                   Camera_transform = None, 
                   time_to_decision_stream: Optional[Stream[float]] = None
                ) -> Tuple[Stream[prediction], Stream[int], Stream]:
    """Adds prediction operators.

    Args:
        obstacles_tracking_stream (:py:class:`erdos.ReadStream`):
            Stream on which
            :py:class:`~pylot.perception.messages.ObstacleTrajectoriesMessage`
            are received.
        vehicle_id_stream (:py:class:`erdos.ReadStream`): A stream on
             which the simulator publishes simulator ego-vehicle id.
        camera_transform (:py:class:`~pylot.utils.Transform`): Transform of the
             center camera relative to the ego vehicle.
        pose_stream (:py:class:`erdos.ReadStream`, optional): Stream on
             which pose info is received.
        point_cloud_stream (:py:class:`erdos.ReadStream`, optional): Stream on
            which point cloud messages are received.

    Returns:
        :py:class:`erdos.ReadStream`: Stream on which
        :py:class:`~pylot.prediction.messages.PredictionMessage` messages are
        published.
    """
    prediction_stream = None
    top_down_segmented_camera_stream = None
    notify_reading_stream = None
    if FLAGS.prediction:
        if FLAGS.prediction_type == 'linear':
            logger.debug('Using linear prediction...')
            prediction_stream = pylot.operator_creator.add_linear_prediction(
                obstacles_tracking_stream, time_to_decision_stream)
        elif FLAGS.prediction_type == 'r2p2':
            logger.debug('Using R2P2 prediction...')
            assert point_cloud_stream is not None
            assert lidar_setup is not None
            prediction_stream = pylot.operator_creator.add_r2p2_prediction(
                point_cloud_stream, obstacles_tracking_stream,
                time_to_decision_stream, lidar_setup)
        else:
            raise ValueError('Unexpected prediction_type {}'.format(
                FLAGS.prediction_type))
        if FLAGS.evaluate_prediction:
            assert pose_stream is not None
            logger.debug('Adding prediction evaluation...')
            pylot.operator_creator.add_prediction_evaluation(
                pose_stream, obstacles_tracking_stream, prediction_stream)
        if FLAGS.visualize_prediction:
            logger.debug('Adding for prediction evaluation...')
            # Add bird's eye camera.
            top_down_transform = pylot.utils.get_top_down_transform(
                camera_transform, FLAGS.top_down_camera_altitude)
            top_down_seg_camera_setup = SegmentedCameraSetup(
                'top_down_segmented_camera', FLAGS.camera_image_width,
                FLAGS.camera_image_height, top_down_transform, 90)
            (top_down_segmented_camera_stream,
             notify_reading_stream) = pylot.operator_creator.add_camera_driver(
                 top_down_seg_camera_setup, vehicle_id_stream,
                 release_sensor_stream)
    else:
        logger.debug('Not using prediction...')
    return (prediction_stream, top_down_segmented_camera_stream,
            notify_reading_stream)


def add_planning(goal_location, 
                 traffic_lights_stream: Stream[TrafficLight], 
                 lanes_stream: Stream[Lane], 
                 open_drive_stream: Stream,
                 global_trajectory_stream, 
                 prediction_stream: Stream[prediction],
                 time_to_decision_stream: Optional[Stream[float]] = None,
                 pose_stream: Optional[Stream[pylot.utils.Pose]] = None) -> Stream:
    """Adds planning operators.

    Args:
        goal_location (:py:class:`~pylot.utils.Location`): The destination.
        pose_stream (:py:class:`erdos.ReadStream`): Stream on which
            pose info is received.
        prediction_stream (:py:class:`erdos.ReadStream`): Stream of
            :py:class:`~pylot.prediction.messages.PredictionMessage` messages
            for predicted obstacles.
        traffic_lights_stream (:py:class:`erdos.ReadStream`): Stream of
            :py:class:`~pylot.perception.messages.TrafficLightsMessage`
            messages for traffic lights.
        open_drive_stream (:py:class:`erdos.ReadStream`, optional):
            Stream on which open drive string representations are received.
            Operators can construct HDMaps out of the open drive strings.
        global_trajectory_stream (:py:class:`erdos.ReadStream`, optional):
            Stream on which global trajectory is received.

    Returns:
        :py:class:`erdos.ReadStream`: Stream on which the waypoints are
        published.
    """
    logger.debug('Using behavior planning...')
    trajectory_stream = pylot.operator_creator.add_behavior_planning(
        pose_stream, open_drive_stream, global_trajectory_stream,
        goal_location)
    logger.debug('Using planning...')
    waypoints_stream = pylot.operator_creator.add_planning(
        pose_stream, prediction_stream, traffic_lights_stream, lanes_stream,
        trajectory_stream, open_drive_stream, time_to_decision_stream)
    return waypoints_stream


def add_control(waypoints_stream,
                ground_vehicle_id_stream: Stream[int],
                perfect_obstacles_stream: Stream[obstacle],
                pose_stream: Optional[Stream[pylot.utils.Pose]] = None) -> Tuple[Stream, Stream]:
    """Adds ego-vehicle control operators.

    Args:
        pose_stream (:py:class:`erdos.ReadStream`, optional): Stream on
            which pose info is received.
        waypoints_stream (:py:class:`erdos.ReadStream`): Stream on which
            waypoints are received.
        ground_vehicle_id_stream (:py:class:`erdos.ReadStream`): Stream on
            which the vehicle id is published.

    Returns:
        :py:class:`erdos.ReadStream`: Stream on which
        :py:class:`~pylot.control.messages.ControlMessage` messages are
        published.
    """
    if FLAGS.control == 'pid':
        logger.debug('Using PID controller...')
        control_stream = pylot.operator_creator.add_pid_control(
            pose_stream, waypoints_stream)
    elif FLAGS.control == 'mpc':
        logger.debug('Using MPC controller...')
        control_stream = pylot.operator_creator.add_mpc(
            pose_stream, waypoints_stream)
    elif FLAGS.control in ['simulator_auto_pilot', 'manual']:
        # TODO: Hack! We synchronize on a single stream, based on a
        # guesestimate of which stream is slowest.
        logger.debug('Using the manual control/autopilot...')
        stream_to_sync_on = waypoints_stream
        if (FLAGS.evaluate_obstacle_detection
                and not FLAGS.perfect_obstacle_detection):
            # Ensure that the perfect obstacle detector doesn't remain
            # behind.
            logger.debug('Synchronizing ticking using the perfect detector'
                         ' stream')
            stream_to_sync_on = perfect_obstacles_stream
        else:
            logger.debug('Synchronizing ticking using the waypoints stream')
        control_stream = pylot.operator_creator.add_synchronizer(
            ground_vehicle_id_stream, stream_to_sync_on)
    else:
        raise ValueError('Unexpected control {}'.format(FLAGS.control))

    if FLAGS.evaluate_control:
        logger.debug('Adding control evaluation operator...')
        pylot.operator_creator.add_control_evaluation(pose_stream,
                                                      waypoints_stream)
    return control_stream


def add_evaluation(vehicle_id_stream: Stream[int], pose_stream: Optional[Stream[pylot.utils.Pose]] = None, imu_stream: Optional[Stream[IMUSetup]] = None) -> Stream:
    if FLAGS.evaluation:
        logger.debug('Adding collision logging sensor...')
        collision_stream = pylot.operator_creator.add_collision_sensor(
            vehicle_id_stream)

        logger.debug('Adding lane invasion sensor...')
        lane_invasion_stream = pylot.operator_creator.add_lane_invasion_sensor(
            vehicle_id_stream)

        logger.debug('Adding traffic light invasion sensor...')
        traffic_light_invasion_stream = \
            pylot.operator_creator.add_traffic_light_invasion_sensor(
                vehicle_id_stream, pose_stream)

        logger.debug('Adding overall evaluation operator...')
        pylot.operator_creator.add_eval_metric_logging(
            collision_stream, lane_invasion_stream,
            traffic_light_invasion_stream, imu_stream, pose_stream)
