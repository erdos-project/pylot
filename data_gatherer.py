import signal

from absl import app, flags

import erdos

import pylot.flags
import pylot.operator_creator
import pylot.simulation.utils
import pylot.utils
from pylot.drivers.sensor_setup import DepthCameraSetup, RGBCameraSetup, \
    SegmentedCameraSetup

FLAGS = flags.FLAGS

# Flags that control what data is recorded.
flags.DEFINE_bool('log_rgb_camera', False,
                  'True to enable center camera RGB logging')
flags.DEFINE_bool('log_segmented_camera', False,
                  'True to enable center segmented camera logging')
flags.DEFINE_bool('log_left_right_cameras', False,
                  'Control whether we log left and right cameras.')
flags.DEFINE_bool('log_depth_camera', False,
                  'True to enable depth camera logging')
flags.DEFINE_bool('log_gnss', False, 'Enable logging of GNSS measurements.')
flags.DEFINE_bool('log_pose', False, 'Enable logging of pose measurements.')
flags.DEFINE_bool('log_imu', False, 'Enable logging of IMU measurements.')
flags.DEFINE_bool('log_lidar', False, 'True to enable lidar logging')
flags.DEFINE_bool('log_obstacles', False,
                  'True to enable obstacle bounding box logging')
flags.DEFINE_bool(
    'log_traffic_lights', False,
    'True to enable traffic lights bounding box and camera logging')
flags.DEFINE_bool('log_multiple_object_tracker', False,
                  'True to enable logging in the MOT format')
flags.DEFINE_bool('log_trajectories', False,
                  'True to enable trajectory logging')
flags.DEFINE_bool('log_chauffeur', False,
                  'True to log data in ChauffeurNet style.')
flags.DEFINE_bool('log_top_down_segmentation', False,
                  'True to enable logging of top down segmentation')

# The location of the center camera relative to the ego-vehicle.
CENTER_CAMERA_LOCATION = pylot.utils.Location(1.0, 0.0, 1.8)


def main(argv):
    transform = pylot.utils.Transform(CENTER_CAMERA_LOCATION,
                                      pylot.utils.Rotation())

    control_loop_stream = erdos.LoopStream()
    release_sensor_stream = erdos.IngestStream()
    pipeline_finish_notify_stream = erdos.IngestStream()
    # Create an operator that connects to the simulator.
    (
        pose_stream,
        pose_stream_for_control,
        ground_traffic_lights_stream,
        ground_obstacles_stream,
        ground_speed_limit_signs_stream,
        ground_stop_signs_stream,
        vehicle_id_stream,
        open_drive_stream,
        global_trajectory_stream,
    ) = pylot.operator_creator.add_simulator_bridge(
        control_loop_stream, release_sensor_stream,
        pipeline_finish_notify_stream)

    # Add sensors.
    rgb_camera_setup = RGBCameraSetup('center_camera',
                                      FLAGS.camera_image_width,
                                      FLAGS.camera_image_height, transform,
                                      FLAGS.camera_fov)
    (center_camera_stream, notify_rgb_stream) = \
        pylot.operator_creator.add_camera_driver(
            rgb_camera_setup, vehicle_id_stream, release_sensor_stream)
    depth_camera_setup = DepthCameraSetup('depth_center_camera',
                                          FLAGS.camera_image_width,
                                          FLAGS.camera_image_height, transform,
                                          FLAGS.camera_fov)
    (depth_camera_stream,
     _) = pylot.operator_creator.add_camera_driver(depth_camera_setup,
                                                   vehicle_id_stream,
                                                   release_sensor_stream)
    seg_camera_setup = SegmentedCameraSetup('seg_center_camera',
                                            FLAGS.camera_image_width,
                                            FLAGS.camera_image_height,
                                            transform, FLAGS.camera_fov)
    (segmented_stream,
     _) = pylot.operator_creator.add_camera_driver(seg_camera_setup,
                                                   vehicle_id_stream,
                                                   release_sensor_stream)

    if FLAGS.log_rgb_camera:
        pylot.operator_creator.add_camera_logging(
            center_camera_stream, 'center_camera_logger_operator', 'center')

    if FLAGS.log_segmented_camera:
        pylot.operator_creator.add_camera_logging(
            segmented_stream, 'center_segmented_camera_logger_operator',
            'segmented')

    if FLAGS.log_depth_camera:
        pylot.operator_creator.add_camera_logging(
            depth_camera_stream, 'depth_camera_logger_operator', 'depth')

    imu_stream = None
    if FLAGS.log_imu:
        (imu_stream, _) = pylot.operator_creator.add_imu(
            pylot.utils.Transform(location=pylot.utils.Location(),
                                  rotation=pylot.utils.Rotation()),
            vehicle_id_stream)
        pylot.operator_creator.add_imu_logging(imu_stream)

    gnss_stream = None
    if FLAGS.log_gnss:
        (gnss_stream, _) = pylot.operator_creator.add_gnss(
            pylot.utils.Transform(location=pylot.utils.Location(),
                                  rotation=pylot.utils.Rotation()),
            vehicle_id_stream)
        pylot.operator_creator.add_gnss_logging(gnss_stream)

    if FLAGS.log_pose:
        pylot.operator_creator.add_pose_logging(pose_stream)

    traffic_lights_stream = None
    traffic_light_camera_stream = None
    if FLAGS.log_traffic_lights:
        tl_camera_setup = RGBCameraSetup('traffic_light_camera',
                                         FLAGS.camera_image_width,
                                         FLAGS.camera_image_height, transform,
                                         45)
        (traffic_light_camera_stream, _) = \
            pylot.operator_creator.add_camera_driver(
                tl_camera_setup, vehicle_id_stream, release_sensor_stream)
        pylot.operator_creator.add_camera_logging(
            traffic_light_camera_stream,
            'traffic_light_camera_logger_operator', 'traffic-light')

        tl_seg_camera_setup = SegmentedCameraSetup(
            'traffic_light_segmented_camera', FLAGS.camera_image_width,
            FLAGS.camera_image_height, transform, 45)
        (traffic_light_segmented_camera_stream, _) = \
            pylot.operator_creator.add_camera_driver(
                tl_seg_camera_setup,
                vehicle_id_stream,
                release_sensor_stream)

        tl_depth_camera_setup = DepthCameraSetup('traffic_light_depth_camera',
                                                 FLAGS.camera_image_width,
                                                 FLAGS.camera_image_height,
                                                 transform, 45)
        (traffic_light_depth_camera_stream, _) = \
            pylot.operator_creator.add_camera_driver(
                tl_depth_camera_setup, vehicle_id_stream,
                release_sensor_stream)

        traffic_lights_stream = \
            pylot.operator_creator.add_perfect_traffic_light_detector(
                ground_traffic_lights_stream,
                traffic_light_camera_stream,
                traffic_light_depth_camera_stream,
                traffic_light_segmented_camera_stream,
                pose_stream)
        pylot.operator_creator.add_bounding_box_logging(
            traffic_lights_stream, 'tl-bboxes')

    if FLAGS.log_left_right_cameras:
        (left_camera_stream, right_camera_stream, _,
         _) = pylot.operator_creator.add_left_right_cameras(
             transform, vehicle_id_stream, release_sensor_stream)
        pylot.operator_creator.add_camera_logging(
            left_camera_stream, 'left_camera_logger_operator', 'left')
        pylot.operator_creator.add_camera_logging(
            right_camera_stream, 'right_camera_logger_operator', 'right')

    point_cloud_stream = None
    if FLAGS.log_lidar:
        (point_cloud_stream, _,
         _) = pylot.operator_creator.add_lidar(transform, vehicle_id_stream,
                                               release_sensor_stream)
        pylot.operator_creator.add_lidar_logging(point_cloud_stream)

    obstacles_stream = None
    if FLAGS.log_obstacles:
        obstacles_stream = pylot.operator_creator.add_perfect_detector(
            depth_camera_stream, center_camera_stream, segmented_stream,
            pose_stream, ground_obstacles_stream,
            ground_speed_limit_signs_stream, ground_stop_signs_stream)
        pylot.operator_creator.add_bounding_box_logging(
            obstacles_stream, 'bboxes')

    if FLAGS.log_multiple_object_tracker:
        pylot.operator_creator.add_multiple_object_tracker_logging(
            obstacles_stream)

    obstacles_tracking_stream = None
    if FLAGS.log_trajectories or FLAGS.log_chauffeur:
        obstacles_tracking_stream = \
            pylot.operator_creator.add_perfect_tracking(
                vehicle_id_stream,
                ground_obstacles_stream,
                pose_stream)
        if FLAGS.log_trajectories:
            pylot.operator_creator.add_trajectory_logging(
                obstacles_tracking_stream)

    top_down_segmented_stream = None
    top_down_camera_setup = None
    if FLAGS.log_chauffeur or FLAGS.log_top_down_segmentation:
        top_down_transform = pylot.utils.get_top_down_transform(
            transform, FLAGS.top_down_camera_altitude)
        top_down_seg_cs = SegmentedCameraSetup('top_down_segmented_camera',
                                               FLAGS.camera_image_width,
                                               FLAGS.camera_image_height,
                                               top_down_transform, 90)
        (top_down_segmented_stream, _) = \
            pylot.operator_creator.add_camera_driver(
                top_down_seg_cs,
                vehicle_id_stream,
                release_sensor_stream)

        if FLAGS.log_top_down_segmentation:
            pylot.operator_creator.add_camera_logging(
                top_down_segmented_stream,
                'top_down_segmented_logger_operator', 'top-down-segmented')

        if FLAGS.log_chauffeur:
            top_down_camera_setup = RGBCameraSetup('top_down_rgb_camera',
                                                   FLAGS.camera_image_width,
                                                   FLAGS.camera_image_height,
                                                   top_down_transform, 90)
            (top_down_camera_stream,
             _) = pylot.operator_creator.add_camera_driver(
                 top_down_camera_setup, vehicle_id_stream,
                 release_sensor_stream)
            pylot.operator_creator.add_chauffeur_logging(
                vehicle_id_stream, pose_stream, obstacles_tracking_stream,
                top_down_camera_stream, top_down_segmented_stream,
                top_down_camera_setup)

    perfect_lane_stream = None
    if FLAGS.log_lane_detection_camera:
        perfect_lane_stream = pylot.operator_creator.add_perfect_lane_detector(
            pose_stream, open_drive_stream, center_camera_stream)

    # TODO: Hack! We synchronize on a single stream, based on a guesestimate
    # of which stream is slowest. Instead, We should synchronize on all output
    # streams, and we should ensure that even the operators without output
    # streams complete.
    if FLAGS.control == 'simulator_auto_pilot':
        # We insert a synchronizing operator that sends back a command when
        # the low watermark progresses on all input stream.
        stream_to_sync_on = center_camera_stream
        if obstacles_tracking_stream is not None:
            stream_to_sync_on = obstacles_tracking_stream
        if traffic_lights_stream is not None:
            stream_to_sync_on = traffic_lights_stream
        if perfect_lane_stream is not None:
            stream_to_sync_on = perfect_lane_stream
        if obstacles_stream is not None:
            stream_to_sync_on = obstacles_stream
        control_stream = pylot.operator_creator.add_synchronizer(
            vehicle_id_stream, stream_to_sync_on)
        control_loop_stream.set(control_stream)
    else:
        raise ValueError(
            "Must be in auto pilot mode. Pass --control=simulator_auto_pilot")

    control_display_stream = None
    streams_to_send_top_on = []
    if pylot.flags.must_visualize():
        control_display_stream, ingest_streams = \
            pylot.operator_creator.add_visualizer(
                pose_stream=pose_stream,
                camera_stream=center_camera_stream,
                tl_camera_stream=traffic_light_camera_stream,
                depth_stream=depth_camera_stream,
                point_cloud_stream=point_cloud_stream,
                segmentation_stream=segmented_stream,
                imu_stream=imu_stream,
                obstacles_stream=obstacles_stream,
                traffic_lights_stream=traffic_lights_stream,
                tracked_obstacles_stream=obstacles_tracking_stream)
        streams_to_send_top_on += ingest_streams

    # Connect an instance to the simulator to make sure that we can turn the
    # synchronous mode off after the script finishes running.
    client, world = pylot.simulation.utils.get_world(FLAGS.simulator_host,
                                                     FLAGS.simulator_port,
                                                     FLAGS.simulator_timeout)

    # Run the data-flow.
    node_handle = erdos.run_async()

    signal.signal(signal.SIGINT, shutdown)

    # Ask all sensors to release their data.
    release_sensor_stream.send(
        erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))
    for stream in streams_to_send_top_on:
        stream.send(erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

    try:
        if pylot.flags.must_visualize():
            pylot.utils.run_visualizer_control_loop(control_display_stream)
        node_handle.wait()
    except KeyboardInterrupt:
        node_handle.shutdown()
        pylot.simulation.utils.set_asynchronous_mode(world)
        if pylot.flags.must_visualize():
            import pygame
            pygame.quit()


def shutdown(sig, frame):
    raise KeyboardInterrupt


if __name__ == '__main__':
    app.run(main)
