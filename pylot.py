from absl import app, flags

import erdos

import pylot.flags
import pylot.component_creator
import pylot.operator_creator
import pylot.utils
from pylot.simulation.utils import get_world, set_asynchronous_mode

FLAGS = flags.FLAGS

flags.DEFINE_list('goal_location', '234, 59, 39', 'Ego-vehicle goal location')

# The location of the center camera relative to the ego-vehicle.
CENTER_CAMERA_LOCATION = pylot.utils.Location(1.5, 0.0, 1.4)


def add_evaluation_operators(vehicle_id_stream, pose_stream, imu_stream,
                             pose_stream_for_control,
                             waypoints_stream_for_control):
    if FLAGS.evaluation:
        # Add the collision sensor.
        collision_stream = pylot.operator_creator.add_collision_sensor(
            vehicle_id_stream)

        # Add the lane invasion sensor.
        lane_invasion_stream = pylot.operator_creator.add_lane_invasion_sensor(
            vehicle_id_stream)

        # Add the traffic light invasion sensor.
        traffic_light_invasion_stream = \
            pylot.operator_creator.add_traffic_light_invasion_sensor(
                vehicle_id_stream, pose_stream)

        # Add the evaluation logger.
        pylot.operator_creator.add_eval_metric_logging(
            collision_stream, lane_invasion_stream,
            traffic_light_invasion_stream, imu_stream, pose_stream)

        # Add control evaluation logging operator.
        pylot.operator_creator.add_control_evaluation(
            pose_stream_for_control, waypoints_stream_for_control)


def driver():
    transform = pylot.utils.Transform(CENTER_CAMERA_LOCATION,
                                      pylot.utils.Rotation())

    control_loop_stream = erdos.LoopStream()
    time_to_decision_loop_stream = erdos.LoopStream()
    if FLAGS.carla_mode == 'pseudo-asynchronous':
        release_sensor_stream = erdos.LoopStream()
        pipeline_finish_notify_stream = erdos.LoopStream()
    else:
        release_sensor_stream = erdos.IngestStream()
        pipeline_finish_notify_stream = erdos.IngestStream()
    notify_streams = []

    # Create operator that bridges between pipeline and CARLA.
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
    ) = pylot.operator_creator.add_carla_bridge(
        control_loop_stream,
        release_sensor_stream,
        pipeline_finish_notify_stream,
    )

    # Add sensors.
    (center_camera_stream, notify_rgb_stream,
     center_camera_setup) = pylot.operator_creator.add_rgb_camera(
         transform, vehicle_id_stream, release_sensor_stream)
    notify_streams.append(notify_rgb_stream)
    if pylot.flags.must_add_depth_camera_sensor():
        (depth_camera_stream, notify_depth_stream,
         depth_camera_setup) = pylot.operator_creator.add_depth_camera(
             transform, vehicle_id_stream, release_sensor_stream)
    else:
        depth_camera_stream = None
    if pylot.flags.must_add_segmented_camera_sensor():
        (ground_segmented_stream, notify_segmented_stream,
         _) = pylot.operator_creator.add_segmented_camera(
             transform, vehicle_id_stream, release_sensor_stream)
    else:
        ground_segmented_stream = None

    if pylot.flags.must_add_lidar_sensor():
        # Place LiDAR sensor in the same location as the center camera.
        (point_cloud_stream, notify_lidar_stream,
         lidar_setup) = pylot.operator_creator.add_lidar(
             transform, vehicle_id_stream, release_sensor_stream)
    else:
        point_cloud_stream = None
        lidar_setup = None

    if FLAGS.obstacle_location_finder_sensor == 'lidar':
        depth_stream = point_cloud_stream
        # Camera sensors are slower than the lidar sensor.
        notify_streams.append(notify_lidar_stream)
    elif FLAGS.obstacle_location_finder_sensor == 'depth_camera':
        depth_stream = depth_camera_stream
        notify_streams.append(notify_depth_stream)
    else:
        raise ValueError(
            'Unknown --obstacle_location_finder_sensor value {}'.format(
                FLAGS.obstacle_location_finder_sensor))

    imu_stream = None
    if pylot.flags.must_add_imu_sensor():
        (imu_stream,
         _) = pylot.operator_creator.add_imu(transform, vehicle_id_stream)

    obstacles_stream = pylot.component_creator.add_obstacle_detection(
        center_camera_stream, center_camera_setup, pose_stream, depth_stream,
        depth_camera_stream, ground_segmented_stream, ground_obstacles_stream,
        ground_speed_limit_signs_stream, ground_stop_signs_stream,
        time_to_decision_loop_stream)
    traffic_lights_stream, tl_camera_stream = \
        pylot.component_creator.add_traffic_light_detection(
            transform, vehicle_id_stream, release_sensor_stream, pose_stream,
            depth_stream, ground_traffic_lights_stream)

    lane_detection_stream = pylot.component_creator.add_lane_detection(
        center_camera_stream, pose_stream)

    obstacles_tracking_stream = pylot.component_creator.add_obstacle_tracking(
        center_camera_stream, center_camera_setup, obstacles_stream,
        depth_stream, vehicle_id_stream, pose_stream, ground_obstacles_stream,
        time_to_decision_loop_stream)

    segmented_stream = pylot.component_creator.add_segmentation(
        center_camera_stream, ground_segmented_stream)

    depth_stream = pylot.component_creator.add_depth(transform,
                                                     vehicle_id_stream,
                                                     center_camera_setup,
                                                     depth_camera_stream)

    if FLAGS.fusion:
        pylot.operator_creator.add_fusion(pose_stream, obstacles_stream,
                                          depth_stream,
                                          ground_obstacles_stream)

    prediction_stream, prediction_camera_stream, notify_prediction_stream = \
        pylot.component_creator.add_prediction(
            obstacles_tracking_stream, vehicle_id_stream, transform,
            release_sensor_stream, pose_stream, point_cloud_stream,
            lidar_setup)
    if notify_prediction_stream:
        notify_streams.append(notify_prediction_stream)

    goal_location = pylot.utils.Location(float(FLAGS.goal_location[0]),
                                         float(FLAGS.goal_location[1]),
                                         float(FLAGS.goal_location[2]))
    waypoints_stream = pylot.component_creator.add_planning(
        goal_location, pose_stream, prediction_stream, center_camera_stream,
        obstacles_stream, traffic_lights_stream, open_drive_stream,
        global_trajectory_stream, time_to_decision_loop_stream)

    if FLAGS.carla_mode == "pseudo-asynchronous":
        # Add a synchronizer in the pseudo-asynchronous mode.
        (
            waypoints_stream_for_control,
            pose_stream_for_control,
            sensor_ready_stream,
            _pipeline_finish_notify_stream,
        ) = pylot.operator_creator.add_planning_pose_synchronizer(
            waypoints_stream, pose_stream_for_control, pose_stream,
            *notify_streams)
        release_sensor_stream.set(sensor_ready_stream)
        pipeline_finish_notify_stream.set(_pipeline_finish_notify_stream)
    else:
        waypoints_stream_for_control = waypoints_stream
        pose_stream_for_control = pose_stream

    control_stream = pylot.component_creator.add_control(
        pose_stream_for_control, waypoints_stream_for_control,
        vehicle_id_stream)
    control_loop_stream.set(control_stream)

    add_evaluation_operators(vehicle_id_stream, pose_stream, imu_stream,
                             pose_stream_for_control,
                             waypoints_stream_for_control)

    time_to_decision_stream = pylot.operator_creator.add_time_to_decision(
        pose_stream, obstacles_stream)
    time_to_decision_loop_stream.set(time_to_decision_stream)

    control_display_stream = None
    streams_to_send_top_on = []
    if pylot.flags.must_visualize():
        control_display_stream, ingest_streams = \
            pylot.operator_creator.add_visualizer(
                pose_stream, center_camera_stream, tl_camera_stream,
                prediction_camera_stream, depth_camera_stream,
                point_cloud_stream, segmented_stream, imu_stream,
                obstacles_stream, traffic_lights_stream,
                obstacles_tracking_stream, lane_detection_stream,
                prediction_stream, waypoints_stream, control_stream)
        streams_to_send_top_on += ingest_streams

    erdos.run_async()

    for stream in streams_to_send_top_on:
        stream.send(erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))
    # If we did not use the pseudo-asynchronous mode, ask the sensors to
    # release their readings whenever.
    if FLAGS.carla_mode != "pseudo-asynchronous":
        release_sensor_stream.send(
            erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))
        pipeline_finish_notify_stream.send(
            erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

    if pylot.flags.must_visualize():
        pylot.utils.run_visualizer_control_loop(control_display_stream)


def main(args):
    # Connect an instance to the simulator to make sure that we can turn the
    # synchronous mode off after the script finishes running.
    client, world = get_world(FLAGS.carla_host, FLAGS.carla_port,
                              FLAGS.carla_timeout)
    try:
        if FLAGS.simulation_recording_file is not None:
            client.start_recorder(FLAGS.simulation_recording_file)
        driver()
    except KeyboardInterrupt:
        if FLAGS.simulation_recording_file is not None:
            client.stop_recorder()
        set_asynchronous_mode(world)
    except Exception:
        if FLAGS.simulation_recording_file is not None:
            client.stop_recorder()
        set_asynchronous_mode(world)
        raise


if __name__ == '__main__':
    app.run(main)
