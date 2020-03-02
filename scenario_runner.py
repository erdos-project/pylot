from absl import app
from absl import flags

import erdos
import carla
import pylot.flags
import pylot.component_creator
import pylot.operator_creator
from pylot.simulation.carla_operator import CarlaOperator
from pylot.simulation.perfect_detector_operator import PerfectDetectorOperator
from pylot.simulation.scenario.person_avoidance_agent_operator \
    import PersonAvoidanceAgentOperator
from pylot.simulation.utils import get_world, set_asynchronous_mode

FLAGS = flags.FLAGS
flags.DEFINE_list('goal_location', '17.73, 327.07, 0.5',
                  'Ego-vehicle goal location')
flags.DEFINE_bool(
    'avoidance_agent', False,
    'True to enable scenario avoidance agent planner and controller')

# The location of the center camera relative to the ego-vehicle.
CENTER_CAMERA_LOCATION = pylot.utils.Location(1.5, 0.0, 1.4)


def add_avoidance_agent(can_bus_stream, obstacles_stream,
                        ground_obstacles_stream, goal_location):
    op_config = erdos.OperatorConfig(
        name=FLAGS.obstacle_detection_model_names[0] + '_agent',
        flow_watermarks=False,
        log_file_name=FLAGS.log_file_name,
        csv_log_file_name=FLAGS.csv_log_file_name,
        profile_file_name=FLAGS.profile_file_name)
    [control_stream] = erdos.connect(
        PersonAvoidanceAgentOperator, op_config,
        [can_bus_stream, obstacles_stream, ground_obstacles_stream],
        goal_location, FLAGS)
    return control_stream


def driver():
    transform = pylot.utils.Transform(CENTER_CAMERA_LOCATION,
                                      pylot.utils.Rotation())
    control_loop_stream = erdos.LoopStream()
    # Create carla operator.
    (can_bus_stream, ground_traffic_lights_stream, ground_obstacles_stream,
     ground_speed_limit_signs_stream, ground_stop_signs_stream,
     vehicle_id_stream, open_drive_stream, global_trajectory_stream
     ) = pylot.operator_creator.add_carla_bridge(control_loop_stream)

    # Add sensors.
    (center_camera_stream,
     center_camera_setup) = pylot.operator_creator.add_rgb_camera(
         transform, vehicle_id_stream)
    if pylot.flags.must_add_depth_camera_sensor():
        (depth_camera_stream,
         depth_camera_setup) = pylot.operator_creator.add_depth_camera(
             transform, vehicle_id_stream)
    else:
        depth_camera_stream = None
    if pylot.flags.must_add_segmented_camera_sensor():
        (ground_segmented_stream,
         _) = pylot.operator_creator.add_segmented_camera(
             transform, vehicle_id_stream)
    else:
        ground_segmented_stream = None
    # Place Lidar sensor in the same location as the center camera.
    (point_cloud_stream,
     lidar_setup) = pylot.operator_creator.add_lidar(transform,
                                                     vehicle_id_stream)
    imu_stream = None
    if FLAGS.imu:
        (imu_stream,
         _) = pylot.operator_creator.add_imu(transform, vehicle_id_stream)

    obstacles_stream = pylot.component_creator.add_obstacle_detection(
        center_camera_stream, center_camera_setup, can_bus_stream,
        point_cloud_stream, depth_camera_stream, ground_segmented_stream,
        ground_obstacles_stream, ground_speed_limit_signs_stream,
        ground_stop_signs_stream)

    traffic_lights_stream = \
        pylot.component_creator.add_traffic_light_detection(
            transform, vehicle_id_stream, can_bus_stream, point_cloud_stream,
            ground_traffic_lights_stream)

    obstacles_tracking_stream = pylot.component_creator.add_obstacle_tracking(
        center_camera_stream, obstacles_stream, can_bus_stream,
        ground_obstacles_stream)

    prediction_stream = pylot.component_creator.add_prediction(
        obstacles_tracking_stream, vehicle_id_stream, transform,
        can_bus_stream)

    goal_location = pylot.utils.Location(float(FLAGS.goal_location[0]),
                                         float(FLAGS.goal_location[1]),
                                         float(FLAGS.goal_location[2]))
    waypoints_stream = pylot.component_creator.add_planning(
        goal_location, can_bus_stream, prediction_stream, center_camera_stream,
        obstacles_stream, traffic_lights_stream, open_drive_stream,
        global_trajectory_stream)

    if FLAGS.avoidance_agent:
        control_stream = add_avoidance_agent(can_bus_stream, obstacles_stream,
                                             ground_obstacles_stream,
                                             goal_location)
    else:
        assert FLAGS.control_agent != 'carla_auto_pilot', \
            'Cannot use CARLA auto pilot in a scenario'
        # Add the behaviour planning and control operator.
        control_stream = pylot.component_creator.add_control(
            can_bus_stream, waypoints_stream)

    control_loop_stream.set(control_stream)

    pylot.operator_creator.add_sensor_visualizers(center_camera_stream,
                                                  depth_camera_stream,
                                                  point_cloud_stream,
                                                  ground_segmented_stream,
                                                  imu_stream, can_bus_stream)
    #erdos.run()


def main(argv):
    # Connect an instance to the simulator to make sure that we can turn the
    # synchronous mode off after the script finishes running.
    client, world = get_world(FLAGS.carla_host, FLAGS.carla_port,
                              FLAGS.carla_timeout)
    if client is None or world is None:
        raise ValueError("There was an issue connecting to the simulator.")

    try:
        driver()
        erdos.run()
    except KeyboardInterrupt:
        set_asynchronous_mode(world)
    except Exception:
        set_asynchronous_mode(world)
        raise


if __name__ == "__main__":
    app.run(main)
