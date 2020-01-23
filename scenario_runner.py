from absl import app
from absl import flags
import erdos
import os

import pylot.flags
import pylot.operator_creator
from pylot.simulation.utils import get_world, set_asynchronous_mode
from pylot.simulation.scenario.perfect_planning_operator \
    import PerfectPlanningOperator

FLAGS = flags.FLAGS
flags.DEFINE_enum('plan', 'stop', ['stop', 'swerve'],
                  'Planning Behavior (stop/swerve)')
flags.DEFINE_list('goal_location', '17.73, 327.07, 0.5',
                  'Ego-vehicle goal location')

# The location of the center camera relative to the ego-vehicle.
CENTER_CAMERA_LOCATION = pylot.utils.Location(1.5, 0.0, 1.4)


def add_planning(can_bus_stream, obstacles_stream, ground_obstacles_stream,
                 goal):
    csv_file_name = FLAGS.csv_log_file_name
    if not FLAGS.perfect_obstacle_detection:
        csv_file_name = 'results/{model}/{model}_{speed}_distance.csv'.format(
            model=FLAGS.obstacle_detection_model_names[0],
            speed=FLAGS.target_speed)
    [control_stream] = erdos.connect(
        PerfectPlanningOperator,
        [can_bus_stream, obstacles_stream, ground_obstacles_stream],
        True,
        'perfect_planning_operator',
        goal,
        FLAGS.plan,
        FLAGS,
        log_file_name=FLAGS.log_file_name,
        csv_file_name=csv_file_name)
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
    (depth_camera_stream,
     depth_camera_setup) = pylot.operator_creator.add_depth_camera(
         transform, vehicle_id_stream)
    (segmented_camera_stream,
     _) = pylot.operator_creator.add_segmented_camera(transform,
                                                      vehicle_id_stream)

    if FLAGS.perfect_obstacle_detection:
        assert (can_bus_stream is not None and depth_camera_stream is not None
                and segmented_camera_stream is not None
                and ground_obstacles_stream is not None
                and ground_speed_limit_signs_stream is not None
                and ground_stop_signs_stream is not None)
        obstacles_stream = pylot.operator_creator.add_perfect_detector(
            depth_camera_stream, center_camera_stream, segmented_camera_stream,
            can_bus_stream, ground_obstacles_stream,
            ground_speed_limit_signs_stream, ground_stop_signs_stream)
    else:
        csv_file_name = 'results/{model}/{model}_{speed}_runtimes.csv'.format(
            model=FLAGS.obstacle_detection_model_names[0],
            speed=FLAGS.target_speed)
        obstacles_stream = pylot.operator_creator.add_obstacle_detection(
            center_camera_stream, csv_file_name)[0]

    pylot.operator_creator.add_camera_visualizer(center_camera_stream,
                                                 'rgb_camera')

    goal_location = pylot.utils.Location(float(FLAGS.goal_location[0]),
                                         float(FLAGS.goal_location[1]),
                                         float(FLAGS.goal_location[2]))
    control_stream = add_planning(can_bus_stream, obstacles_stream,
                                  ground_obstacles_stream, goal_location)
    control_loop_stream.set(control_stream)


def main(args):
    # Connect an instance to the simulator to make sure that we can turn the
    # synchronous mode off after the script finishes running.
    client, world = get_world(FLAGS.carla_host, FLAGS.carla_port,
                              FLAGS.carla_timeout)
    if client is None or world is None:
        raise ValueError("There was an issue connecting to the simulator.")

    if not FLAGS.perfect_obstacle_detection:
        if not os.path.exists('./results'):
            os.mkdir('results')

        if not os.path.exists('./results/{}'.format(
                FLAGS.obstacle_detection_model_names[0])):
            os.mkdir('results/{}'.format(
                FLAGS.obstacle_detection_model_names[0]))

    try:
        erdos.run(driver)
    except KeyboardInterrupt:
        set_asynchronous_mode(world)
    except Exception:
        set_asynchronous_mode(world)
        raise


if __name__ == "__main__":
    app.run(main)
