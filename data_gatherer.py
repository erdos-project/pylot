from absl import app
from absl import flags

import pylot.config
import pylot.simulation.utils
import pylot.operator_creator

import erdos.graph

FLAGS = flags.FLAGS
CENTER_CAMERA_NAME = 'front_rgb_camera'
DEPTH_CAMERA_NAME = 'front_depth_camera'
SEGMENTED_CAMERA_NAME = 'front_semantic_camera'
LEFT_CAMERA_NAME = 'left_rgb_camera'
RIGHT_CAMERA_NAME = 'right_rgb_camera'

# Flags that control what data is recorded.
flags.DEFINE_string('data_path', 'data/',
                    'Path where to store Carla camera images')
flags.DEFINE_integer('log_every_nth_frame', 1,
                     'Control how often the script logs frames')

# Flags for logging data from left and right cameras.
flags.DEFINE_bool('camera_left_right', False,
                  'Control whether we log left and right cameras.')
flags.DEFINE_float('offset_left_right', 0.05,
                   'How much we offset the left and right cameras '
                   'from the center.')


def create_camera_setups():
    # Note: main assumes that the first camera setup returned by this method is
    # always the rgb_camera_setup.
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
    if FLAGS.camera_left_right:
        location_left = pylot.simulation.utils.Location(
            2.0, -1 * FLAGS.offset_left_right, 1.4)
        rotation_left = pylot.simulation.utils.Rotation(0, 0, 0)
        transform_left = pylot.simulation.utils.Transform(
            location_left, rotation_left)

        left_camera_setup = pylot.simulation.utils.CameraSetup(
            LEFT_CAMERA_NAME,
            'sensor.camera.rgb',
            FLAGS.carla_camera_image_width,
            FLAGS.carla_camera_image_height,
            transform_left)

        location_right = pylot.simulation.utils.Location(
            2.0, FLAGS.offset_left_right, 1.4)
        rotation_right = pylot.simulation.utils.Rotation(0, 0, 0)
        transform_right = pylot.simulation.utils.Transform(
            location_right, rotation_right)

        right_camera_setup = pylot.simulation.utils.CameraSetup(
            RIGHT_CAMERA_NAME,
            'sensor.camera.rgb',
            FLAGS.carla_camera_image_width,
            FLAGS.carla_camera_image_height,
            transform_right)
        return [rgb_camera_setup, depth_camera_setup, segmented_camera_setup,
                left_camera_setup, right_camera_setup]
    else:
        return [rgb_camera_setup, depth_camera_setup, segmented_camera_setup]


def create_lidar_setups():
    lidar_setups = []
    if FLAGS.lidar:
        location = pylot.simulation.utils.Location(2.0, 0.0, 1.4)
        rotation = pylot.simulation.utils.Rotation(0, 0, 0)
        lidar_transform = pylot.simulation.utils.Transform(location, rotation)
        lidar_setup = pylot.simulation.utils.LidarSetup(
            name='front_center_lidar',
            lidar_type='sensor.lidar.ray_cast',
            transform=lidar_transform,
            range=5000,  # in centimeters
            rotation_frequency=20,
            channels=32,
            upper_fov=15,
            lower_fov=-30,
            points_per_second=500000)
        lidar_setups.append(lidar_setup)
    return lidar_setups


def main(argv):
    # Define graph
    graph = erdos.graph.get_current_graph()

    camera_setups = create_camera_setups()
    lidar_setups = create_lidar_setups()

    # Add operator that interacts with the Carla simulator.
    (carla_op,
     camera_ops,
     lidar_ops) = pylot.operator_creator.create_driver_ops(
         graph, camera_setups, lidar_setups)

    # Add visual operators.
    pylot.operator_creator.add_visualization_operators(
        graph, camera_ops, lidar_ops, CENTER_CAMERA_NAME, DEPTH_CAMERA_NAME)

    # Add an operator that logs BGR frames and segmented frames.
    camera_log_ops = [pylot.operator_creator.create_camera_logger_op(graph)]
    lidar_log_ops = [pylot.operator_creator.create_lidar_logger_op(graph)]

    # Connect the camera logging ops with the camera ops.
    graph.connect(camera_ops, camera_log_ops)
    graph.connect(lidar_ops, lidar_log_ops)

    # Add operator that converts from 3D bounding boxes to 2D bouding boxes.
    detector_ops = [
        pylot.operator_creator.create_perfect_detector_op(
            graph, camera_setups[0])]
    # Connect the detector to the cameras.
    graph.connect([carla_op] + camera_ops, detector_ops)

    # Add operator that logs bboxes to json.
    bbox_logger_ops = [
        pylot.operator_creator.create_bounding_box_logger_op(graph)]
    graph.connect(detector_ops, bbox_logger_ops)

    # Add agent that uses ground data to drive around.
    agent_op = pylot.operator_creator.create_ground_agent_op(graph)
    graph.connect([carla_op], [agent_op])
    graph.connect([agent_op], [carla_op])

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
