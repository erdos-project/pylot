from absl import app
from absl import flags

import pylot.config
from pylot.loggers.bounding_box_logger_operator import BoundingBoxLoggerOp
from pylot.loggers.camera_logger_operator import CameraLoggerOp
from pylot.loggers.lidar_logger_operator import LidarLoggerOp
from pylot.simulation.perfect_detector_operator import PerfectDetectorOp
import pylot.simulation.utils
import pylot.operator_creator
import pylot.utils

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
flags.DEFINE_integer('timeout', 10,
                     'Timeout limit for Carla operator')

# Flags for logging data from left and right cameras.
flags.DEFINE_bool('camera_left_right', False,
                  'Control whether we log left and right cameras.')
flags.DEFINE_float('offset_left_right', 0.05,
                  'How much we offset the left and right cameras from the center.')

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
    if FLAGS.camera_left_right:
        location_left = pylot.simulation.utils.Location(2.0, -1 * FLAGS.offset_left_right, 1.4)
        rotation_left = pylot.simulation.utils.Rotation(0, 0, 0)
        transform_left = pylot.simulation.utils.Transform(location_left, rotation_left)

        left_camera_setup = pylot.simulation.utils.CameraSetup(
            LEFT_CAMERA_NAME,
            'sensor.camera.rgb',
            FLAGS.carla_camera_image_width,
            FLAGS.carla_camera_image_height,
            transform_left)

        location_right = pylot.simulation.utils.Location(2.0, FLAGS.offset_left_right, 1.4)
        rotation_right = pylot.simulation.utils.Rotation(0, 0, 0)
        transform_right = pylot.simulation.utils.Transform(location_right, rotation_right)

        right_camera_setup = pylot.simulation.utils.CameraSetup(
            RIGHT_CAMERA_NAME,
            'sensor.camera.rgb',
            FLAGS.carla_camera_image_width,
            FLAGS.carla_camera_image_height,
            transform_right)
        return (rgb_camera_setup, depth_camera_setup, segmented_camera_setup,
                left_camera_setup, right_camera_setup)
    else:
        return (rgb_camera_setup, depth_camera_setup, segmented_camera_setup)


def create_camera_logger_op(graph):
    camera_logger_op = graph.add(
        CameraLoggerOp,
        name='camera_logger_op',
        init_args={'flags': FLAGS,
                   'log_file_name': FLAGS.log_file_name,
                   'csv_file_name': FLAGS.csv_log_file_name})
    return camera_logger_op

def create_lidar_logger_op(graph):
    lidar_logger_op = graph.add(
        LidarLoggerOp,
        name='lidar_logger_op',
        init_args={'flags': FLAGS,
                   'log_file_name': FLAGS.log_file_name,
                   'csv_file_name': FLAGS.csv_log_file_name})
    return lidar_logger_op


def create_perfect_detector_op(graph, bgr_camera_setup):
    output_stream_name = bgr_camera_setup.name + '_detected'
    perfect_det_op = graph.add(
        PerfectDetectorOp,
        name='perfect_detector',
        init_args={'bgr_camera_setup': bgr_camera_setup,
                   'output_stream_name': output_stream_name,
                   'flags': FLAGS,
                   'log_file_name': FLAGS.log_file_name,
                   'csv_file_name': FLAGS.csv_log_file_name},
        setup_args={'output_stream_name': output_stream_name})
    return perfect_det_op


def create_bounding_box_logger_op(graph):
    bbox_logger_op = graph.add(
        BoundingBoxLoggerOp,
        name='bounding_box_logger',
        init_args={'flags': FLAGS})
    return bbox_logger_op

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

    if FLAGS.camera_left_right:
        (bgr_camera_setup,
         depth_camera_setup,
         segmented_camera_setup,
         left_camera_setup,
         right_camera_setup) = create_camera_setups()
        camera_setups = [bgr_camera_setup,
                         depth_camera_setup,
                         segmented_camera_setup,
                         left_camera_setup,
                         right_camera_setup]
    else:
        (bgr_camera_setup,
         depth_camera_setup,
         segmented_camera_setup) = create_camera_setups()
        camera_setups = [bgr_camera_setup,
                         depth_camera_setup,
                         segmented_camera_setup]
    
    lidar_setups = create_lidar_setups()

    lidar_setups = create_lidar_setups()

    # Add operator that interacts with the Carla simulator.
    if '0.8' in FLAGS.carla_version:
        carla_op = pylot.operator_creator.create_carla_legacy_op(
            graph, camera_setups, lidar_setups)
        # The legacy carla op implements the camera drivers.
        camera_ops = [carla_op]
    elif '0.9' in FLAGS.carla_version:
        carla_op = pylot.operator_creator.create_carla_op(graph)
        camera_ops = [pylot.operator_creator.create_camera_driver_op(graph, cs)
                      for cs in camera_setups]
        lidar_ops = [pylot.operator_creator.create_lidar_driver_op(graph, ls)
                     for ls in lidar_setups]
        graph.connect([carla_op], camera_ops + lidar_ops)
    else:
        raise ValueError(
            'Unexpected Carla version {}'.format(FLAGS.carla_version))

    # Add an operator that logs BGR frames and segmented frames.
    logging_ops = [create_camera_logger_op(graph), create_lidar_logger_op(graph)]

    # Connect the camera logging ops with the camera ops.
    graph.connect(camera_ops, logging_ops)

    # Add operator that converts from 3D bounding boxes to 2D bounding boxes.
    detector_ops = [create_perfect_detector_op(graph, bgr_camera_setup)]
    # Connect the detector to the cameras.
    if '0.8' in FLAGS.carla_version:
        graph.connect([carla_op], detector_ops)
    elif '0.9' in FLAGS.carla_version:
        graph.connect([carla_op] + camera_ops, detector_ops)

    # Add operator that logs bboxes to json.
    bbox_logger_ops = [create_bounding_box_logger_op(graph)]
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
