from absl import app
from absl import flags

import erdos.graph
from erdos.op import Op
from erdos.timestamp import Timestamp

import pylot.config
from pylot.control.messages import ControlMessage
import pylot.operator_creator
import pylot.simulation.utils
import pylot.utils

FLAGS = flags.FLAGS
CENTER_CAMERA_NAME = 'front_rgb_camera'
DEPTH_CAMERA_NAME = 'front_depth_camera'
SEGMENTED_CAMERA_NAME = 'front_semantic_camera'
LEFT_CAMERA_NAME = 'left_rgb_camera'
RIGHT_CAMERA_NAME = 'right_rgb_camera'

# Flags that control what data is recorded.
flags.DEFINE_integer('log_every_nth_frame', 1,
                     'Control how often the script logs frames')

# Flags for logging data from left and right cameras.
flags.DEFINE_bool('camera_left_right', False,
                  'Control whether we log left and right cameras.')
flags.DEFINE_float('offset_left_right', 0.05,
                   'How much we offset the left and right cameras '
                   'from the center.')
flags.DEFINE_bool('log_bounding_boxes', True,
                  'True to enable bounding box logging')


class SynchronizerOp(Op):
    def __init__(self, name):
        super(SynchronizerOp, self).__init__(name)

    @staticmethod
    def setup_streams(input_streams):
        input_streams.add_completion_callback(SynchronizerOp.on_watermark)
        # Set no watermark on the output stream so that we do not
        # close the watermark loop with the carla operator.
        return [pylot.utils.create_control_stream()]

    def on_watermark(self, msg):
        control_msg = ControlMessage(
            0, 0, 0, False, False, msg.timestamp)
        self.get_output_stream('control_stream').send(control_msg)


def create_camera_setups():
    # Note: main assumes that the first camera setup returned by this method is
    # always the rgb_camera_setup.
    location = pylot.simulation.utils.Location(1.5, 0.0, 1.4)
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
            1.5, -1 * FLAGS.offset_left_right, 1.4)
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
            1.5, FLAGS.offset_left_right, 1.4)
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
        location = pylot.simulation.utils.Location(1.5, 0.0, 1.4)
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


def add_perfect_detection_component(
        graph, camera_setups, carla_op, camera_ops):
    if FLAGS.carla_auto_pilot and not FLAGS.log_bounding_boxes:
        return [], []
    # Add operator that converts from 3D bounding boxes to 2D bouding boxes.
    ground_obstacles_stream_name = 'perfect_detector'
    detector_ops = [
        pylot.operator_creator.create_perfect_detector_op(
            graph, camera_setups[0], ground_obstacles_stream_name)]
    # Connect the detector to the cameras.
    graph.connect([carla_op] + camera_ops, detector_ops)

    if FLAGS.log_bounding_boxes:
        # Add operator that logs bboxes to json.
        bbox_logger_ops = [
            pylot.operator_creator.create_bounding_box_logger_op(graph)]
        graph.connect(detector_ops, bbox_logger_ops)
    else:
        bbox_logger_ops = []
    return detector_ops, bbox_logger_ops

def add_perfect_tracking_component(
        graph, carla_op):
    if not FLAGS.perfect_tracking:
        return []
    ground_tracking_stream_name = 'perfect_tracker'
    tracking_op = [
        pylot.operator_creator.create_perfect_tracking_op(
            graph, ground_tracking_stream_name)]
    graph.connect([carla_op], tracking_op)
    return tracking_op

def main(argv):
    # Define graph
    graph = erdos.graph.get_current_graph()

    camera_setups = create_camera_setups()
    lidar_setups = create_lidar_setups()

    # Add operator that interacts with the Carla simulator.
    (carla_op,
     camera_ops,
     lidar_ops) = pylot.operator_creator.create_driver_ops(
         graph, camera_setups, lidar_setups, auto_pilot=FLAGS.carla_auto_pilot)

    # Add visual operators.
    pylot.operator_creator.add_visualization_operators(
        graph, camera_ops, lidar_ops, CENTER_CAMERA_NAME, DEPTH_CAMERA_NAME)

    # Add an operator that logs BGR frames and segmented frames.
    camera_log_ops = [pylot.operator_creator.create_camera_logger_op(graph)]
    lidar_log_ops = [pylot.operator_creator.create_lidar_logger_op(graph)]

    # Connect the camera logging ops with the camera ops.
    graph.connect(camera_ops, camera_log_ops)
    graph.connect(lidar_ops, lidar_log_ops)

    detector_ops, bbox_logger_ops = add_perfect_detection_component(
        graph, camera_setups, carla_op, camera_ops)

    tracking_ops = add_perfect_tracking_component(graph, carla_op)
    trajectory_log_ops = [pylot.operator_creator.create_trajectory_logger_op(graph)]
    graph.connect(tracking_ops, trajectory_log_ops)

    if FLAGS.carla_auto_pilot:
        # We do not need planning and agent ops if we're running in
        # auto pilot mode. Instead, we insert a synchronizing operator
        # that only sends back a command when all the operators in their
        # data-flow have finished processing a message.
        sync_op = graph.add(SynchronizerOp, name='sync_op')
        graph.connect(
            camera_ops + lidar_ops + camera_log_ops + lidar_log_ops +
            detector_ops + bbox_logger_ops,
            [sync_op])
        graph.connect([sync_op], [carla_op])
    else:
        # Add agent that uses ground data to drive around.
        agent_op = pylot.operator_creator.create_ground_agent_op(graph)
        graph.connect([carla_op], [agent_op])
        graph.connect([agent_op], [carla_op])

        goal_location = (234.269989014, 59.3300170898, 39.4306259155)
        goal_orientation = (1.0, 0.0, 0.22)

        if '0.8' in FLAGS.carla_version:
            planning_op = pylot.operator_creator.create_legacy_planning_op(
                graph, 'Town{:02d}'.format(FLAGS.carla_town),
                goal_location, goal_orientation)
        elif '0.9' in FLAGS.carla_version:
            planning_op = pylot.operator_creator.create_planning_op(
                graph, goal_location)
        else:
            raise ValueError('Unexpected Carla version')
        graph.connect([carla_op], [planning_op])
        graph.connect([planning_op], [agent_op])

    graph.execute(FLAGS.framework)


if __name__ == '__main__':
    app.run(main)
