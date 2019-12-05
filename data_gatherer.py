from absl import app
from absl import flags

import erdos.graph
from erdos.op import Op

import pylot.config
from pylot.control.messages import ControlMessage
import pylot.operator_creator
import pylot.simulation.utils
import pylot.utils


FLAGS = flags.FLAGS

# Flags that control what data is recorded.
flags.DEFINE_integer('log_every_nth_frame', 1,
                     'Control how often the script logs frames')

# Flags for logging data from left and right cameras.
flags.DEFINE_bool('log_left_right_cameras', False,
                  'Control whether we log left and right cameras.')
flags.DEFINE_float('offset_left_right_cameras', 0.05,
                   'How much we offset the left and right cameras '
                   'from the center.')
flags.DEFINE_bool('log_bounding_boxes', False,
                  'True to enable bounding box logging')
flags.DEFINE_bool('log_cameras', False, 'True to enable camera logging')
flags.DEFINE_bool('log_lidar', False, 'True to enable lidar logging')
flags.DEFINE_bool('log_trajectories', False,
                  'True to enable trajectory logging')
flags.DEFINE_bool('log_chauffeur', False,
                  'True to log data in ChauffeurNet style.')
flags.DEFINE_bool('log_kalman', False,
                    'True to log Kalman state estimates')


CENTER_CAMERA_LOCATION_X = 1.5
CENTER_CAMERA_LOCATION_Y = 0.0
CENTER_CAMERA_LOCATION_Z = 1.4
CENTER_CAMERA_LOCATION = pylot.simulation.utils.Location(
    CENTER_CAMERA_LOCATION_X,
    CENTER_CAMERA_LOCATION_Y,
    CENTER_CAMERA_LOCATION_Z)


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


def _add_depth_estimation_camera_setups(camera_setups):
    location_left = pylot.simulation.utils.Location(
        CENTER_CAMERA_LOCATION_X,
        CENTER_CAMERA_LOCATION_Y - FLAGS.offset_left_right_cameras,
        CENTER_CAMERA_LOCATION_Z)
    rotation_left = pylot.simulation.utils.Rotation(0, 0, 0)
    transform_left = pylot.simulation.utils.Transform(
        location_left, rotation_left)

    left_camera_setup = pylot.simulation.utils.CameraSetup(
        pylot.utils.LEFT_CAMERA_NAME,
        'sensor.camera.rgb',
        FLAGS.carla_camera_image_width,
        FLAGS.carla_camera_image_height,
        transform_left)
    camera_setups.append(left_camera_setup)

    location_right = pylot.simulation.utils.Location(
        CENTER_CAMERA_LOCATION_X,
        CENTER_CAMERA_LOCATION_Y + FLAGS.offset_left_right_cameras,
        CENTER_CAMERA_LOCATION_Z)
    rotation_right = pylot.simulation.utils.Rotation(0, 0, 0)
    transform_right = pylot.simulation.utils.Transform(
        location_right, rotation_right)

    right_camera_setup = pylot.simulation.utils.CameraSetup(
        pylot.utils.RIGHT_CAMERA_NAME,
        'sensor.camera.rgb',
        FLAGS.carla_camera_image_width,
        FLAGS.carla_camera_image_height,
        transform_right)
    camera_setups.append(right_camera_setup)


def _add_top_down_segmented_camera_setup(camera_setups):
    # Height calculation relies on the fact that the camera's FOV is 90.
    top_down_location = pylot.simulation.utils.Location(
        CENTER_CAMERA_LOCATION_X,
        CENTER_CAMERA_LOCATION_Y,
        CENTER_CAMERA_LOCATION_Z + FLAGS.top_down_lateral_view)
    top_down_rotation = pylot.simulation.utils.Rotation(-90, 0, 0)
    top_down_transform = pylot.simulation.utils.Transform(
        top_down_location, top_down_rotation)
    top_down_segmented_camera_setup = pylot.simulation.utils.CameraSetup(
        pylot.utils.TOP_DOWN_SEGMENTED_CAMERA_NAME,
        'sensor.camera.semantic_segmentation',
        FLAGS.carla_camera_image_width,
        FLAGS.carla_camera_image_height,
        top_down_transform,
        fov=90)
    camera_setups.append(top_down_segmented_camera_setup)
    return top_down_segmented_camera_setup


def _add_top_down_rgb_camera_setup(camera_setups):
    # Height calculation relies on the fact that the camera's FOV is 90.
    top_down_location = pylot.simulation.utils.Location(
        CENTER_CAMERA_LOCATION_X,
        CENTER_CAMERA_LOCATION_Y,
        CENTER_CAMERA_LOCATION_Z + FLAGS.top_down_lateral_view)
    top_down_rotation = pylot.simulation.utils.Rotation(-90, 0, 0)
    top_down_transform = pylot.simulation.utils.Transform(
        top_down_location, top_down_rotation)
    top_down_camera_setup = pylot.simulation.utils.CameraSetup(
        pylot.utils.TOP_DOWN_CAMERA_NAME,
        'sensor.camera.rgb',
        FLAGS.carla_camera_image_width,
        FLAGS.carla_camera_image_height,
        top_down_transform,
        fov=90)
    camera_setups.append(top_down_camera_setup)


def create_camera_setups():
    camera_setups = []

    # Note: main assumes that the first camera setup returned by this method is
    # always the rgb_camera_setup.
    rotation = pylot.simulation.utils.Rotation(0, 0, 0)
    transform = pylot.simulation.utils.Transform(
        CENTER_CAMERA_LOCATION, rotation)
    rgb_camera_setup = pylot.simulation.utils.CameraSetup(
        pylot.utils.CENTER_CAMERA_NAME,
        'sensor.camera.rgb',
        FLAGS.carla_camera_image_width,
        FLAGS.carla_camera_image_height,
        transform)
    camera_setups.append(rgb_camera_setup)
    depth_camera_setup = pylot.simulation.utils.CameraSetup(
        pylot.utils.DEPTH_CAMERA_NAME,
        'sensor.camera.depth',
        FLAGS.carla_camera_image_width,
        FLAGS.carla_camera_image_height,
        transform)
    camera_setups.append(depth_camera_setup)
    segmented_camera_setup = pylot.simulation.utils.CameraSetup(
        pylot.utils.FRONT_SEGMENTED_CAMERA_NAME,
        'sensor.camera.semantic_segmentation',
        FLAGS.carla_camera_image_width,
        FLAGS.carla_camera_image_height,
        transform)
    camera_setups.append(segmented_camera_setup)

    if FLAGS.log_left_right_cameras:
        _add_depth_estimation_camera_setups(camera_setups)

    top_down_segmented_camera_setup = None
    if FLAGS.top_down_segmentation or FLAGS.log_chauffeur:
        top_down_segmented_camera_setup = _add_top_down_segmented_camera_setup(
            camera_setups)
        if FLAGS.log_chauffeur:
            _add_top_down_rgb_camera_setup(camera_setups)

    return camera_setups, top_down_segmented_camera_setup


def create_lidar_setups():
    lidar_setups = []
    if FLAGS.log_lidar:
        rotation = pylot.simulation.utils.Rotation(0, 0, 0)
        lidar_transform = pylot.simulation.utils.Transform(
            CENTER_CAMERA_LOCATION, rotation)
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
    detector_ops = []
    bbox_logger_ops = []
    if FLAGS.log_bounding_boxes:
        # Add operator that converts from 3D bboxes to 2D bboxes.
        ground_obstacles_stream_name = 'perfect_detector'
        detector_ops = [
            pylot.operator_creator.create_perfect_detector_op(
                graph, camera_setups[0], ground_obstacles_stream_name)]
        # Connect the detector to the cameras.
        graph.connect([carla_op] + camera_ops, detector_ops)
        # Add operator that logs bboxes to json.
        bbox_logger_ops = [
            pylot.operator_creator.create_bounding_box_logger_op(graph)]
        graph.connect(detector_ops, bbox_logger_ops)
    return detector_ops, bbox_logger_ops


def add_perfect_tracking_component(graph, carla_op):
    if FLAGS.log_trajectories or FLAGS.log_chauffeur:
        ground_tracking_stream_name = 'perfect_tracker'
        tracking_op = [
            pylot.operator_creator.create_perfect_tracking_op(
                graph, ground_tracking_stream_name)]
        graph.connect([carla_op], tracking_op)
        return tracking_op
    else:
        return []


def main(argv):
    # Get default data-flow graph.
    graph = erdos.graph.get_current_graph()

    camera_setups, top_down_segmented_camera_setup = create_camera_setups()
    lidar_setups = create_lidar_setups()

    # Add operator that interacts with the Carla simulator.
    (carla_op,
     camera_ops,
     lidar_ops) = pylot.operator_creator.create_driver_ops(
         graph, camera_setups, lidar_setups, auto_pilot=FLAGS.carla_auto_pilot)

    # Add an operator that logs BGR frames and segmented frames.
    camera_log_ops = []
    if FLAGS.log_cameras:
        camera_log_ops = [
            pylot.operator_creator.create_camera_logger_op(graph)]
    lidar_log_ops = []
    if FLAGS.log_lidar:
        lidar_log_ops = [pylot.operator_creator.create_lidar_logger_op(graph)]

    chauffeur_log_ops = []
    if FLAGS.log_chauffeur:
        chauffeur_log_ops = [pylot.operator_creator.create_chauffeur_logger_op(
            graph, top_down_segmented_camera_setup,
            pylot.utils.TOP_DOWN_SEGMENTED_CAMERA_NAME)]

    kalman_op = []
    if FLAGS.log_kalman:
        kalman_op = [pylot.operator_creator.create_kalman_logger_op(graph)]
        graph.connect([carla_op], kalman_op)


    # Connect the camera logging ops with the camera ops.
    graph.connect(camera_ops, camera_log_ops + chauffeur_log_ops)
    graph.connect(lidar_ops, lidar_log_ops)

    detector_ops, bbox_logger_ops = add_perfect_detection_component(
        graph, camera_setups, carla_op, camera_ops)

    tracking_ops = add_perfect_tracking_component(graph, carla_op)
    trajectory_log_ops = []
    if FLAGS.log_trajectories:
        trajectory_log_ops = [
            pylot.operator_creator.create_trajectory_logger_op(graph)]
    graph.connect(tracking_ops, trajectory_log_ops + chauffeur_log_ops)

    multiple_object_tracker_logger_op = [
        pylot.operator_creator.create_multiple_object_tracker_logger_op(graph)]
    graph.connect(detector_ops, multiple_object_tracker_logger_op)

    # Add visual operators.
    pylot.operator_creator.add_visualization_operators(
        graph,
        camera_ops,
        lidar_ops,
        tracking_ops,
        pylot.utils.CENTER_CAMERA_NAME,
        pylot.utils.DEPTH_CAMERA_NAME,
        pylot.utils.FRONT_SEGMENTED_CAMERA_NAME,
        pylot.utils.TOP_DOWN_SEGMENTED_CAMERA_NAME,
        top_down_segmented_camera_setup)

    if FLAGS.carla_auto_pilot:
        # We do not need planning and agent ops if we're running in
        # auto pilot mode. Instead, we insert a synchronizing operator
        # that only sends back a command when all the operators in their
        # data-flow have finished processing a message.
        sync_op = graph.add(SynchronizerOp, name='sync_op')
        graph.connect(
            camera_ops + lidar_ops + camera_log_ops + lidar_log_ops +
            detector_ops + bbox_logger_ops + chauffeur_log_ops,
            [sync_op])
        graph.connect([sync_op], [carla_op])
        graph.connect([carla_op], chauffeur_log_ops)
    else:
        raise ValueError("Must be in auto pilot mode. Pass --carla_auto_pilot")

    graph.execute("ros")


if __name__ == '__main__':
    app.run(main)
