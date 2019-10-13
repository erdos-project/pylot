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
flags.DEFINE_bool('chauffeur', False, 'True to log data in ChauffeurNet style.')


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
    camera_setups = []

    # Note: main assumes that the first camera setup returned by this method is
    # always the rgb_camera_setup.
    location = pylot.simulation.utils.Location(1.5, 0.0, 1.4)
    rotation = pylot.simulation.utils.Rotation(0, 0, 0)
    transform = pylot.simulation.utils.Transform(location, rotation)
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

    if FLAGS.chauffeur:
        # Height calculation relies on the fact that the camera's FOV is 90.
        top_down_location = pylot.simulation.utils.Location(1.5, 0.0, 1.4 + FLAGS.top_down_lateral_view)
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
        top_down_camera_setup = pylot.simulation.utils.CameraSetup(
            pylot.utils.TOP_DOWN_CAMERA_NAME,
            'sensor.camera.rgb',
            FLAGS.carla_camera_image_width,
            FLAGS.carla_camera_image_height,
            top_down_transform,
            fov=90)
        camera_setups.append(top_down_camera_setup)

    return camera_setups, top_down_segmented_camera_setup


def create_lidar_setups():
    lidar_setups = []
    if FLAGS.lidar and not FLAGS.chauffeur:
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

    camera_setups, chauffeur_camera_setup = create_camera_setups()
    lidar_setups = create_lidar_setups()

    # Add operator that interacts with the Carla simulator.
    (carla_op,
     camera_ops,
     _) = pylot.operator_creator.create_driver_ops(
        graph, camera_setups, lidar_setups, auto_pilot=FLAGS.carla_auto_pilot)

    chauffeur_log_ops = [pylot.operator_creator.create_chauffeur_logger_op(graph, chauffeur_camera_setup,
                                                                           pylot.utils.TOP_DOWN_SEGMENTED_CAMERA_NAME)]
    graph.connect(camera_ops, chauffeur_log_ops)

    tracking_ops = add_perfect_tracking_component(graph, carla_op)
    graph.connect(camera_ops + tracking_ops, chauffeur_log_ops)
    if FLAGS.carla_auto_pilot:
        # We do not need planning and agent ops if we're running in
        # auto pilot mode. Instead, we insert a synchronizing operator
        # that only sends back a command when all the operators in their
        # data-flow have finished processing a message.
        sync_op = graph.add(SynchronizerOp, name='sync_op')
        graph.connect(
            camera_ops + chauffeur_log_ops,
            [sync_op])
        graph.connect([sync_op], [carla_op])
        graph.connect([carla_op], chauffeur_log_ops)

    else:
        raise ValueError("Must be in auto pilot mode.")

    graph.execute(FLAGS.framework)


if __name__ == '__main__':
    app.run(main)
