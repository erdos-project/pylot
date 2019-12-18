from absl import app
from absl import flags
import erdust

from pylot.control.messages import ControlMessage
import pylot.flags
import pylot.operator_creator
import pylot.simulation.utils

FLAGS = flags.FLAGS

# Flags that control what data is recorded.
flags.DEFINE_integer('log_every_nth_frame', 1,
                     'Control how often the script logs frames')
flags.DEFINE_bool('log_rgb_camera', False,
                  'True to enable center camera RGB logging')
flags.DEFINE_bool('log_segmented_camera', False,
                  'True to enable center segmented camera logging')
flags.DEFINE_bool('log_left_right_cameras', False,
                  'Control whether we log left and right cameras.')
flags.DEFINE_bool('log_depth_camera', False,
                  'True to enable depth camera logging')
flags.DEFINE_bool('log_imu', False,
                  'Enable logging of IMU measurements.')
flags.DEFINE_bool('log_lidar', False, 'True to enable lidar logging')
flags.DEFINE_bool('log_obstacles', False,
                  'True to enable obstacle bounding box logging')
flags.DEFINE_bool(
    'log_traffic_lights',
    False,
    'True to enable traffic lights bounding box and camera logging')
flags.DEFINE_bool('log_multiple_object_tracker', False,
                  'True to enable logging in the MOT format')
flags.DEFINE_bool('log_trajectories', False,
                  'True to enable trajectory logging')
flags.DEFINE_bool('log_chauffeur', False,
                  'True to log data in ChauffeurNet style.')
flags.DEFINE_bool('log_top_down_segmentation', False,
                  'True to enable logging of top down segmentation')

CENTER_CAMERA_LOCATION = pylot.simulation.utils.Location(1.5, 0.0, 1.4)


class SynchronizerOperator(erdust.Operator):
    def __init__(self,
                 wait_stream,
                 control_stream,
                 flags):
        erdust.add_watermark_callback(
            [wait_stream], [control_stream], self.on_watermark)
        self._flags = flags

    @staticmethod
    def connect(wait_stream):
        # Set no watermark on the output stream so that we do not
        # close the watermark loop with the carla operator.
        control_stream = erdust.WriteStream()
        return [control_stream]

    def on_watermark(self, timestamp, control_stream):
        # The control message is ignored by the bridge operator because
        # data gathering is conducted using auto pilot. Send default control
        # message.
        control_msg = ControlMessage(0, 0, 0, False, False, timestamp)
        control_stream.send(control_msg)


def driver():
    transform = pylot.simulation.utils.Transform(
        CENTER_CAMERA_LOCATION, pylot.simulation.utils.Rotation(0, 0, 0))

    control_loop_stream = erdust.LoopStream()
    # Create carla operator.
    (can_bus_stream,
     ground_traffic_lights_stream,
     ground_obstacles_stream,
     ground_speed_limit_signs_stream,
     ground_stop_signs_stream,
     vehicle_id_stream) = pylot.operator_creator.add_carla_bridge(
         control_loop_stream)

    # Add sensors.
    (center_camera_stream,
     rgb_camera_setup) = pylot.operator_creator.add_rgb_camera(
         transform, vehicle_id_stream)
    (depth_camera_stream,
     depth_camera_setup) = pylot.operator_creator.add_depth_camera(
         transform, vehicle_id_stream)
    (segmented_stream, _) = pylot.operator_creator.add_segmented_camera(
        transform, vehicle_id_stream)

    if FLAGS.log_rgb_camera:
        pylot.operator_creator.add_camera_logging(
            center_camera_stream,
            'center_camera_logger_operator',
            'carla-center-')

    if FLAGS.log_segmented_camera:
        pylot.operator_creator.add_camera_logging(
            segmented_stream,
            'center_segmented_camera_logger_operator',
            'carla-segmented-')

    if FLAGS.log_depth_camera:
        pylot.operator_creator.add_depth_camera_logging(depth_camera_stream)

    imu_stream = None
    if FLAGS.log_imu:
        (imu_stream, _) = pylot.operator_creator.add_imu(
            transform, vehicle_id_stream)
        pylot.operator_creator.add_imu_logging(imu_stream)

    traffic_lights_stream = None
    if FLAGS.log_traffic_lights:
        (traffic_light_camera_stream,
         traffic_light_camera_setup) = pylot.operator_creator.add_rgb_camera(
             transform, vehicle_id_stream, 'traffic_light_camera', 45)
        pylot.operator_creator.add_camera_logging(
            traffic_light_camera_stream,
            'traffic_light_camera_logger_operator',
            'carla-traffic-light-')
        (traffic_light_segmented_camera_stream, _) = \
            pylot.operator_creator.add_segmented_camera(
                transform,
                vehicle_id_stream,
                'traffic_light_segmented_camera',
                45)
        (traffic_light_depth_camera_stream, _) = \
            pylot.operator_creator.add_depth_camera(
                transform, vehicle_id_stream, 'traffic_light_depth_camera', 45)
        traffic_lights_stream = \
            pylot.operator_creator.add_perfect_traffic_light_detector(
                ground_traffic_lights_stream,
                traffic_light_camera_stream,
                traffic_light_depth_camera_stream,
                traffic_light_segmented_camera_stream,
                can_bus_stream)
        pylot.operator_creator.add_bounding_box_logging(traffic_lights_stream)

    if FLAGS.log_left_right_cameras:
        (left_camera_stream,
         right_camera_stream) = pylot.operator_creator.add_left_right_cameras(
             transform, vehicle_id_stream)
        pylot.operator_creator.add_camera_logging(
            left_camera_stream,
            'left_camera_logger_operator', 'carla-left-')
        pylot.operator_creator.add_camera_logging(
            right_camera_stream,
            'right_camera_logger_operator', 'carla-right-')

    point_cloud_stream = None
    if FLAGS.log_lidar:
        (point_cloud_stream, _) = pylot.operator_creator.add_lidar(
            transform, vehicle_id_stream)
        pylot.operator_creator.add_lidar_logging(point_cloud_stream)

    obstacles_stream = None
    if FLAGS.log_obstacles:
        obstacles_stream = pylot.operator_creator.add_perfect_detector(
            depth_camera_stream,
            center_camera_stream,
            segmented_stream,
            can_bus_stream,
            ground_obstacles_stream,
            ground_speed_limit_signs_stream,
            ground_stop_signs_stream,
            rgb_camera_setup)
        pylot.operator_creator.add_bounding_box_logging(obstacles_stream)

    if FLAGS.log_multiple_object_tracker:
        pylot.operator_creator.add_multiple_object_tracker_logging(
            obstacles_stream)

    obstacles_tracking_stream = None
    if FLAGS.log_trajectories or FLAGS.log_chauffeur:
        obstacles_tracking_stream = \
            pylot.operator_creator.add_perfect_tracking(
                ground_obstacles_stream,
                can_bus_stream)
        if FLAGS.log_trajectories:
            pylot.operator_creator.add_trajectory_logging(
                obstacles_tracking_stream)

    top_down_segmented_stream = None
    top_down_camera_setup = None
    if FLAGS.log_chauffeur or FLAGS.log_top_down_segmentation:
        top_down_transform = pylot.simulation.utils.get_top_down_transform(
            transform, FLAGS.top_down_lateral_view)
        (top_down_segmented_stream, _) = \
            pylot.operator_creator.add_segmented_camera(
                top_down_transform,
                vehicle_id_stream,
                name='top_down_segmented_camera',
                fov=90)

        if FLAGS.log_top_down_segmentation:
            pylot.operator_creator.add_camera_logging(
                top_down_segmented_stream,
                'top_down_segmented_logger_operator',
                'carla-top-down-segmented-')

        if FLAGS.log_chauffeur:
            (top_down_camera_stream,
             top_down_camera_setup) = \
                pylot.operator_creator.add_rgb_camera(
                    top_down_transform,
                    vehicle_id_stream,
                    name='top_down_rgb_camera',
                    fov=90)
            pylot.operator_creator.add_chauffeur_logging(
                vehicle_id_stream,
                can_bus_stream,
                obstacles_tracking_stream,
                top_down_camera_stream,
                top_down_segmented_stream,
                top_down_camera_setup)

    prediction_stream = None
    pylot.operator_creator.add_visualizers(
        center_camera_stream,
        depth_camera_stream,
        point_cloud_stream,
        segmented_stream,
        imu_stream,
        can_bus_stream,
        top_down_segmented_stream,
        obstacles_tracking_stream,
        prediction_stream,
        top_down_camera_setup)

    # TODO: Hack! We synchronize on a single stream, based on a guesestimated
    # of which stream is slowest. Instead, We should synchronize on all output
    # streams, and we should ensure that even the operators without output
    # streams complete.
    if FLAGS.carla_auto_pilot:
        # We insert a synchronizing operator that sends back a command when
        # the low watermark progresses on all input stream.
        stream_to_sync_on = center_camera_stream
        if obstacles_tracking_stream is not None:
            stream_to_sync_on = obstacles_tracking_stream
        if traffic_lights_stream is not None:
            stream_to_sync_on = traffic_lights_stream
        if obstacles_stream is not None:
            stream_to_sync_on = obstacles_stream
        (control_stream,) = erdust.connect(
            SynchronizerOperator,
            [stream_to_sync_on],
            False,  # Does not flow watermarks.
            FLAGS)
        control_loop_stream.set(control_stream)
    else:
        raise ValueError("Must be in auto pilot mode. Pass --carla_auto_pilot")


def main(argv):
    erdust.run(driver)


if __name__ == '__main__':
    app.run(main)
