from absl import app
from absl import flags

import erdos

import pylot.flags
import pylot.operator_creator
import pylot.utils
from pylot.control.messages import ControlMessage
from pylot.drivers.sensor_setup import DepthCameraSetup, RGBCameraSetup, \
    SegmentedCameraSetup

FLAGS = flags.FLAGS

flags.DEFINE_bool('compute_detection_decay', False,
                  'True to enable ground truth object detection evaluation.')
flags.DEFINE_bool('compute_segmentation_decay', False,
                  'True to enable ground truth segmentation evaluation')
flags.DEFINE_integer('decay_max_latency', 400,
                     'Max latency to evaluate in ground truth experiments')

# The location of the center camera relative to the ego-vehicle.
CENTER_CAMERA_LOCATION = pylot.utils.Location(1.0, 0.0, 1.8)


class SynchronizerOperator(erdos.Operator):
    def __init__(self, wait_stream, control_stream, flags):
        erdos.add_watermark_callback([wait_stream], [control_stream],
                                     self.on_watermark)
        self._flags = flags

    @staticmethod
    def connect(wait_stream):
        # Set no watermark on the output stream so that we do not
        # close the watermark loop with the simulator bridge operator.
        control_stream = erdos.WriteStream()
        return [control_stream]

    def on_watermark(self, timestamp, control_stream):
        # The control message is ignored by the bridge operator because
        # data gathering is conducted using auto pilot. Send default control
        # message.
        control_msg = ControlMessage(0, 0, 0, False, False, timestamp)
        control_stream.send(control_msg)


def main(argv):
    """ Computes ground obstacle detection and segmentation decay."""
    transform = pylot.utils.Transform(CENTER_CAMERA_LOCATION,
                                      pylot.utils.Rotation())

    control_loop_stream = erdos.LoopStream()
    release_sensor_stream = erdos.IngestStream()
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
    ) = pylot.operator_creator.add_simulator_bridge(control_loop_stream,
                                                    release_sensor_stream)

    # Add camera sensors.
    rgb_camera_setup = RGBCameraSetup('center_camera',
                                      FLAGS.camera_image_width,
                                      FLAGS.camera_image_height, transform,
                                      FLAGS.camera_fov)
    (center_camera_stream,
     _) = pylot.operator_creator.add_camera_driver(rgb_camera_setup,
                                                   vehicle_id_stream,
                                                   release_sensor_stream)

    depth_camera_setup = DepthCameraSetup('depth_center_camera',
                                          FLAGS.camera_image_width,
                                          FLAGS.camera_image_height, transform,
                                          FLAGS.camera_fov)
    (depth_camera_stream,
     _) = pylot.operator_creator.add_camera_driver(depth_camera_setup,
                                                   vehicle_id_stream,
                                                   release_sensor_stream)

    segmented_camera_setup = SegmentedCameraSetup('segmented_center_camera',
                                                  FLAGS.camera_image_width,
                                                  FLAGS.camera_image_height,
                                                  transform, FLAGS.camera_fov)
    (segmented_stream,
     _) = pylot.operator_creator.add_camera_driver(segmented_camera_setup,
                                                   vehicle_id_stream,
                                                   release_sensor_stream)

    map_stream = None
    if FLAGS.compute_detection_decay:
        obstacles_stream = pylot.operator_creator.add_perfect_detector(
            depth_camera_stream, center_camera_stream, segmented_stream,
            pose_stream, ground_obstacles_stream,
            ground_speed_limit_signs_stream, ground_stop_signs_stream)
        map_stream = pylot.operator_creator.add_detection_decay(
            obstacles_stream)

    iou_stream = None
    if FLAGS.compute_segmentation_decay:
        iou_stream = pylot.operator_creator.add_segmentation_decay(
            segmented_stream)

    # TODO: Hack! We synchronize on a single stream, based on a guesestimated
    # of which stream is slowest. Instead, We should synchronize on all output
    # streams, and we should ensure that even the operators without output
    # streams complete.
    if FLAGS.control == 'simulator_auto_pilot':
        stream_to_sync_on = iou_stream
        if map_stream is not None:
            stream_to_sync_on = map_stream
        op_config = erdos.OperatorConfig(name='synchronizer_operator',
                                         flow_watermarks=False)
        (control_stream, ) = erdos.connect(SynchronizerOperator, op_config,
                                           [stream_to_sync_on], FLAGS)
        control_loop_stream.set(control_stream)
    else:
        raise ValueError(
            "Must be in auto pilot mode. Pass --control=simulator_auto_pilot")

    erdos.run_async()

    # Ask all sensors to release their data.
    release_sensor_stream.send(
        erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))


if __name__ == '__main__':
    app.run(main)
