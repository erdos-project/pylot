"""This module implements an operator acts like a camera driver when
using the CARLA simulator.

The operator attaches a camera to the ego vehicle, receives camera frames from
CARLA, and sends them on its output stream.
"""

import erdos
import np

from pylot.perception.camera_frame import CameraFrame
from pylot.perception.messages import DepthFrameMessage, FrameMessage, \
    SegmentedFrameMessage
from pylot.perception.segmentation.segmented_frame import SegmentedFrame


class CarlaCameraLaneDetectionOperator(erdos.Operator):
    """Publishes images onto the desired stream from a camera.

    This operator attaches a vehicle at the required position with respect to
    the vehicle, registers callback functions to retrieve the images and
    publishes it to downstream operators.

    Args:
        detected_lane_stream (:py:class:`erdos.ReadStream`): Stream on
            which the operator receives the detected lanes.
        camera_setup (:py:class:`pylot.drivers.sensor_setup.RGBCameraSetup`):
            Setup of the camera.
        flags (absl.flags): Object to be used to access absl flags.
    """

    def __init__(self, detected_lane_stream: ReadStream, camera_setup, flags):
        detected_lane_stream.add_callback(self.on_lane_update, [])
        # erdos.add_watermark_callback([detected_lane_stream], [], self.on_watermark)
        self._detected_lane_stream = detected_lane_stream
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._camera_setup = camera_setup

    @staticmethod
    def connect(detected_lane_stream):
        return []

    @erdos.profile_method()
    def on_lane_update(self, lanes_message: Message):
        detected_lanes = lanes_message.data
        black_img = np.zeros((self._camera_setup.height,
                              self._camera_setup.width, 3),
                             dtype=np.dtype("uint8"))
        frame = CameraFrame(black_img, 'BGR',
                            self._camera_setup)
        for lane in detected_lanes:
            lane.draw_on_frame(frame)

        frame.save(lanes_message.timestamp, self._flags.data_path, "lane")

    def run(self):
        pass
