import numpy as np
import PIL.Image as Image

import pylot.utils
from pylot.perception.segmentation.utils import transform_to_cityscapes_palette

from erdos.op import Op
from erdos.utils import setup_csv_logging, setup_logging


class CameraLoggerOp(Op):
    def __init__(self, name, flags, log_file_name=None, csv_file_name=None):
        super(CameraLoggerOp, self).__init__(name)
        self._flags = flags
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._last_bgr_timestamp = -1
        self._last_segmented_timestamp = -1

    @staticmethod
    def setup_streams(input_streams):
        input_streams.filter(pylot.utils.is_camera_stream).add_callback(
            CameraLoggerOp.on_bgr_frame)
        input_streams.filter(
            pylot.utils.is_ground_segmented_camera_stream).add_callback(
                CameraLoggerOp.on_segmented_frame)
        return []

    def on_bgr_frame(self, msg):
        # Ensure we didn't skip a frame.
        if self._last_bgr_timestamp != -1:
            assert self._last_bgr_timestamp + 1 == msg.timestamp.coordinates[1]
        self._last_bgr_timestamp = msg.timestamp.coordinates[1]
        if self._last_bgr_timestamp % self._flags.log_every_nth_frame != 0:
            return
        # Write the image.
        assert msg.encoding == 'BGR', 'Expects BGR frames'
        rgb_array = pylot.utils.bgr_to_rgb(msg.frame)
        file_name = '{}carla-{}.png'.format(
            self._flags.data_path, self._last_bgr_timestamp)
        rgb_img = Image.fromarray(np.uint8(rgb_array))
        rgb_img.save(file_name)

    def on_segmented_frame(self, msg):
        # Ensure we didn't skip a frame.
        if self._last_segmented_timestamp != -1:
            assert (self._last_segmented_timestamp + 1 ==
                    msg.timestamp.coordinates[1])
        self._last_segmented_timestamp = msg.timestamp.coordinates[1]
        if self._last_bgr_timestamp % self._flags.log_every_nth_frame != 0:
            return
        frame = transform_to_cityscapes_palette(msg.frame)
        # Write the segmented image.
        img = Image.fromarray(np.uint8(frame))
        file_name = '{}carla-segmented-{}.png'.format(
            self._flags.data_path, self._last_segmented_timestamp)
        img.save(file_name)
