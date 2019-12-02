import erdust
import numpy as np
import os
import PIL.Image as Image

import pylot.utils


class RGBCameraLoggerOperator(erdust.Operator):
    """ Logs RGB camera frames."""

    def __init__(self, bgr_camera_stream, name, flags, filename_prefix):
        bgr_camera_stream.add_callback(self.on_bgr_frame)
        self._flags = flags
        self._bgr_frame_cnt = 0
        self._filename_prefix = filename_prefix

    @staticmethod
    def connect(bgr_camera_stream):
        return []

    def on_bgr_frame(self, msg):
        self._bgr_frame_cnt += 1
        if self._bgr_frame_cnt % self._flags.log_every_nth_frame != 0:
            return
        # Write the image.
        assert msg.encoding == 'BGR', 'Expects BGR frames'
        rgb_array = pylot.utils.bgr_to_rgb(msg.frame)
        file_name = os.path.join(
            self._flags.data_path,
            self._filename_prefix + str(msg.timestamp.coordinates[0]) + '.png')
        rgb_img = Image.fromarray(np.uint8(rgb_array))
        rgb_img.save(file_name)
