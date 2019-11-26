import erdust
import numpy as np
import os
import PIL.Image as Image

from pylot.perception.segmentation.utils import transform_to_cityscapes_palette


class SemanticCameraLoggerOp(erdust.Operator):
    """ Logs segmented frames using the Cityscapes palette."""

    def __init__(self, segmented_camera_stream, name, flags, filename_prefix):
        segmented_camera_stream.add_callback(self.on_segmented_frame)
        self._flags = flags
        self._depth_frame_cnt = 0
        self._filename_prefix = filename_prefix

    @staticmethod
    def connect(segmented_camera_stream):
        return []

    def on_segmented_frame(self, msg):
        self._segmented_frame_cnt += 1
        if self._segmented_frame_cnt % self._flags.log_every_nth_frame != 0:
            return
        frame = transform_to_cityscapes_palette(msg.frame)
        # Write the segmented image.
        img = Image.fromarray(np.uint8(frame))
        file_name = os.path.join(
            self._flags.data_path,
            self._filename_prefix + str(msg.timestamp.coordinates[0]) + '.png')
        img.save(file_name)
