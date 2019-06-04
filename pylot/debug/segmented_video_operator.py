import cv2
import numpy as np
import PIL.Image as Image

from erdos.op import Op
from erdos.utils import setup_logging

from perception.segmentation.utils import transform_to_cityscapes_palette
from pylot_utils import rgb_to_bgr, is_ground_segmented_camera_stream


class SegmentedVideoOperator(Op):
    def __init__(self, name, flags, log_file_name=None):
        super(SegmentedVideoOperator, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._flags = flags

    @staticmethod
    def setup_streams(input_streams):
        input_streams.filter(is_ground_segmented_camera_stream) \
                     .add_callback(SegmentedVideoOperator.display_frame)
        return []

    def display_frame(self, msg):
        frame = transform_to_cityscapes_palette(msg.frame)
        img = Image.fromarray(np.uint8(frame))
        open_cv_image = rgb_to_bgr(np.array(img))
        cv2.imshow(self.name, open_cv_image)
        cv2.waitKey(1)

    def execute(self):
        self.spin()
