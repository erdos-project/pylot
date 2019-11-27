import cv2
import erdust
import numpy as np
import PIL.Image as Image

from pylot.perception.segmentation.utils import transform_to_cityscapes_palette
from pylot.utils import rgb_to_bgr


class SegmentedVideoOperator(erdust.Operator):
    """ Subscribes to the ground segmented stream, and visualizes frames."""
    def __init__(self, segmented_camera_stream, name):
        self.name = name
        segmented_camera_stream.add_callback(self.display_frame)

    @staticmethod
    def connect(segmented_camera_stream):
        return []

    def display_frame(self, msg):
        frame = transform_to_cityscapes_palette(msg.frame)
        img = Image.fromarray(np.uint8(frame))
        open_cv_image = rgb_to_bgr(np.array(img))
        cv2.imshow(self.name, open_cv_image)
        cv2.waitKey(1)
