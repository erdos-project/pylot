import cv2
import numpy as np
import os
import PIL.Image as Image

import pylot.perception.detection.utils
import pylot.utils


class CameraFrame(object):
    """Class that stores camera frames.

    Args:
        frame: A numpy array storring the frame.
        camera_setup (:py:class:`~pylot.drivers.sensor_setup.CameraSetup`):
            The camera setup used by the sensor that generated this frame.

    Attributes:
        frame: A numpy array storring the frame.
        encoding (:obj:`str`): The encoding of the frame (BGR | RGB).
        camera_setup (:py:class:`~pylot.drivers.sensor_setup.CameraSetup`):
            The camera setup used by the sensor that generated this frame.
    """
    def __init__(self, frame, encoding, camera_setup=None):
        self.frame = frame
        if encoding != 'BGR' and encoding != 'RGB':
            raise ValueError('Unsupported encoding {}'.format(encoding))
        self.encoding = encoding
        self.camera_setup = camera_setup

    @classmethod
    def from_carla_frame(cls, carla_frame, camera_setup):
        """Creates a pylot camera frame from a CARLA frame.

        Returns:
            :py:class:`.CameraFrame`: A BGR camera frame.
        """
        import carla
        if not isinstance(carla_frame, carla.Image):
            raise ValueError('carla_frame should be of type carla.Image')
        _frame = np.frombuffer(carla_frame.raw_data, dtype=np.dtype("uint8"))
        _frame = np.reshape(_frame, (carla_frame.height, carla_frame.width, 4))
        return cls(_frame[:, :, :3], 'BGR', camera_setup)

    def as_numpy_array(self):
        """Returns the camera frame as a numpy array."""
        return self.frame.astype(np.uint8)

    def as_bgr_numpy_array(self):
        """Returns the camera frame as a BGR encoded numpy array."""
        if self.encoding == 'RGB':
            return self.frame[:, :, ::-1]
        else:
            return self.frame

    def as_rgb_numpy_array(self):
        """Returns the camera frame as a RGB encoded numpy array."""
        if self.encoding == 'BGR':
            return self.frame[:, :, ::-1]
        else:
            return self.frame

    def annotate_with_bounding_boxes(
        self,
        timestamp,
        detected_obstacles,
        bbox_color_map=pylot.perception.detection.utils.GROUND_COLOR_MAP):
        pylot.utils.add_timestamp(self.frame, timestamp)
        for obstacle in detected_obstacles:
            obstacle.visualize_on_img(self.frame, bbox_color_map)

    def visualize(self, window_name, timestamp=None):
        """Creates a cv2 window to visualize the camera frame."""
        if self.encoding != 'BGR':
            image_np = self.as_bgr_numpy_array()
        else:
            image_np = self.frame
        if timestamp is not None:
            pylot.utils.add_timestamp(image_np, timestamp)
        cv2.imshow(window_name, image_np)
        cv2.waitKey(1)

    def draw_point(self, point, color, r=3):
        cv2.circle(self.frame, (int(point.x), int(point.y)), r, color, -1)

    def save(self, timestamp, data_path, file_base):
        """Saves the camera frame to a file.

        Args:
            timestamp (:obj:`int`): Timestamp associated with the camera frame.
            data_path (:obj:`str`): Path where to save the camera frame.
            file_base (:obj:`str`): Base name of the file.
        """
        if self.encoding != 'RGB':
            image_np = self.as_rgb_numpy_array()
        else:
            image_np = self.frame
        file_name = os.path.join(data_path,
                                 '{}-{}.png'.format(file_base, timestamp))
        img = Image.fromarray(image_np)
        img.save(file_name)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'CameraFrame(encoding: {}, camera_setup: {})'.format(
            self.encoding, self.camera_setup)
