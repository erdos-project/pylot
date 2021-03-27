import os
from typing import Tuple, Union

import PIL.Image as Image

import cv2

import numpy as np

import pylot.perception.detection.utils
import pylot.utils
from pylot.drivers.sensor_setup import CameraSetup
from pylot.utils import Vector2D


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
    def __init__(self,
                 frame,
                 encoding: str,
                 camera_setup: Union[CameraSetup, None] = None):
        self.frame = frame
        if encoding != 'BGR' and encoding != 'RGB':
            raise ValueError('Unsupported encoding {}'.format(encoding))
        self.encoding = encoding
        self.camera_setup = camera_setup

    @classmethod
    def from_simulator_frame(cls, simulator_frame, camera_setup: CameraSetup):
        """Creates a pylot camera frame from a simulator frame.

        Returns:
            :py:class:`.CameraFrame`: A BGR camera frame.
        """
        from carla import Image
        if not isinstance(simulator_frame, Image):
            raise ValueError('simulator_frame should be of type Image')
        _frame = np.frombuffer(simulator_frame.raw_data,
                               dtype=np.dtype("uint8"))
        _frame = np.reshape(_frame,
                            (simulator_frame.height, simulator_frame.width, 4))
        return cls(np.array(_frame[:, :, :3]), 'BGR', camera_setup)

    def as_numpy_array(self):
        """Returns the camera frame as a numpy array."""
        return self.frame.astype(np.uint8)

    def as_bgr_numpy_array(self):
        """Returns the camera frame as a BGR encoded numpy array."""
        if self.encoding == 'RGB':
            return np.array(self.frame[:, :, ::-1])
        else:
            return self.frame

    def as_rgb_numpy_array(self):
        """Returns the camera frame as a RGB encoded numpy array."""
        if self.encoding == 'BGR':
            return self.frame[:, :, ::-1]
        else:
            return self.frame

    def annotate_with_bounding_boxes(self,
                                     timestamp,
                                     detected_obstacles,
                                     transform=None,
                                     bbox_color_map=pylot.perception.detection.
                                     utils.PYLOT_BBOX_COLOR_MAP):
        pylot.utils.add_timestamp(self.frame, timestamp)
        for obstacle in detected_obstacles:
            obstacle.draw_on_frame(self,
                                   bbox_color_map,
                                   ego_transform=transform)

    def draw_box(self,
                 start_point: Vector2D,
                 end_point: Vector2D,
                 color: Tuple[int, int, int],
                 thickness: float = 2):
        """Draw a colored box defined by start_point, end_point."""
        start = (int(start_point.x), int(start_point.y))
        end = (int(end_point.x), int(end_point.y))
        cv2.rectangle(self.frame, start, end, color, thickness)

    def draw_3d_box(self, corners, color: Tuple[int, int, int]):
        for corner in corners:
            self.draw_point(corner, color)

    def draw_point(self, point: Vector2D, color, r: float = 3):
        """Draws a point on the frame.

        Args:
            point (:py:class:`~pylot.utils.Vector2D`): Where to draw the point.
            color: RGB tuple for the color of the point.
        """
        cv2.circle(self.frame, (int(point.x), int(point.y)), r, color, -1)

    def draw_text(self,
                  point: Vector2D,
                  text: str,
                  color: Tuple[int, int, int] = (255, 255, 255)):
        """Draws text on the frame.

        Args:
            point (:py:class:`~pylot.utils.Vector2D`): Where to draw the text.
            text (:obj:`str`): The text to draw.
            color: RGB tuple for the color of the text.
        """
        cv2.putText(self.frame,
                    text, (int(point.x), int(point.y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    thickness=1,
                    lineType=cv2.LINE_AA)

    def in_frame(self, point: Vector2D) -> bool:
        """Checks if a point is within the frame."""
        return (0 <= point.x <= self.camera_setup.width
                and 0 <= point.y <= self.camera_setup.height)

    def resize(self, width: int, height: int):
        """Resizes the frame."""
        self.camera_setup.set_resolution(width, height)
        self.frame = cv2.resize(self.frame,
                                dsize=(width, height),
                                interpolation=cv2.INTER_NEAREST)

    def visualize(self, pygame_display, timestamp=None):
        """Visualizes the frame on a pygame display."""
        import pygame
        if timestamp is not None:
            pylot.utils.add_timestamp(self.frame, timestamp)
        if self.encoding != 'RGB':
            image_np = self.as_rgb_numpy_array()
        else:
            image_np = self.frame
        image_np = np.transpose(image_np, (1, 0, 2))
        pygame.surfarray.blit_array(pygame_display, image_np)
        pygame.display.flip()

    def save(self, timestamp: int, data_path: str, file_base: str):
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
        return 'CameraFrame(encoding: {}, camera_setup: {}, frame: {})'.format(
            self.encoding, self.camera_setup, self.frame)

    def __str__(self):
        return 'CameraFrame(encoding: {}, camera_setup: {})'.format(
            self.encoding, self.camera_setup)
