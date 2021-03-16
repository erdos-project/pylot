import os

import PIL.Image as Image

import cv2

import numpy as np

import pylot.utils
from pylot.perception.detection.utils import BoundingBox2D

from skimage import measure

# Semantic Labels
CITYSCAPES_LABELS = {
    0: "unlabeled",
    1: "building",
    2: "fence",
    3: "other",
    4: "person",
    5: "pole",
    6: "road_line",
    7: "road",
    8: "sidewalk",
    9: "vegetation",
    10: "car",
    11: "wall",
    12: "traffic_sign",
    13: "sky",
    14: "ground",
    15: "bridge",
    16: "rail_track",
    17: "guard_rail",
    18: "traffic_light",
    19: "static",
    20: "dynamic",
    21: "water",
    22: "terrain",
}

# Cityscapes palette.
CITYSCAPES_CLASSES = {
    0: [0, 0, 0],  # None
    1: [70, 70, 70],  # Buildings
    2: [190, 153, 153],  # Fences
    3: [72, 0, 90],  # Other
    4: [220, 20, 60],  # Pedestrians
    5: [153, 153, 153],  # Poles
    6: [157, 234, 50],  # RoadLines
    7: [128, 64, 128],  # Roads
    8: [244, 35, 232],  # Sidewalks
    9: [107, 142, 35],  # Vegetation
    10: [0, 0, 255],  # Vehicles
    11: [102, 102, 156],  # Walls
    12: [220, 220, 0],  # TrafficSigns
    13: [70, 130, 180],  # Sky
    14: [81, 0, 81],  # Ground
    15: [150, 100, 100],  # Bridge
    16: [230, 150, 140],  # RailTrack
    17: [180, 165, 180],  # GuardRail
    18: [250, 170, 30],  # TrafficLight
    19: [110, 190, 160],  # Static
    20: [170, 120, 50],  # Dynamic
    21: [45, 60, 150],  # Water
    22: [145, 170, 100]  # Terrain
}

# XXX(ionel): Note! These classes do not cover all
# the classes from CITYSCAPES. Hence, we can't compare segmentation
# outputs to ground truth.


class SegmentedFrame(object):
    """Stores a semantically segmented frame.

    Args:
        frame: A numpy array storring the segmented frame.
        encoding (:obj:`str`): The encoding of the frame (simulator | cityscapes).  # noqa: E501
        camera_setup (:py:class:`~pylot.drivers.sensor_setup.SegmentedCameraSetup`):  # noqa: E501
            The camera setup used by the sensor that generated this frame.

    Attributes:
        frame: A numpy array storring the segmented frame.
        encoding (:obj:`str`): The encoding of the frame (simulator | cityscapes).  # noqa: E501
        camera_setup (:py:class:`~pylot.drivers.sensor_setup.SegmentedCameraSetup`):  # noqa: E501
            The camera setup used by the sensor that generated this frame.
    """
    def __init__(self, frame, encoding, camera_setup):
        if encoding == 'simulator' or encoding == 'cityscapes':
            self._frame = frame
            self.encoding = encoding
            self.camera_setup = camera_setup
        else:
            raise ValueError(
                'Unexpected encoding {} for segmented frame'.format(encoding))
        self._class_masks = None

    @classmethod
    def from_simulator_image(cls, simulator_image, camera_setup):
        """Creates a pylot camera frame from a simulator frame.

        Note:
            This conversion is slow.

        Returns:
            :py:class:`.SegmentedFrame`: A segmented camera frame.
        """
        # Converts the array containing simulator semantic segmentation labels
        # to a 2D array containing the label of each pixel.
        from carla import Image
        if not isinstance(simulator_image, Image):
            raise ValueError('simulator_image should be of type Image')
        __frame = np.frombuffer(simulator_image.raw_data,
                                dtype=np.dtype("uint8"))
        __frame = np.reshape(
            __frame, (simulator_image.height, simulator_image.width, 4))
        return cls(__frame[:, :, 2], 'simulator', camera_setup)

    def as_cityscapes_palette(self):
        """Returns the frame to the CARLA cityscapes pallete.

        Returns:
           A numpy array.
        """
        if self.encoding == 'cityscapes':
            return self._frame
        else:
            result = np.zeros((self._frame.shape[0], self._frame.shape[1], 3),
                              dtype=np.uint8)
            for key, value in CITYSCAPES_CLASSES.items():
                result[np.where(self._frame == key)] = value
            return result

    def as_numpy_array(self):
        """Returns the segmented frame as a numpy array."""
        return self._frame

    def transform_to_cityscapes(self):
        """Transforms the frame to a cityscapes frame."""
        self._frame = self.as_cityscapes_palette()
        self.encoding = 'cityscapes'

    def in_frame(self, point):
        """Checks if a point is within the frame."""
        return (0 <= point.x <= self.camera_setup.width
                and 0 <= point.y <= self.camera_setup.height)

    def get_traffic_sign_bounding_boxes(self, min_width=2, min_height=3):
        """Extracts traffic sign bounding boxes from the frame.

        Returns:
            list(:py:class:`~pylot.perception.detection.utils.BoundingBox2D`):
            Traffic sign bounding boxes.
        """
        assert self.encoding == 'simulator', \
            'Not implemented on cityscapes encoding'
        # Set the pixels we are interested in to True.
        traffic_signs_frame = self._get_traffic_sign_pixels()
        # Extracts bounding box from frame.
        bboxes = []
        # Labels the connected segmented pixels.
        map_labeled = measure.label(traffic_signs_frame, connectivity=1)
        # Extract the regions out of the labeled frames.
        for region in measure.regionprops(map_labeled):
            x_min = region.bbox[1]
            x_max = region.bbox[3]
            y_min = region.bbox[0]
            y_max = region.bbox[2]
            # Filter the bboxes that are extremely small.
            if x_max - x_min > min_width and y_max - y_min > min_height:
                bboxes.append(BoundingBox2D(x_min, x_max, y_min, y_max))
        return bboxes

    def _get_per_class_masks(self):
        """ Build a cache of class key to frame mask."""
        assert self.encoding == 'simulator', \
            'Not implemented on cityscapes encoding'
        if self._class_masks is not None:
            return self._class_masks
        else:
            self._class_masks = []
            for key, value in CITYSCAPES_CLASSES.items():
                mask = np.zeros((self._frame.shape[0], self._frame.shape[1]))
                mask[np.where(self._frame == key)] = 1
                self._class_masks.append(mask)
        return self._class_masks

    def compute_semantic_iou(self, other_frame):
        """Computes IoU for a segmented frame.

        Args:
            other_frame (:py:class:`.SegmentedFrame`): The frame for which to
            compute IoU.

        Returns:
            A tuple comprising of mIoU and a list of IoUs.
        """
        assert (self.encoding == 'cityscapes' and other_frame.encoding
                == 'cityscapes'), 'Not implemented on simulator encoding'
        iou = {}
        for key, value in CITYSCAPES_CLASSES.items():
            #  Do not include None in the mIoU
            if key == 0:
                continue
            target = np.zeros((self._frame.shape[0], self._frame.shape[1], 3))
            prediction = np.zeros(
                (self._frame.shape[0], self._frame.shape[1], 3))
            target[np.where(self._frame == value)] = 1
            prediction[np.where(other_frame._frame == value)] = 1
            intersection = np.logical_and(target, prediction)
            union = np.logical_or(target, prediction)
            sum_intersection = np.sum(intersection)
            sum_union = np.sum(union)
            # Ignore non-existing classes.
            if sum_union > 0:
                iou[key] = float(sum_intersection) / float(sum_union)
        mean_iou = np.mean(list(iou.values()))
        return (mean_iou, iou)

    def compute_semantic_iou_using_masks(self, other_frame):
        """Computes IoU for a segmented frame.

        Computes IoU from per class image masks. This method is the fastest if
        the frames already have precomputed masks.

        Args:
            other_frame (:py:class:`.SegmentedFrame`): The frame for which to
            compute IoU.

        Returns:
            A tuple comprising of mIoU and a list of IoUs.
        """
        assert self.encoding == 'simulator' and \
            other_frame.encoding == 'simulator', \
            'Not implemented on cityscapes encoding'
        masks = self._get_per_class_masks()
        other_masks = other_frame._get_per_class_masks()
        iou = {}
        for i in range(1, len(CITYSCAPES_CLASSES)):
            intersection = np.logical_and(masks[i], other_masks[i])
            union = np.logical_or(masks[i], other_masks[i])
            sum_intersection = np.sum(intersection)
            sum_union = np.sum(union)

            # Ignore non-existing classes.
            if sum_union > 0:
                iou[i] = float(sum_intersection) / float(sum_union)
                mean_iou = np.mean(list(iou.values()))
        return (mean_iou, iou)

    def save_per_class_masks(self, data_path, timestamp):
        assert self.encoding == 'simulator', \
            'Not implemented on cityscapes encoding'
        masks = self._get_per_class_masks()
        assert len(timestamp.coordinates) == 1
        for k, v in CITYSCAPES_LABELS.items():
            file_name = os.path.join(
                data_path, '{}-{}.png'.format(v, timestamp.coordinates[0]))
            img = Image.fromarray(masks[k])
            img = img.convert('RGB')
            img.save(file_name)

    def save(self, timestamp, data_path, file_base):
        """Saves the segmented frame to a file.

        Args:
            timestamp (:obj:`int`): Timestamp associated with the segmented
                frame.
            data_path (:obj:`str`): Path where to save the segmented frame.
            file_base (:obj:`str`): Base name of the file.
        """
        file_name = os.path.join(data_path,
                                 '{}-{}.png'.format(file_base, timestamp))
        img = Image.fromarray(self.as_cityscapes_palette())
        img.save(file_name)

    def visualize(self, pygame_display, timestamp=None):
        import pygame
        cityscapes_frame = self.as_cityscapes_palette()
        if timestamp is not None:
            pylot.utils.add_timestamp(cityscapes_frame, timestamp)
        image_np = np.transpose(cityscapes_frame, (1, 0, 2))
        pygame.surfarray.blit_array(pygame_display, image_np)
        pygame.display.flip()

    def draw_box(self, start_point, end_point, color, thickness=3):
        """Draw a colored box defined by start_point, end_point."""
        start = (int(start_point.x), int(start_point.y))
        end = (int(end_point.x), int(end_point.y))
        cv2.rectangle(self._frame, start, end, color, thickness)

    def draw_point(self, point, color, r=3):
        """Draws a colored point on the segmented frame."""
        cv2.circle(self._frame, (int(point.x), int(point.y)), r, color, -1)

    def draw_text(self, point, text, color=(255, 255, 255)):
        cv2.putText(self._frame,
                    text, (int(point.x), int(point.y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    thickness=1,
                    lineType=cv2.LINE_AA)

    def _get_traffic_sign_pixels(self):
        """Returns a frame with the traffic sign pixels set to True."""
        # Shape is height, width
        traffic_signs_frame = np.zeros(
            (self._frame.shape[0], self._frame.shape[1]), dtype=np.bool)
        # 12 is the key for TrafficSigns segmentation in CARLA.
        # Apply mask to only select traffic signs and traffic lights.
        traffic_signs_frame[np.where(
            np.logical_or(self._frame == 12, self._frame == 18))] = True
        return traffic_signs_frame

    def __repr__(self):
        return 'SegmentedFrame(encoding: {}, camera_setup: {}, frame: {})'.format(  # noqa: E501
            self.encoding, self.camera_setup, self._frame)

    def __str__(self):
        return 'SegmentedFrame(encoding: {}, camera_setup: {}, frame: {})'.format(  # noqa: E501
            self.encoding, self.camera_setup, self._frame)
