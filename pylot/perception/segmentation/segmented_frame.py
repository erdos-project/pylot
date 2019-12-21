import cv2
import numpy as np
import os
import PIL.Image as Image
from skimage import measure

from pylot.utils import add_timestamp
from pylot.simulation.utils import to_bgra_array

# Semantic Labels
CITYSCAPES_LABELS = {
    0: "unlabeled",
    1: "building",
    2: "fence",
    3: "other",
    4: "pedestrian",
    5: "pole",
    6: "road_line",
    7: "road",
    8: "sidewalk",
    9: "vegetation",
    10: "car",
    11: "wall",
    12: "traffic_sign",
}

# Cityscapes palette.
CITYSCAPES_CLASSES = {
    0: [0, 0, 0],         # None
    1: [70, 70, 70],      # Buildings
    2: [190, 153, 153],   # Fences
    3: [72, 0, 90],       # Other
    4: [220, 20, 60],     # Pedestrians
    5: [153, 153, 153],   # Poles
    6: [157, 234, 50],    # RoadLines
    7: [128, 64, 128],    # Roads
    8: [244, 35, 232],    # Sidewalks
    9: [107, 142, 35],    # Vegetation
    10: [0, 0, 255],      # Vehicles
    11: [102, 102, 156],  # Walls
    12: [220, 220, 0]     # TrafficSigns
}
# XXX(ionel): Note! These Carla cityscapes classes do not cover all
# the classes from CITYSCAPES. Hence, we can't compare segmentation
# outputs to ground truth.


class SegmentedFrame(object):
    """ Stores a semantically segmented frame."""

    def __init__(self, frame, encoding='carla'):
        if encoding == 'carla':
            # Convert the array containing CARLA semantic segmentation labels
            # to a 2D array containing the label of each pixel.
            self._frame = to_bgra_array(frame)[:, :, 2]
        elif encoding == 'cityscapes':
            raise ValueError('Transformation from cityscapes to carla encoding'
                             ' is not yet implemented!')
        else:
            raise ValueError(
                'Unexpected encoding {} for segmented frame'.format(encoding))
        self._class_masks = None

    def as_cityscapes_palette(self):
        """ Transforms the frame to the Carla cityscapes pallete.

        Note: this conversion is slow.
        """
        result = np.zeros((self._frame.shape[0], self._frame.shape[1], 3),
                          dtype=np.uint8)
        for key, value in CITYSCAPES_CLASSES.items():
            result[np.where(self._frame == key)] = value
        return result

    def as_numpy_array(self):
        return self._frame

    def get_traffic_sign_bounding_boxes(self, min_width=2, min_height=3):
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
                bboxes.append((x_min, x_max, y_min, y_max))
        return bboxes

    def get_per_class_masks(self):
        """ Build a cache of class key to frame mask."""
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
        """ Computes IoU for a segmented frame.

        Args:
            other_frame: The frame for which to compute IoU.
        Returns:
            A tuple comprising of mIoU and a list of IoUs.
        """
        iou = {}
        for key, value in CITYSCAPES_CLASSES.items():
            #  Do not include None in the mIoU
            if key == 0:
                continue
            target = np.zeros((self._frame.shape[0], self._frame.shape[1], 3))
            prediction = np.zeros((self._frame.shape[0],
                                   self._frame.shape[1],
                                   3))
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
        """ Computes IoU for a segmented frame.

        Computes IoU from per class image masks. This method is the fastest if
        the frames already have precomputed masks.

        Args:
            other_frame: The frame for which to compute IoU.
        Returns:
            A tuple comprising of mIoU and a list of IoUs.
        """
        masks = self.get_per_class_masks()
        other_masks = other_frame.get_per_class_masks()
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
        masks = self.get_per_class_masks()
        for k, v in CITYSCAPES_LABELS.items():
            file_name = os.path.join(
                data_path, '{}-{}.png'.format(v, timestamp.coordinates[0]))
            img = Image.fromarray(masks[k])
            img = img.convert('RGB')
            img.save(file_name)

    def save(self, file_name):
        img = Image.fromarray(self.as_cityscapes_palette())
        img.save(file_name)

    def visualize(self, window_name, timestamp=None):
        cityscapes_frame = self.as_cityscapes_palette()
        if timestamp is not None:
            add_timestamp(timestamp, cityscapes_frame)
        cv2.imshow(window_name, cityscapes_frame)
        cv2.waitKey(1)

    def _get_traffic_sign_pixels(self):
        """ Returns a frame with the traffic sign pixels set to True."""
        # Shape is height, width
        traffic_signs_frame = np.zeros((self._frame.shape[0],
                                        self._frame.shape[1]),
                                       dtype=np.bool)
        # 12 is the key for TrafficSigns segmentation in Carla.
        # Apply mask to only select traffic signs and traffic lights.
        traffic_signs_frame[np.where(self._frame == 12)] = True
        return traffic_signs_frame
