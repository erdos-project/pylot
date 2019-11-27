import numpy as np

# Semantic Labels
LABEL_2_PIXEL = {
    "unlabeled": (0, 0, 0),
    "building": (70, 70, 70),
    "fence": (190, 153, 153),
    "other": (250, 170, 160),
    "pedestrian": (220, 20, 60),
    "pole": (153, 153, 153),
    "road_line": (157, 234, 50),
    "road": (128, 64, 128),
    "sidewalk": (244, 35, 232),
    "vegetation": (107, 142, 35),
    "car": (0, 0, 255),
    "wall": (102, 102, 156),
    "traffic_sign": (220, 220, 0),
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


def get_traffic_sign_pixels(segmented_frame):
    """ Returns a frame with the traffic sign pixels set to True."""
    # Shape is height, width
    traffic_signs_frame = np.zeros((segmented_frame.shape[0],
                                    segmented_frame.shape[1]),
                                   dtype=np.bool)
    # 12 is the key for TrafficSigns segmentation in Carla.
    # Apply mask to only select traffic signs and traffic lights.
    traffic_signs_frame[np.where(segmented_frame == 12)] = True
    return traffic_signs_frame


def transform_to_cityscapes_palette(frame_array):
    """ Transforms a frame to the Carla cityscapes pallete."""
    result = np.zeros((frame_array.shape[0], frame_array.shape[1], 3))
    for key, value in CITYSCAPES_CLASSES.items():
        result[np.where(frame_array == key)] = value
    return result


def compute_semantic_iou(ground_frame, predicted_frame):
    """ Computes IoU for a segmented frame.

    Args:
        ground_frame: Ground segmented frame.
        predicted_frame: The frame for which to compute IoU.
    Returns:
        A tuple comprising of mIoU and a list of IoUs.
    """
    iou = {}
    for key, value in CITYSCAPES_CLASSES.items():
        #  Do not include None in the mIoU
        if key == 0:
            continue
        target = np.zeros((ground_frame.shape[0], ground_frame.shape[1], 3))
        prediction = np.zeros((ground_frame.shape[0],
                               ground_frame.shape[1],
                               3))
        target[np.where(ground_frame == value)] = 1
        prediction[np.where(predicted_frame == value)] = 1
        intersection = np.logical_and(target, prediction)
        union = np.logical_or(target, prediction)
        sum_intersection = np.sum(intersection)
        sum_union = np.sum(union)
        # Ignore non-existing classes.
        if sum_union > 0:
            iou[key] = float(sum_intersection) / float(sum_union)
    mean_iou = np.mean(iou.values())
    return (mean_iou, iou)


def compute_semantic_iou_from_masks(ground_masks, pred_masks):
    """ Computes IoU from per class image masks.

    Args:
        ground_masks: A dict of class key to ground frame mask.
        pred_masks: A dict of class key to predicted frame mask.
    Returns:
        A tuple comprising of mIoU and a list of IoUs.
    """
    iou = {}
    for i in range(1, len(CITYSCAPES_CLASSES)):
        intersection = np.logical_and(ground_masks[i], pred_masks[i])
        union = np.logical_or(ground_masks[i], pred_masks[i])
        sum_intersection = np.sum(intersection)
        sum_union = np.sum(union)

        # Ignore non-existing classes.
        if sum_union > 0:
            iou[i] = float(sum_intersection) / float(sum_union)
    mean_iou = np.mean(iou.values())
    return (mean_iou, iou)


def generate_masks(frame):
    """ Returns a mask image for each Cityscapes class."""
    masks = []
    for key, value in CITYSCAPES_CLASSES.items():
        mask = np.zeros((frame.shape[0], frame.shape[1]))
        mask[np.where(frame == key)] = 1
        masks.append(mask)
    return masks
