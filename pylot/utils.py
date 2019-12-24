import cv2
import logging
import math
import numpy as np
import os
import time


def add_timestamp(timestamp, image_np):
    txt_font = cv2.FONT_HERSHEY_SIMPLEX
    timestamp_txt = '{}'.format(timestamp)
    # Put timestamp text.
    cv2.putText(image_np,
                timestamp_txt, (5, 15),
                txt_font,
                0.5, (0, 0, 0),
                thickness=1,
                lineType=cv2.LINE_AA)


def bgra_to_bgr(image_np):
    return image_np[:, :, :3]


def bgra_to_rgb(image_np):
    image_np = image_np[:, :, :3]
    image_np = image_np[:, :, ::-1]


def bgr_to_rgb(image_np):
    return image_np[:, :, ::-1]


def rgb_to_bgr(image_np):
    return image_np[:, :, ::-1]


def time_epoch_ms():
    return int(time.time() * 1000)


def compute_magnitude_angle(target_loc, cur_loc, orientation):
    """
    Computes relative angle and distance between a target and a current
    location.

    Args:
        target_loc: Location of the target.
        cur_loc: Location of the reference object.
        orientation: Orientation of the reference object

    Returns:
        Tuple of distance to the target and the angle
    """
    target_vector = np.array(
        [target_loc.x - cur_loc.x, target_loc.y - cur_loc.y])
    norm_target = np.linalg.norm(target_vector)

    forward_vector = np.array([
        math.cos(math.radians(orientation)),
        math.sin(math.radians(orientation))
    ])
    d_angle = math.degrees(
        math.acos(np.dot(forward_vector, target_vector) / norm_target))

    return (norm_target, d_angle)


def is_within_distance_ahead(cur_loc, dst_loc, orientation, max_distance):
    """
    Check if a location is within a distance in a given orientation.

    Args:
        cur_loc: The current location.
        dst_loc: The location to compute distance for.
        orientation: Orientation of the reference object.
        max_distance: Maximum allowed distance.
    Returns:
        True if other location is within max_distance.
    """
    target_vector = np.array([dst_loc.x - cur_loc.x, dst_loc.y - cur_loc.y])
    norm_dst = np.linalg.norm(target_vector)
    # Return if the vector is too small.
    if norm_dst < 0.001:
        return True
    if norm_dst > max_distance:
        return False
    forward_vector = np.array([
        math.cos(math.radians(orientation)),
        math.sin(math.radians(orientation))
    ])
    d_angle = math.degrees(
        math.acos(np.dot(forward_vector, target_vector) / norm_dst))
    return d_angle < 90.0


def set_tf_loglevel(level):
    """ To be used to suppress TensorFlow logging."""
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)
