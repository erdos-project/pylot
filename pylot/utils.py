import cv2
import math
import numpy as np

from erdos.data_stream import DataStream

CENTER_CAMERA_NAME = 'front_rgb_camera'
LEFT_CAMERA_NAME = 'front_left_rgb_camera'
RIGHT_CAMERA_NAME = 'front_right_rgb_camera'
DEPTH_CAMERA_NAME = 'front_depth_camera'
FRONT_SEGMENTED_CAMERA_NAME = 'front_semantic_camera'
TOP_DOWN_SEGMENTED_CAMERA_NAME = 'top_down_semantic_camera'
TOP_DOWN_CAMERA_NAME = 'top_down_rgb_camera'

# Sensor streams
def is_camera_stream(stream):
    return (stream.get_label('sensor_type') == 'camera' and
            stream.get_label('camera_type') == 'sensor.camera.rgb')


def is_center_camera_stream(stream):
    return (stream.get_label('sensor_type') == 'camera' and
            stream.get_label('camera_type') == 'sensor.camera.rgb' and
            stream.name == CENTER_CAMERA_NAME)


def is_left_camera_stream(stream):
    return (stream.get_label('sensor_type') == 'camera' and
            stream.get_label('camera_type') == 'sensor.camera.rgb' and
            stream.name == LEFT_CAMERA_NAME)


def is_right_camera_stream(stream):
    return (stream.get_label('sensor_type') == 'camera' and
            stream.get_label('camera_type') == 'sensor.camera.rgb' and
            stream.name == RIGHT_CAMERA_NAME)


def is_top_down_camera_stream(stream):
    return (stream.get_label('sensor_type') == 'camera' and
            stream.get_label('camera_type') == 'sensor.camera.rgb' and
            stream.name == TOP_DOWN_CAMERA_NAME)


def is_depth_camera_stream(stream):
    return (stream.get_label('sensor_type') == 'camera' and
            stream.get_label('camera_type') == 'sensor.camera.depth')


def create_camera_stream(camera_setup, ground='true'):
    labels = {'sensor_type': 'camera',
              'camera_type': camera_setup.camera_type,
              'ground': ground}
    if camera_setup.camera_type == 'sensor.camera.semantic_segmentation':
        labels['segmented'] = 'true'
    return DataStream(name=camera_setup.name, labels=labels)


def is_lidar_stream(stream):
    return stream.get_label('sensor_type') == 'sensor.lidar.ray_cast'


def create_lidar_stream(lidar_setup):
    return DataStream(name=lidar_setup.name,
                      labels={'sensor_type': lidar_setup.lidar_type})


def is_imu_stream(stream):
    return stream.get_label('sensor_type') == 'sensor.other.imu'


def create_imu_stream(imu_setup):
    return DataStream(name=imu_setup.name,
                      labels={'sensor_type': imu_setup.imu_type})


# Ground streams
def is_ground_segmented_camera_stream(stream):
    return (stream.get_label('sensor_type') == 'camera' and
            stream.get_label('camera_type') == 'sensor.camera.semantic_segmentation' and
            stream.get_label('ground') == 'true')


def create_vehicle_id_stream():
    return DataStream(name='vehicle_id_stream')


def is_ground_vehicle_id_stream(stream):
    return stream.name == 'vehicle_id_stream'


def create_ground_pedestrians_stream():
    return DataStream(name='pedestrians')


def is_ground_pedestrians_stream(stream):
    return stream.name == 'pedestrians'


def create_ground_vehicles_stream():
    return DataStream(name='vehicles')


def is_ground_vehicles_stream(stream):
    return stream.name == 'vehicles'


def create_ground_traffic_lights_stream():
    return DataStream(name='traffic_lights')


def is_ground_traffic_lights_stream(stream):
    return stream.name == 'traffic_lights'


def create_ground_speed_limit_signs_stream():
    return DataStream(name='speed_limit_signs')


def is_ground_speed_limit_signs_stream(stream):
    return stream.name == 'speed_limit_signs'


def create_ground_stop_signs_stream():
    return DataStream(name='stop_signs')


def is_ground_stop_signs_stream(stream):
    return stream.name == 'stop_signs'

def create_ground_tracking_stream(name):
    return DataStream(name=name,
                      labels={'tracking': 'true'})

def is_ground_tracking_stream(stream):
    return stream.get_label('tracking') == 'true'

def create_linear_prediction_stream(name):
    return DataStream(name=name,
                      labels={'prediction': 'true'})

def is_prediction_stream(stream):
    return stream.get_label('prediction') == 'true'

# ERDOS streams
def create_segmented_camera_stream(name):
    return DataStream(name=name,
                      labels={'segmented': 'true'})


def is_segmented_camera_stream(stream):
    return stream.get_label('segmented') == 'true'


def is_front_segmented_camera_stream(stream):
    return stream.name == FRONT_SEGMENTED_CAMERA_NAME


def is_top_down_segmented_camera_stream(stream):
    return stream.name == TOP_DOWN_SEGMENTED_CAMERA_NAME


def is_non_ground_segmented_camera_stream(stream):
    return (stream.get_label('segmented') == 'true' and
            stream.get_label('ground') != 'true')


def create_obstacles_stream(name):
    return DataStream(name=name, labels={'obstacles': 'true'})


def is_obstacles_stream(stream):
    return stream.get_label('obstacles') == 'true'


def create_traffic_lights_stream(name):
    return DataStream(name=name, labels={'traffic_lights': 'true'})


def is_traffic_lights_stream(stream):
    return stream.get_label('traffic_lights') == 'true'


def create_depth_estimation_stream(name):
    return DataStream(name=name, labels={'depth_estiomation': 'true'})


def is_depth_estimation_stream(stream):
    return stream.get_label('depth_estimation') == 'true'


def create_fusion_stream(name):
    return DataStream(name=name, labels={'fusion_output': 'true'})


def is_fusion_stream(stream):
    return stream.get_label('fusion_output') == 'true'


def create_control_stream():
    # XXX(ionel): HACK! We set no_watermark to avoid closing the cycle in
    # the data-flow.
    return DataStream(name='control_stream',
                      labels={'no_watermark': 'true'})


def is_control_stream(stream):
    return stream.name == 'control_stream'


def create_waypoints_stream():
    return DataStream(name='waypoints')


def is_waypoints_stream(stream):
    return stream.name == 'waypoints'


def is_tracking_stream(stream):
    return stream.get_label('tracking') == 'true'


def create_detected_lane_stream(name):
    return DataStream(name=name,
                      labels={'detected_lanes': 'true'})


def is_detected_lane_stream(stream):
    return stream.get_label('detected_lanes') == 'true'


def is_global_trajectory_stream(stream):
    return (stream.get_label('global') == 'true' and
            stream.get_label('waypoints') == 'true')


def is_open_drive_stream(stream):
    return stream.name == 'open_drive_stream'


def create_can_bus_stream():
    return DataStream(name='can_bus')


def is_can_bus_stream(stream):
    return stream.name == 'can_bus'

def create_imu_stream():
    return DataStream(name='imu')

def is_imu_stream(stream):
    return stream.get_label('sensor_type') == 'sensor.other.imu'



def add_timestamp(timestamp, image_np):
    txt_font = cv2.FONT_HERSHEY_SIMPLEX
    timestamp_txt = '{}'.format(timestamp)
    # Put timestamp text.
    cv2.putText(image_np, timestamp_txt, (5, 15), txt_font, 0.5,
                (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)


def bgra_to_bgr(image_np):
    return image_np[:, :, :3]


def bgra_to_rgb(image_np):
    image_np = image_np[:, :, :3]
    image_np = image_np[:, :, ::-1]


def bgr_to_rgb(image_np):
    return image_np[:, :, ::-1]


def rgb_to_bgr(image_np):
    return image_np[:, :, ::-1]


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
    target_vector = np.array([target_loc.x - cur_loc.x,
                              target_loc.y - cur_loc.y])
    norm_target = np.linalg.norm(target_vector)

    forward_vector = np.array([math.cos(math.radians(orientation)),
                               math.sin(math.radians(orientation))])
    d_angle = math.degrees(
        math.acos(np.dot(forward_vector, target_vector) / norm_target))

    return (norm_target, d_angle)


def is_within_distance_ahead(
        cur_loc, dst_loc, orientation, max_distance):
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
    target_vector = np.array([dst_loc.x - cur_loc.x,
                              dst_loc.y - cur_loc.y])
    norm_dst = np.linalg.norm(target_vector)
    # Return if the vector is too small.
    if norm_dst < 0.001:
        return True
    if norm_dst > max_distance:
        return False
    forward_vector = np.array(
        [math.cos(math.radians(orientation)),
         math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(
        np.dot(forward_vector, target_vector) / norm_dst))
    return d_angle < 90.0
