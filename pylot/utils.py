import cv2
from erdos.data_stream import DataStream


# Sensor streams
def is_camera_stream(stream):
    return (stream.get_label('sensor_type') == 'camera' and
            stream.get_label('camera_type') == 'sensor.camera.rgb')


def is_depth_camera_stream(stream):
    return (stream.get_label('sensor_type') == 'camera' and
            stream.get_label('camera_type') == 'sensor.camera.depth')


def create_camera_stream(camera_stream):
    return DataStream(name=camera_stream.name,
                      labels={'sensor_type': 'camera',
                              'camera_type': camera_stream.type})


def is_lidar_stream(stream):
    return stream.get_label('sensor_type') == 'sensor.lidar.ray_cast'


def create_lidar_stream(lidar_setup):
    return DataStream(name=lidar_setup.name,
                      labels={'sensor_type': lidar_setup.type})


# Ground streams
def is_ground_segmented_camera_stream(stream):
    return (stream.get_label('sensor_type') == 'camera' and
            stream.get_label('camera_type') == 'sensor.camera.semantic_segmentation')


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


def create_ground_traffic_signs_stream():
    return DataStream(name='traffic_signs')


def is_ground_traffic_signs_stream(stream):
    return stream.name == 'traffic_signs'


# ERDOS streams
def create_segmented_camera_stream(name):
    return DataStream(name=name,
                      labels={'segmented': 'true'})


def is_segmented_camera_stream(stream):
    return stream.get_label('segmented') == 'true'


def create_obstacles_stream(name):
    return DataStream(name=name, labels={'obstacles': 'true'})


def is_obstacles_stream(stream):
    return stream.get_label('obstacles') == 'true'


def create_traffic_lights_stream(name):
    return DataStream(name=name, labels={'traffic_lights': 'true'})


def is_traffic_lights_stream(stream):
    return stream.get_label('traffic_lights') == 'true'


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
