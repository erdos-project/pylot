import carla
import cv2
from lxml import etree
import math
import numpy as np
import random

counter = 0
camera_matrix, camera, world, vehicle, world_map = None, None, None, None, None
roads = {}


class Matrix(object):
    def __init__(self, location, rotation, width, height, fov=90):
        """ Initializes the transformation matrix given the location and the
        rotation of the given actor.

        Args:
            location: The location of the actor. (x, y, z)
            rotation: The rotation of the actor. (roll, pitch, yaw)
            width: The width of the image to project on.
            height: The height of the image to project on.
            fov: The fov of the camera.
        """
        self.location = location
        self.rotation = rotation
        self.width, self.height = width, height

        # Create the extrinsic matrix.
        matrix = np.matrix(np.identity(4))
        cy = math.cos(np.radians(rotation.yaw))
        sy = math.sin(np.radians(rotation.yaw))
        cr = math.cos(np.radians(rotation.roll))
        sr = math.sin(np.radians(rotation.roll))
        cp = math.cos(np.radians(rotation.pitch))
        sp = math.sin(np.radians(rotation.pitch))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = (cp * cy)
        matrix[0, 1] = (cy * sp * sr - sy * cr)
        matrix[0, 2] = -1.0 * (cy * sp * cr + sy * sr)
        matrix[1, 0] = (sy * cp)
        matrix[1, 1] = (sy * sp * sr + cy * cr)
        matrix[1, 2] = (cy * sr - sy * sp * cr)
        matrix[2, 0] = (sp)
        matrix[2, 1] = -1.0 * (cp * sr)
        matrix[2, 2] = (cp * cr)
        self.extrinsic_matrix = matrix

        # Create the intrinsic matrix.
        intrinsic_matrix = np.identity(3)
        intrinsic_matrix[0, 2] = width / 2.0
        intrinsic_matrix[1, 2] = height / 2.0
        intrinsic_matrix[0, 0] = intrinsic_matrix[
            1, 1] = width / (2.0 * math.tan(fov * math.pi / 360.0))
        self.intrinsic_matrix = intrinsic_matrix

    def transform_to_view(self, location):
        # Get the location of the object in the world.
        world_points = [[location.x], [location.y], [location.z], [1]]

        # Convert the points to the sensor coordinates.
        transformed_points = np.dot(np.linalg.inv(self.extrinsic_matrix),
                                    world_points)

        # Convert the points to an unreal space.
        unreal_points = np.concatenate([
            transformed_points[1, :], -transformed_points[2, :],
            transformed_points[0, :]
        ])

        # Convert to screen points.
        screen_points = np.transpose(
            np.dot(self.intrinsic_matrix, unreal_points))

        # Normalize the points
        x = screen_points[:, 0] / screen_points[:, 2]
        y = screen_points[:, 1] / screen_points[:, 2]
        z = screen_points[:, 2]
        return x, y, z


def getElevation(length, sectionElevation):
    """ This function gives the elevation from the start of the section to the
    given length using the third order polynomial coefficients derived from
    the section elevation.
    """
    a, b = sectionElevation.attrib['a'], sectionElevation.attrib['b']
    c, d = sectionElevation.attrib['c'], sectionElevation.attrib['d']
    a, b, c, d = np.float32([a, b, c, d])
    elevation = a + b * length + c * (length**2) + d * (length**3)
    return elevation


def getLineCoordinates(waypoint, roads):
    # Get the road information from the map.
    road = roads[waypoint.road_id]
    geometries = road.find('planView').findall('geometry')
    elevationProfile = road.find('elevationProfile')
    laneSections = road.find('lanes').findall('laneSection')
    s_index = waypoint.section_id

    geometry = geometries[s_index]
    sectionElevation = elevationProfile.getchildren()[s_index]

    # Calculate the starting point of the lane given the center of
    # the road and the lane ID.
    start_point = carla.Location(x=float(geometry.attrib['x']),
                                 y=-float(geometry.attrib['y']),
                                 z=getElevation(0, sectionElevation))
    # Find the section of the road that contains the lane where Carla tells
    # me the waypoint is.
    laneSection = laneSections[s_index]
    lane_id = 0
    if waypoint.lane_id > 0:
        laneSection = laneSection.find('left')
        lane_id = len(laneSection.getchildren()) - waypoint.lane_id - 1
        return None, None, None, None
    elif waypoint.lane_id < 0:
        laneSection = laneSection.find('right')
        lane_id = waypoint.lane_id
        lane = None
        for l in laneSection.getchildren():
            if int(l.attrib['id']) == waypoint.lane_id:
                lane = l
        laneOffset = carla.Location(y=-1 * lane_id *
                                    float(lane.find('width').attrib['a']))
        markerWidth = 0.0
        for marker in lane.findall('roadMark'):
            if marker.attrib.has_key('width'):
                markerWidth = float(marker.attrib['width'])
        markerOffset = carla.Location(y=markerWidth / 2.0)
        return start_point + laneOffset + markerOffset, None, None, None
    else:
        laneSection = laneSection.find('center')

    # Find the lane instance with the required lane id.
    lane = None
    for l in laneSection.getchildren():
        if int(l.attrib['id']) == waypoint.lane_id:
            lane = l

    # Find the offset of the lane from the center of the road given the
    # identifier and the width of the lanes.
    laneOffset = carla.Location(y=-1 * lane_id *
                                float(lane.find('width').attrib['a']))

    # Find the width of the markers so that we can figure out the two
    # starting points of the polygon.
    markerWidth = 0.0
    for marker in lane.findall('roadMark'):
        if marker.attrib.has_key('width'):
            markerWidth = float(marker.attrib['width'])
    markerOffset = carla.Location(y=markerWidth / 2.0)

    # Compute the two lower points of the polygon.
    marker_start_left = start_point + laneOffset + markerOffset
    marker_start_right = start_point + laneOffset - markerOffset

    return marker_start_left, marker_start_right, None, None

    # Calculate the end point of this section of the lane using the length
    # of the section and its heading.
    #tangent = np.float32(geometry.attrib['hdg'])
    #length = np.float32(geometry.attrib['length'])
    #end_point = carla.Location(
    #    x=start_point.x + (length * np.cos(tangent)),
    #    y=start_point.y + (length * np.sin(tangent)),
    #    z=getElevation(length, sectionElevation))


def process_images(msg):
    global counter, camera_matrix
    camera_matrix = Matrix(camera.get_transform().location,
                           camera.get_transform().rotation, 800, 600)
    waypoint = world_map.get_waypoint(vehicle.get_location()).next(10.0)[0]

    # Convert the BGRA image to BGR.
    image = np.frombuffer(msg.raw_data, dtype=np.dtype('uint8'))
    image = np.reshape(image, (msg.height, msg.width, 4))[:, :, :3]
    image = np.copy(image)

    road = roads[waypoint.road_id]
    geometries = road.find('planView').findall('geometry')
    s_index = waypoint.section_id

    # For the given geometry, draw the lane.
    # TODO (Sukrit) :: Ideally, we should look at the future, and also
    # show trajectories upto a fixed length. For now, let's just look at the
    # most recent geometry.
    if geometries[s_index].getchildren()[0].tag == 'line':
        wp = waypoint
        print("The road ID is {}, the section ID is {}, and the lane ID is {}".
              format(wp.road_id, wp.section_id, wp.lane_id))
        print("The waypoint location is {}, {}, {}".format(
            wp.transform.location.x, wp.transform.location.y,
            wp.transform.location.z))
        marker_start_left, _, _, _ = getLineCoordinates(waypoint, roads)
        if marker_start_left is not None:
            x, y, z = camera_matrix.transform_to_view(marker_start_left)
            if z > 0:
                cv2.circle(image, (int(x), int(y)),
                           10,
                           color=(255, 0, 0),
                           thickness=5)

        #print("The x, y value of the waypoint is {}, {}".format(
        #    waypoint.transform.location.x, waypoint.transform.location.y))
        #print("The initital x, y values of the segment are {}, {}".format(
        #    start_point.x, start_point.y))
        #print("The final x, y values of the segment are {}, {}".format(
        #    end_point.x, end_point.y))
    else:
        print("The geometry was {} which we do not support.".format(
            geometries[s_index].getchildren()[0].tag))

    cv2.imshow("Image", image)
    counter += 1


def spawn_driving_vehicle(client, world):
    """ This function spawns the driving vehicle and puts it into
    an autopilot mode.

    Args:
        client: The carla.Client instance representing the simulation to
          connect to.
        world: The world inside the current simulation.

    Returns:
        A carla.Actor instance representing the vehicle that was just spawned.
    """
    # Get the blueprint of the vehicle and set it to AutoPilot.
    vehicle_bp = random.choice(
        world.get_blueprint_library().filter('vehicle.*'))
    while not vehicle_bp.has_attribute('number_of_wheels') or not int(
            vehicle_bp.get_attribute('number_of_wheels')) == 4:
        vehicle_bp = random.choice(
            world.get_blueprint_library().filter('vehicle.*'))
    vehicle_bp.set_attribute('role_name', 'autopilot')

    # Get the spawn point of the vehicle.
    start_pose = random.choice(world.get_map().get_spawn_points())

    # Spawn the vehicle.
    batch = [
        carla.command.SpawnActor(vehicle_bp, start_pose).then(
            carla.command.SetAutopilot(carla.command.FutureActor, True))
    ]
    vehicle_id = client.apply_batch_sync(batch)[0].actor_id

    # Find the vehicle and return the carla.Actor instance.
    vehicle = None
    while vehicle == None:
        world.tick()
        vehicle = world.get_actors().find(vehicle_id)
    return vehicle


def spawn_rgb_camera(world, location, rotation, vehicle):
    """ This method spawns an RGB camera with the default parameters and the
    given location and rotation. It also attaches the camera to the given
    actor.

    Args:
        world: The world inside the current simulation.
        location: The carla.Location instance representing the location where
          the camera needs to be spawned with respect to the vehicle.
        rotation: The carla.Rotation instance representing the rotation of the
          spawned camera.
        vehicle: The carla.Actor instance to attach the camera to.

    Returns:
        An instance of the camera spawned in the world.
    """
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('sensor_tick', '1.0')
    transform = carla.Transform(location=location, rotation=rotation)
    return world.spawn_actor(camera_bp, transform, attach_to=vehicle)


def main():
    global vehicle, world, world_map, camera, roads

    # Connect to the CARLA instance.
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    world_map = world.get_map()

    # Generate the OpenDrive map.
    od_map = etree.fromstring(world_map.to_opendrive())
    for child in od_map.getchildren():
        if child.tag == 'road':
            roads[int(child.attrib['id'])] = child

    # Spawn the vehicle.
    vehicle = spawn_driving_vehicle(client, world)

    # Spawn the camera and register a function to listen to the images.
    camera_location = carla.Location(x=2.0, y=0.0, z=1.8)
    camera_rotation = carla.Rotation(roll=0, pitch=0, yaw=0)
    camera = spawn_rgb_camera(world, camera_location, camera_rotation, vehicle)

    # Create the camera matrix.
    camera.listen(process_images)

    return vehicle, camera, world


if __name__ == "__main__":
    vehicle, camera, world = main()
    try:
        while True:
            cv2.waitKey(0)
            pass
    except KeyboardInterrupt:
        # Destroy the actors.
        vehicle.destroy()
        camera.destroy()
