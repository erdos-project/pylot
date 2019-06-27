from matplotlib import pyplot as plt
import time

import carla

import pylot.utils
import pylot.simulation.messages
import pylot.simulation.utils
from pylot.simulation.utils import to_bgra_array, to_erdos_transform
from pylot.simulation.carla_utils import get_world


def on_camera_msg(carla_image):
    game_time = int(carla_image.timestamp * 1000)
    print("Received camera msg {}".format(game_time))
    global last_frame
    last_frame = pylot.utils.bgra_to_bgr(to_bgra_array(carla_image))
    plt.imshow(last_frame)
    plt.show()


def add_camera(world, transform, callback):
    camera_blueprint = world.get_blueprint_library().find(
        'sensor.camera.rgb')
    camera_blueprint.set_attribute('image_size_x', '800')
    camera_blueprint.set_attribute('image_size_y', '600')
    camera = world.spawn_actor(camera_blueprint, transform)
    # Register callback to be invoked when a new frame is received.
    camera.listen(callback)
    return camera


def explore_api(world, transform):
    hd_map = world.get_map()
    waypoint = hd_map.get_waypoint(
        transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Any  # Return if any type of lane
    )

    # Get left lane.
    left_w = waypoint.get_left_lane()
    # Get right lane.
    right_w = waypoint.get_right_lane()

#    Junction API is only available in the nightly? version.
#    print("Is junction {}".format(waypoint.is_junction))
#    print("Junction is {}".format(waypoint.junction_id))

    print("Is intersection {}".format(waypoint.is_intersection))
    print("Lane change {}".format(waypoint.lane_change))
    print("Lane type {}".format(waypoint.lane_type))
    print("Lane width {}".format(waypoint.lane_width))
    print("Lane id {}".format(waypoint.lane_id))
    print("Road id {}".format(waypoint.road_id))
    # Section id doesn't change as often as road id does.
    # Unclear what section_id means.
    print("Section id {}".format(waypoint.section_id))
    print("Left lane marking {}".format(waypoint.left_lane_marking))
    print("Right lane marking  {}".format(waypoint.right_lane_marking))
    # s starts from 0 in each road id.
    print("Frenet s  {}".format(waypoint.s))
    # Next receives distance in meters. S in reset whenever the road id
    # changes.
    print("Next waypoints {}".format(waypoint.next(1.0)))


def setup_world():
    # Connect to the Carla simulator.
    client, world = get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    world.apply_settings(settings)
    return world


def get_spawnpoint(world, index):
    spawn_points = world.get_map().get_spawn_points()
    return spawn_points[index]


def main():
    world = setup_world()
    transform = get_spawnpoint(world, 0)
    print('Spawning at {}'.format(to_erdos_transform(transform)))
    try:
        camera = add_camera(world, transform, on_camera_msg)

        # Move the spectactor view to the camera position.
        world.get_spectator().set_transform(transform)
        world.tick()

        explore_api(world, transform)
    finally:
        # Destroy the actors.
        camera.destroy()

if __name__ == '__main__':
    main()
